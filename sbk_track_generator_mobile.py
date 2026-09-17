import math
from dataclasses import dataclass

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
from streamlit_folium import st_folium


st.set_page_config(
    page_title="SBK Spårgenerator",
    page_icon="🐕",
    layout="centered",
)

CACHE_TTL = 900

# Track geometry preferences
MAX_TURN_DEG = 90.0
DEFAULT_PREFERRED_LEG_SEPARATION_M = 25.0
DEFAULT_MIN_LEG_SEPARATION_M = 12.0
DEFAULT_EXIT_CLEARANCE_M = 25.0
DEFAULT_START_CLEARANCE_M = 40.0

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


@dataclass
class RuleProfile:
    name: str
    target_length: int
    angles: int
    objects: int
    min_leg: int
    max_leg: int
    boundary_margin: int


RULE_PROFILES = {
    "Appell": RuleProfile(
        "Appell", 300, 2, 3, 60, 100, 10
    ),
    "Lägre": RuleProfile(
        "Lägre", 1000, 5, 8, 100, 200, 10
    ),
    "Högre": RuleProfile(
        "Högre", 1200, 6, 8, 100, 200, 10
    ),
    "Elit": RuleProfile(
        "Elit", 1500, 7, 8, 100, 200, 10
    ),
}


for key, value in {
    "location": None,
    "drawn_area": None,
    "drawn_area_geojson": None,
    "drawn_area_key": None,
    "preferred_start": None,
    "preferred_start_geojson": None,
    "preferred_exit": None,
    "preferred_exit_geojson": None,
    "map_revision": 0,
    "osm_layers": None,
    "osm_area_key": None,
    "osm_count": 0,
    "osm_reduced": False,
    "candidates": None,
    "candidate_key": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def transformers(lat, lon):
    epsg = utm_epsg(lat, lon)
    to_xy = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    to_ll = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )
    return to_xy, to_ll


def xy_ll(x, y, to_ll):
    lon, lat = to_ll.transform(x, y)
    return lat, lon


def parse_google_maps_pin(value):
    if not value:
        return None

    import re

    patterns = [
        r"@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)",
        r"[?&]q=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, value.strip())
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon

    return None


def folium_polygon_to_shapely(coords, to_xy):
    """Convert [(lat, lon), ...] to a projected Shapely Polygon."""
    if not coords or len(coords) < 3:
        return None

    xy = [to_xy.transform(lon, lat) for lat, lon in coords]

    if xy[0] != xy[-1]:
        xy.append(xy[0])

    polygon = Polygon(xy)

    if polygon.is_empty or polygon.area <= 0:
        return None

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        return None

    if isinstance(polygon, Polygon):
        return polygon

    if isinstance(polygon, MultiPolygon):
        return max(polygon.geoms, key=lambda p: p.area)

    return None



def polygon_leaflet_bounds(area, to_ll):
    """Return geographic bounds suitable for Folium fit_bounds()."""
    if area is None or area.is_empty:
        return None

    minx, miny, maxx, maxy = area.bounds
    corners = [
        xy_ll(minx, miny, to_ll),
        xy_ll(minx, maxy, to_ll),
        xy_ll(maxx, miny, to_ll),
        xy_ll(maxx, maxy, to_ll),
    ]
    lats = [p[0] for p in corners]
    lons = [p[1] for p in corners]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

def sample_point_in_area(area, rng, max_attempts=2000):
    """Return a random point inside the selected Shapely area."""
    if area is None or area.is_empty:
        return None

    minx, miny, maxx, maxy = area.bounds

    for _ in range(max_attempts):
        point = Point(
            rng.uniform(minx, maxx),
            rng.uniform(miny, maxy),
        )
        if area.covers(point):
            return point

    return None


def choose_start_point(
    area, rng, preferred_start=None, radius_m=35.0, prepared_area=None
):
    """Choose a start point, preferring the user's selected map position."""
    if preferred_start is None:
        return sample_point_in_area(area, rng)

    area_covers = prepared_area.covers if prepared_area is not None else area.covers

    if not area_covers(preferred_start):
        return None

    # Try positions near the selected point so generation is not forced to
    # fail merely because the exact point gives an impossible first leg.
    for _ in range(80):
        distance = float(rng.uniform(0.0, radius_m))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        point = Point(
            preferred_start.x + distance * math.cos(angle),
            preferred_start.y + distance * math.sin(angle),
        )
        if area_covers(point):
            return point

    return preferred_start


def segment_allowed(
    segment,
    area,
    hard_geometry=None,
    prepared_area=None,
    prepared_hard=None,
):
    """Fast staged test for polygon containment and mapped obstacles."""
    if segment is None or segment.is_empty or area is None or area.is_empty:
        return False

    # Prepared geometries make the repeated covers/intersects predicates much
    # cheaper during candidate generation. Fall back to ordinary predicates
    # when no prepared geometry was supplied.
    if prepared_area is not None:
        if not prepared_area.covers(segment):
            return False
    elif not area.covers(segment):
        return False

    if hard_geometry is not None and not hard_geometry.is_empty:
        if prepared_hard is not None:
            if prepared_hard.intersects(segment):
                return False
        elif segment.intersects(hard_geometry):
            return False

    return True


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def overpass_query(south, west, north, east):
    import random
    import requests

    bbox = f"{south},{west},{north},{east}"

    full_query = f"""
    [out:json][timeout:50];
    (
      nwr["landuse"~"^(forest|meadow|grass|farmland|residential|industrial|commercial|quarry)$"]({bbox});
      nwr["natural"~"^(wood|scrub|water)$"]({bbox});
      nwr["waterway"="riverbank"]({bbox});
      way["highway"]({bbox});
      nwr["building"]({bbox});
      way["barrier"]({bbox});
    );
    out geom;
    """

    # If public Overpass instances are overloaded, this smaller query still
    # supplies the geometry most important for safe track generation.
    fallback_query = f"""
    [out:json][timeout:70];
    (
      nwr["natural"~"^(wood|water)$"]({bbox});
      nwr["landuse"~"^(forest|residential|industrial|commercial|quarry)$"]({bbox});
      way["highway"]({bbox});
      nwr["building"]({bbox});
      way["barrier"]({bbox});
    );
    out geom;
    """

    endpoints = list(OVERPASS_ENDPOINTS)
    random.shuffle(endpoints)
    errors = []

    session = requests.Session()
    session.headers.update({"User-Agent": "SBK-Spargenerator/1.0"})

    # Try all servers with the complete dataset first. GET is useful here
    # because public/proxy caches can serve identical bbox queries quickly.
    for endpoint in endpoints:
        try:
            response = session.get(
                endpoint,
                params={"data": full_query},
                timeout=(8, 32),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")

    # Retry with the reduced safety-oriented query and a more generous read
    # timeout. This avoids failing the whole application just because one
    # optional terrain category is expensive to retrieve.
    for endpoint in endpoints:
        try:
            response = session.post(
                endpoint,
                data={"data": fallback_query},
                timeout=(10, 70),
            )
            response.raise_for_status()
            result = response.json()
            result["_reduced_osm_query"] = True
            return result
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")

    raise RuntimeError(
        "Alla Overpass-servrar svarade för långsamt eller med fel. "
        "Försök igen om en liten stund. Senaste fel: " + errors[-1]
    )


def elements_to_geometries(data, to_xy):
    polygons = []
    lines = []

    for element in data.get("elements", []):
        geometry = element.get("geometry")
        if not geometry:
            continue

        coords = [
            to_xy.transform(point["lon"], point["lat"])
            for point in geometry
        ]

        if len(coords) < 2:
            continue

        tags = element.get("tags", {})

        if len(coords) >= 4 and coords[0] == coords[-1]:
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append((polygon, tags))
        else:
            lines.append((LineString(coords), tags))

    return polygons, lines


def build_osm_layers(data, to_xy):
    polygons, lines = elements_to_geometries(data, to_xy)

    forest = []
    soft = []
    hard_area = []
    water = []
    roads = []
    paths = []
    barriers = []

    for geom, tags in polygons:
        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")
        building = tags.get("building")

        if landuse == "forest" or natural == "wood":
            forest.append(geom)
        elif natural == "scrub":
            soft.append(geom)
        elif landuse in {"meadow", "grass", "farmland"}:
            soft.append(geom)
        elif natural == "water" or tags.get("waterway") == "riverbank":
            water.append(geom)
        elif (
            landuse in {
                "residential", "industrial",
                "commercial", "quarry"
            }
            or building is not None
        ):
            hard_area.append(geom)

    for geom, tags in lines:
        highway = tags.get("highway")
        barrier = tags.get("barrier")

        if barrier:
            barriers.append(geom)

        if highway:
            if highway in {
                "footway", "path", "track", "bridleway",
                "cycleway", "steps", "pedestrian"
            }:
                paths.append(geom)
            else:
                roads.append(geom)

    forest_union = unary_union(forest) if forest else Polygon()
    soft_union = unary_union(soft) if soft else Polygon()
    water_union = unary_union(water) if water else Polygon()

    road_union = (
        unary_union([g.buffer(3.0) for g in roads])
        if roads else Polygon()
    )

    barrier_union = (
        unary_union([g.buffer(1.5) for g in barriers])
        if barriers else Polygon()
    )

    hard_area_union = (
        unary_union(hard_area)
        if hard_area else Polygon()
    )

    hard_union = unary_union([
        hard_area_union,
        water_union,
        road_union,
        barrier_union,
    ])
    if not hard_union.is_empty:
        hard_union = hard_union.simplify(0.35, preserve_topology=True)

    return {
        "forest": forest_union,
        "soft": soft_union,
        "water": water_union,
        "roads": road_union,
        # Paths are not used by the current generator, so avoid an expensive
        # unary_union here.
        "paths": Polygon(),
        "barriers": barrier_union,
        "hard": hard_union,
    }


def random_segment_lengths(
    target_length,
    num_angles,
    rng,
    min_leg=60,
    max_leg=100,
):
    """Generate leg lengths whose sum is target_length and respect bounds."""
    n_legs = max(2, int(num_angles) + 1)
    target = float(target_length)
    low = float(max(1, min_leg))
    high = float(max(low, max_leg))

    if target < n_legs * low or target > n_legs * high:
        return None

    # Start at the minimum and distribute the remaining length randomly while
    # never exceeding max_leg. This avoids the old clip-then-rescale behavior,
    # which could silently violate both bounds.
    lengths = np.full(n_legs, low, dtype=float)
    remaining = target - n_legs * low
    capacities = np.full(n_legs, high - low, dtype=float)

    while remaining > 1e-9:
        available = np.flatnonzero(capacities > 1e-9)
        if available.size == 0:
            return None

        weights = rng.uniform(0.75, 1.25, size=available.size)
        shares = remaining * weights / weights.sum()
        added_total = 0.0

        for idx, share in zip(available, shares):
            add = min(float(share), capacities[idx])
            lengths[idx] += add
            capacities[idx] -= add
            added_total += add

        if added_total <= 1e-12:
            return None
        remaining -= added_total

    # Correct tiny floating-point residue on the final leg with capacity.
    residue = target - float(lengths.sum())
    if abs(residue) > 1e-9:
        for idx in range(n_legs - 1, -1, -1):
            candidate = lengths[idx] + residue
            if low - 1e-9 <= candidate <= high + 1e-9:
                lengths[idx] = candidate
                break

    return lengths.tolist()


def generate_candidate(
    area,
    forest,
    soft,
    hard_geometry,
    target_length,
    num_angles,
    num_objects,
    rng,
    min_leg=60,
    max_leg=100,
    preferred_start=None,
    start_radius_m=35.0,
    preferred_exit=None,
    preferred_separation_m=25.0,
    minimum_separation_m=12.0,
    exit_clearance_m=25.0,
    start_clearance_m=40.0,
    appell_mode=False,
    prepared_area=None,
    prepared_hard=None,
):
    """Generate an open, non-self-intersecting track with clear access."""
    if area is None or area.is_empty:
        return None

    start = choose_start_point(
        area,
        rng,
        preferred_start=preferred_start,
        radius_m=start_radius_m,
        prepared_area=prepared_area,
    )
    if start is None:
        return None

    points = [start]
    leg_lengths = random_segment_lengths(
        target_length, num_angles, rng,
        min_leg=min_leg, max_leg=max_leg,
    )
    if leg_lengths is None:
        return None

    current = start
    previous_heading = None
    accepted_segments = []
    signed_turns = []
    track_length_so_far = 0.0
    minimum_observed_separation = float("inf")
    start_zone = start.buffer(float(start_clearance_m))

    for leg_length in leg_lengths:
        accepted = False

        for _ in range(80):
            proposed_turn = None
            if previous_heading is None:
                heading = float(rng.uniform(0.0, 360.0))
            else:
                if appell_mode:
                    # Appell: each of the two angles must be 90° ±5°.
                    proposed_turn = float(rng.uniform(85.0, 95.0))
                else:
                    proposed_turn = float(rng.uniform(35.0, MAX_TURN_DEG))

                if rng.random() < 0.5:
                    proposed_turn = -proposed_turn

                # Consecutive turns may continue in the same direction.
                # When they do, the leg length is adapted below so the path
                # expands outward instead of curling tightly around its start.
                heading = (previous_heading + proposed_turn) % 360.0

            # Adapt leg length when several turns continue in the same
            # direction. The more the path has turned, and the closer the
            # current point is to the start, the more strongly we extend the
            # next leg outward. This allows >180° cumulative turning without
            # producing a tight spiral around the start.
            adjusted_leg_length = float(leg_length)

            if proposed_turn is not None:
                same_direction_total = abs(proposed_turn)
                same_direction_count = 1

                for old_turn in reversed(signed_turns):
                    if old_turn * proposed_turn <= 0:
                        break
                    same_direction_total += abs(old_turn)
                    same_direction_count += 1

                if same_direction_total > 120.0:
                    radial_distance = current.distance(start)

                    # Desired radial scale grows as cumulative turning grows.
                    # This is deliberately a preference through leg extension;
                    # all ordinary area/obstacle/spacing checks still apply.
                    turn_factor = min(
                        max((same_direction_total - 120.0) / 180.0, 0.0),
                        1.5,
                    )
                    desired_radius = (
                        float(start_clearance_m)
                        + 0.18 * track_length_so_far
                        + 35.0 * turn_factor
                    )

                    if radial_distance < desired_radius:
                        extension = min(
                            desired_radius - radial_distance,
                            float(max_leg) * 0.75,
                        )
                        adjusted_leg_length += extension

                    # Successive same-direction turns should generally not
                    # become progressively shorter, which is a common cause
                    # of inward curling.
                    if accepted_segments:
                        previous_length = accepted_segments[-1].length
                        growth = 1.0 + min(
                            0.08 * max(same_direction_count - 1, 0),
                            0.30,
                        )
                        adjusted_leg_length = max(
                            adjusted_leg_length,
                            previous_length * growth,
                        )

            # Keep adaptive legs within a sensible upper bound. This bound is
            # intentionally above the normal max_leg because the adjustment
            # is specifically used to open a long same-direction sequence.
            adjusted_leg_length = min(
                adjusted_leg_length,
                max(float(max_leg) * 1.6, float(leg_length)),
            )

            angle_rad = math.radians(heading)
            candidate_point = Point(
                current.x + adjusted_leg_length * math.cos(angle_rad),
                current.y + adjusted_leg_length * math.sin(angle_rad),
            )
            segment = LineString([
                (current.x, current.y),
                (candidate_point.x, candidate_point.y),
            ])

            if not segment_allowed(
                segment,
                area,
                hard_geometry,
                prepared_area=prepared_area,
                prepared_hard=prepared_hard,
            ):
                continue

            non_adjacent_segments = accepted_segments[:-1]

            # Cheap bounding-box rejection before exact GEOS predicates. Most
            # older legs are nowhere near a proposed segment, especially on
            # long Högre/Elit tracks.
            sx0, sy0, sx1, sy1 = segment.bounds
            nearby_segments = []
            margin = float(minimum_separation_m)
            for old in non_adjacent_segments:
                ox0, oy0, ox1, oy1 = old.bounds
                if not (
                    sx1 + margin < ox0
                    or ox1 + margin < sx0
                    or sy1 + margin < oy0
                    or oy1 + margin < sy0
                ):
                    nearby_segments.append(old)

            if any(segment.intersects(old) for old in nearby_segments):
                continue

            # After the first two legs, do not let the track return close to
            # its own start. This leaves an open access/escape zone instead of
            # wrapping later legs around the starting position.
            if len(accepted_segments) >= 2:
                if segment.intersects(start_zone):
                    continue

            # Exact distance is only useful for legs close enough to affect
            # either the hard floor or the preferred-separation score.
            score_margin = max(
                float(minimum_separation_m),
                float(preferred_separation_m),
            )
            distance_candidates = []
            for old in non_adjacent_segments:
                ox0, oy0, ox1, oy1 = old.bounds
                if not (
                    sx1 + score_margin < ox0
                    or ox1 + score_margin < sx0
                    or sy1 + score_margin < oy0
                    or oy1 + score_margin < sy0
                ):
                    distance_candidates.append(old)

            distances = [segment.distance(old) for old in distance_candidates]
            closest = min(distances) if distances else float("inf")

            # Absolute floor: never accept legs closer than this.
            if closest < minimum_separation_m:
                continue

            points.append(candidate_point)
            accepted_segments.append(segment)
            track_length_so_far += segment.length
            current = candidate_point
            previous_heading = heading
            minimum_observed_separation = min(
                minimum_observed_separation, closest
            )
            accepted = True
            break

        if not accepted:
            return None

    track = LineString([(point.x, point.y) for point in points])

    # Adaptive anti-curling can change the nominal total length. Keep only
    # candidates reasonably close to the requested length; scoring below
    # further prefers the closest ones.
    length_error_ratio = abs(track.length - target_length) / max(
        float(target_length), 1.0
    )
    if length_error_ratio > 0.15:
        return None

    if not area.covers(track):
        return None
    if hard_geometry is not None and not hard_geometry.is_empty:
        if prepared_hard is not None:
            if prepared_hard.intersects(track):
                return None
        elif track.intersects(hard_geometry):
            return None
    if not track.is_simple:
        return None

    if minimum_observed_separation == float("inf"):
        minimum_observed_separation = preferred_separation_m

    start_distance = (
        start.distance(preferred_start)
        if preferred_start is not None else 0.0
    )

    # Reward tracks that progress away from the start rather than folding
    # back around it. This is a preference rather than a hard endpoint rule.
    end_point = Point(track.coords[-1])
    start_to_end_distance = start.distance(end_point)
    openness_ratio = start_to_end_distance / max(track.length, 1.0)

    exit_distance = 0.0
    exit_clearance = float("inf")
    exit_segment = None

    if preferred_exit is not None:
        exit_distance = end_point.distance(preferred_exit)
        exit_segment = LineString([
            (end_point.x, end_point.y),
            (preferred_exit.x, preferred_exit.y),
        ])

        # The exit route may leave the selected polygon, but it must not pass
        # through mapped hard obstacles.
        if hard_geometry is not None and not hard_geometry.is_empty:
            if prepared_hard is not None:
                if prepared_hard.intersects(exit_segment):
                    return None
            elif exit_segment.intersects(hard_geometry):
                return None

        # Ignore the final track leg because the exit route necessarily starts
        # at its endpoint. Keep the walking route clear of all earlier legs.
        earlier_legs = accepted_segments[:-1]
        if earlier_legs:
            exit_clearance = min(
                exit_segment.distance(old) for old in earlier_legs
            )
            if exit_clearance < minimum_separation_m:
                return None
        else:
            exit_clearance = exit_clearance_m

    return {
        "geometry": track,
        "length": track.length,
        "points": points,
        "min_separation": float(minimum_observed_separation),
        "preferred_separation": float(preferred_separation_m),
        "start_distance": float(start_distance),
        "exit_distance": float(exit_distance),
        "exit_clearance": float(exit_clearance),
        "exit_segment": exit_segment,
        "preferred_exit_clearance": float(exit_clearance_m),
        "start_to_end_distance": float(start_to_end_distance),
        "openness_ratio": float(openness_ratio),
        "length_error_ratio": float(length_error_ratio),
        "rule_profile": "Appell" if appell_mode else None,
    }


def candidate_score(candidate, forest, soft):
    track = candidate["geometry"]

    if track.length <= 0:
        return -1e9

    forest_length = (
        track.intersection(forest).length
        if not forest.is_empty else 0
    )
    soft_length = (
        track.intersection(soft).length
        if not soft.is_empty else 0
    )

    preferred = max(candidate.get("preferred_separation", 25.0), 1.0)
    separation = candidate.get("min_separation", preferred)

    # Full bonus at the preferred distance; progressively smaller bonus
    # below it. The absolute minimum is enforced by generate_candidate().
    separation_bonus = min(separation / preferred, 1.0) * 30.0

    # Prefer candidates starting close to the user's chosen point.
    start_penalty = min(candidate.get("start_distance", 0.0), 100.0) * 0.35

    # If an exit point is selected, strongly prefer ending close to it and
    # having a clear walking corridor from track end to that point.
    exit_distance_penalty = min(
        candidate.get("exit_distance", 0.0), 250.0
    ) * 0.45

    preferred_exit_clearance = max(
        candidate.get("preferred_exit_clearance", 25.0), 1.0
    )
    exit_clearance = candidate.get(
        "exit_clearance", preferred_exit_clearance
    )
    exit_clearance_bonus = min(
        exit_clearance / preferred_exit_clearance, 1.0
    ) * 35.0

    # Strongly prefer an open, progressing shape. A folded/loop-like track
    # has a small straight-line start-to-end distance relative to its length.
    openness_bonus = min(
        candidate.get("openness_ratio", 0.0) / 0.35, 1.0
    ) * 45.0

    # Adaptive leg lengths are allowed, but candidates closest to the user's
    # requested total length are preferred.
    length_penalty = candidate.get("length_error_ratio", 0.0) * 180.0

    return (
        forest_length / track.length * 100.0
        + soft_length / track.length * 35.0
        + separation_bonus
        + exit_clearance_bonus
        + openness_bonus
        - length_penalty
        - start_penalty
        - exit_distance_penalty
    )


def generate_best(
    area,
    forest,
    soft,
    hard_geometry,
    target_length,
    num_angles,
    num_objects,
    seed,
    n_candidates=250,
    min_leg=60,
    max_leg=100,
    preferred_start=None,
    start_radius_m=35.0,
    preferred_exit=None,
    preferred_separation_m=25.0,
    minimum_separation_m=12.0,
    exit_clearance_m=25.0,
    start_clearance_m=40.0,
    appell_mode=False,
):
    rng = np.random.default_rng(seed)
    scored = []

    # These geometries are queried hundreds/thousands of times per generation.
    # Preparing them once builds an internal spatial index for fast predicates.
    prepared_area = prep(area) if area is not None and not area.is_empty else None
    prepared_hard = (
        prep(hard_geometry)
        if hard_geometry is not None and not hard_geometry.is_empty
        else None
    )

    for _ in range(n_candidates):
        candidate = generate_candidate(
            area=area,
            forest=forest,
            soft=soft,
            hard_geometry=hard_geometry,
            target_length=target_length,
            num_angles=num_angles,
            num_objects=num_objects,
            rng=rng,
            min_leg=min_leg,
            max_leg=max_leg,
            preferred_start=preferred_start,
            start_radius_m=start_radius_m,
            preferred_exit=preferred_exit,
            preferred_separation_m=preferred_separation_m,
            minimum_separation_m=minimum_separation_m,
            exit_clearance_m=exit_clearance_m,
            start_clearance_m=start_clearance_m,
            appell_mode=appell_mode,
            prepared_area=prepared_area,
            prepared_hard=prepared_hard,
        )

        if candidate is None:
            continue

        score = candidate_score(candidate, forest, soft)
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        return None, []

    return scored[0][1], scored


def select_diverse_candidates(
    scored_candidates,
    max_count=5,
    min_shape_distance=35.0,
):
    selected = []

    for score, candidate in scored_candidates:
        geometry = candidate["geometry"]
        too_similar = False

        for _, other in selected:
            distance = geometry.hausdorff_distance(
                other["geometry"]
            )
            if distance < min_shape_distance:
                too_similar = True
                break

        if not too_similar:
            selected.append((score, candidate))

        if len(selected) >= max_count:
            break

    if len(selected) < max_count:
        selected_ids = {id(c) for _, c in selected}

        for score, candidate in scored_candidates:
            if id(candidate) in selected_ids:
                continue

            selected.append((score, candidate))

            if len(selected) >= max_count:
                break

    return selected


def choose_objects(candidate, num_objects, rng):
    """
    Place objects as evenly as possible along the track.

    Rules:
    - The final object is always exactly at the end of the track.
    - Other objects must be at least 10 m from every angle.
    - Among valid positions, choose a distribution that is as even as
      possible along the complete track.
    """
    track = candidate["geometry"]

    # Appell has a fixed three-object layout:
    # 1) middle of first leg
    # 2) middle of second leg
    # 3) exactly at the end of the track
    #
    # The candidate geometry contains one straight LineString made from the
    # track vertices, so the first two leg lengths can be measured directly.
    if num_objects == 3 and candidate.get("rule_profile") == "Appell":
        coords = list(track.coords)
        if len(coords) >= 4:
            leg1 = math.hypot(
                coords[1][0] - coords[0][0],
                coords[1][1] - coords[0][1],
            )
            leg2 = math.hypot(
                coords[2][0] - coords[1][0],
                coords[2][1] - coords[1][1],
            )
            distances = [
                0.5 * leg1,
                leg1 + 0.5 * leg2,
                float(track.length),
            ]
            return [
                {
                    "object": i,
                    "distance_m": float(distance),
                    "x": track.interpolate(float(distance)).x,
                    "y": track.interpolate(float(distance)).y,
                }
                for i, distance in enumerate(distances, start=1)
            ]

    if track.length <= 0 or num_objects <= 0:
        return []

    # One requested object means the mandatory end object only.
    if num_objects == 1:
        distances = [float(track.length)]
    else:
        # Distances of all internal vertices ("angles") measured along track.
        coords = list(track.coords)
        angle_distances = []
        cumulative = 0.0

        for i in range(1, len(coords)):
            x0, y0 = coords[i - 1]
            x1, y1 = coords[i]
            cumulative += math.hypot(x1 - x0, y1 - y0)

            # Exclude the final endpoint: the mandatory last object is
            # explicitly allowed there.
            if i < len(coords) - 1:
                angle_distances.append(cumulative)

        n_regular = num_objects - 1

        # Start from perfectly even target positions. Include the endpoint in
        # the conceptual spacing, but reserve it for the mandatory last object.
        ideal_targets = np.linspace(
            0.0,
            float(track.length),
            num_objects + 1,
        )[1:-1]

        # Fine one-metre candidate grid. Positions within 10 m of any angle
        # are excluded. Also reserve the final 10 m for the end object so a
        # regular object cannot crowd it.
        max_regular_distance = max(float(track.length) - 10.0, 0.0)
        grid = np.arange(0.0, max_regular_distance + 0.5, 1.0)

        valid = [
            float(d)
            for d in grid
            if d > 0.0
            and all(abs(d - a) >= 10.0 for a in angle_distances)
        ]

        selected = []

        # Assign each ideal target to the nearest still-usable point. Keep
        # ordering and a modest separation between objects.
        previous = -float("inf")
        for target in ideal_targets[:n_regular]:
            candidates = [
                d for d in valid
                if d > previous + 1.0
            ]

            if not candidates:
                break

            best = min(candidates, key=lambda d: abs(d - float(target)))
            selected.append(best)
            previous = best

        # If the greedy pass could not place every object, fill remaining
        # slots using valid points that maximize distance from already chosen
        # objects. This favors an even distribution rather than clustering.
        while len(selected) < n_regular:
            remaining = [
                d for d in valid
                if all(abs(d - s) > 1.0 for s in selected)
            ]
            if not remaining:
                break

            anchors = [0.0] + selected + [float(track.length)]
            best = max(
                remaining,
                key=lambda d: min(abs(d - a) for a in anchors),
            )
            selected.append(best)
            selected.sort()

        # In very short/constrained tracks it may be geometrically impossible
        # to place every requested non-end object while respecting the 10 m
        # angle rule. Never violate the angle rule just to reach the count.
        distances = selected[:n_regular] + [float(track.length)]

    return [
        {
            "object": i,
            "distance_m": float(distance),
            "x": track.interpolate(float(distance)).x,
            "y": track.interpolate(float(distance)).y,
        }
        for i, distance in enumerate(distances, start=1)
    ]


def gpx_text(candidate, to_ll, objects=None):
    track = candidate["geometry"]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="SBK Spårgenerator"',
        ' xmlns="http://www.topografix.com/GPX/1/1">',
        '<trk><name>SBK spår</name><trkseg>',
    ]

    for x, y in track.coords:
        lat, lon = xy_ll(x, y, to_ll)
        lines.append(
            f'<trkpt lat="{lat:.8f}" lon="{lon:.8f}"></trkpt>'
        )

    lines.extend(["</trkseg></trk>"])

    for obj in objects or []:
        lat, lon = xy_ll(obj["x"], obj["y"], to_ll)
        lines.append(
            f'<wpt lat="{lat:.8f}" lon="{lon:.8f}">'
            f'<name>Objekt {obj["object"]}</name></wpt>'
        )

    lines.append("</gpx>")
    return "\n".join(lines)


def csv_bytes(candidate, to_ll, objects=None):
    rows = []

    for i, (x, y) in enumerate(candidate["geometry"].coords, 1):
        lat, lon = xy_ll(x, y, to_ll)
        rows.append({
            "type": "track",
            "index": i,
            "latitude": lat,
            "longitude": lon,
            "distance_m": "",
            "object": "",
        })

    for obj in objects or []:
        lat, lon = xy_ll(obj["x"], obj["y"], to_ll)
        rows.append({
            "type": "object",
            "index": "",
            "latitude": lat,
            "longitude": lon,
            "distance_m": obj["distance_m"],
            "object": obj["object"],
        })

    return pd.DataFrame(rows).to_csv(
        index=False, encoding="utf-8-sig"
    ).encode("utf-8-sig")


def add_geom_to_map(m, geom, to_ll, name, fill_opacity=0.20):
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return

    for poly in polygons:
        locations = [
            xy_ll(x, y, to_ll)
            for x, y in poly.exterior.coords
        ]

        folium.Polygon(
            locations=locations,
            tooltip=name,
            fill=True,
            fill_opacity=fill_opacity,
            weight=1,
        ).add_to(m)


def add_lines_to_map(m, geom, to_ll, name, weight=2):
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
    else:
        return

    for line in lines:
        locations = [
            xy_ll(x, y, to_ll)
            for x, y in line.coords
        ]

        folium.PolyLine(
            locations=locations,
            tooltip=name,
            weight=weight,
        ).add_to(m)


def add_track_to_map(m, candidate, to_ll, objects=None):
    track = candidate["geometry"]

    locations = [
        xy_ll(x, y, to_ll)
        for x, y in track.coords
    ]

    folium.PolyLine(
        locations=locations,
        tooltip="Spår",
        weight=5,
    ).add_to(m)

    if locations:
        folium.Marker(
            locations[0],
            tooltip="Start",
            icon=folium.Icon(icon="play"),
        ).add_to(m)

        folium.Marker(
            locations[-1],
            tooltip="Slut",
            icon=folium.Icon(icon="stop"),
        ).add_to(m)

    for obj in objects or []:
        lat, lon = xy_ll(obj["x"], obj["y"], to_ll)

        folium.Marker(
            [lat, lon],
            tooltip=(
                f"Objekt {obj['object']} "
                f"({obj['distance_m']:.0f} m)"
            ),
            icon=folium.Icon(icon="flag", prefix="fa"),
        ).add_to(m)


# ============================================================
# User interface
# ============================================================

st.title("🐕 SBK Spårgenerator")

st.warning(
    "⚠️ Ansvarsfriskrivning: Detta skript är helt autogenererat av ChatGPT. "
    "Användaren ansvarar själv för att kontrollera att det genererade spåret "
    "är korrekt och lämpligt innan spåret påbörjas. Användaren ansvarar även "
    "för sin egen och hundens säkerhet under användning av spåret."
)

st.caption(
    "Rita ett valfritt polygonområde och generera spår inom området."
)

with st.expander("📍 Plats", expanded=True):
    google_pin = st.text_input(
        "Google Maps-position",
        placeholder="59.334591, 18.063240 eller Google Maps-länk",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Använd position", use_container_width=True):
            parsed = parse_google_maps_pin(google_pin)

            if parsed is None:
                st.error(
                    "Kunde inte tolka positionen. "
                    "Ange exempelvis 59.334591, 18.063240."
                )
            else:
                st.session_state.location = parsed
                st.session_state.drawn_area = None
                st.session_state.drawn_area_geojson = None
                st.session_state.drawn_area_key = None
                st.session_state.preferred_start = None
                st.session_state.preferred_start_geojson = None
                st.session_state.preferred_exit = None
                st.session_state.preferred_exit_geojson = None
                st.session_state.osm_layers = None
                st.session_state.osm_area_key = None
                st.session_state.candidates = None
                st.session_state.map_revision += 1
                st.rerun()

    with col2:
        if st.button("Rensa plats", use_container_width=True):
            st.session_state.location = None
            st.session_state.drawn_area = None
            st.session_state.drawn_area_geojson = None
            st.session_state.osm_layers = None
            st.session_state.osm_area_key = None
            st.session_state.candidates = None
            st.rerun()

    if st.session_state.location:
        lat, lon = st.session_state.location
        st.success(f"Position: {lat:.6f}, {lon:.6f}")


if st.session_state.location is None:
    st.info(
        "Ange först en position. Därefter kan du rita "
        "det område där spåret får placeras."
    )
    st.stop()

lat, lon = st.session_state.location
to_xy, to_ll = transformers(lat, lon)

point_selection_mode = st.radio(
    "Kartverktyg",
    ["Rita område", "Startpunkt", "Utgångspunkt"],
    index=0 if st.session_state.drawn_area is None else 1,
    horizontal=True,
    help=(
        "Rita område använder polygonverktyget. För Startpunkt och "
        "Utgångspunkt klickar du direkt en gång på kartan; inget separat "
        "markörverktyg behövs."
    ),
)

m = folium.Map(
    location=[lat, lon],
    zoom_start=15,
    control_scale=True,
)

# Once an area has been selected, it becomes the map's primary focus.
# The original position remains available as a reference marker, but no longer
# determines the viewport.
if st.session_state.drawn_area is not None:
    selected_bounds = polygon_leaflet_bounds(
        st.session_state.drawn_area, to_ll
    )
    if selected_bounds is not None:
        m.fit_bounds(selected_bounds, padding=(25, 25))

folium.Marker(
    [lat, lon],
    tooltip="Vald position",
    icon=folium.Icon(color="blue", icon="map-marker"),
).add_to(m)

if st.session_state.drawn_area_geojson is not None:
    folium.GeoJson(
        st.session_state.drawn_area_geojson,
        name="Valt område",
        style_function=lambda feature: {
            "fillOpacity": 0.10,
            "weight": 3,
        },
    ).add_to(m)


if st.session_state.preferred_start is not None:
    start_lat, start_lon = xy_ll(
        st.session_state.preferred_start.x,
        st.session_state.preferred_start.y,
        to_ll,
    )
    folium.Marker(
        [start_lat, start_lon],
        tooltip="Önskad startpunkt",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

if st.session_state.preferred_exit is not None:
    exit_lat, exit_lon = xy_ll(
        st.session_state.preferred_exit.x,
        st.session_state.preferred_exit.y,
        to_ll,
    )
    folium.Marker(
        [exit_lat, exit_lon],
        tooltip="Önskad utgångspunkt",
        icon=folium.Icon(color="red", icon="sign-out"),
    ).add_to(m)

if point_selection_mode == "Rita område":
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
            },
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(m)

map_data = st_folium(
    m,
    width=None,
    height=520,
    returned_objects=["all_drawings", "last_clicked"],
    key=f"track_area_map_{st.session_state.map_revision}",
)


if map_data:
    drawings = map_data.get("all_drawings", []) or []

    # Polygon state is handled only in area-drawing mode.
    if point_selection_mode == "Rita område":
        polygon_drawings = [
            drawing
            for drawing in drawings
            if drawing.get("geometry", {}).get("type") == "Polygon"
        ]

        for drawing in polygon_drawings:
            geometry = drawing.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            if not coordinates:
                continue

            ring = coordinates[0]
            coords = [
                (lat_value, lon_value)
                for lon_value, lat_value in ring
            ]
            area = folium_polygon_to_shapely(coords, to_xy)

            if area is None:
                continue

            # Store the raw selected polygon here. Track boundary margin is
            # an option shown later in the UI, so it is deliberately NOT
            # applied while processing the map event.
            usable_area = area

            new_key = (
                round(area.area, 1),
                round(area.centroid.x, 1),
                round(area.centroid.y, 1),
            )

            if st.session_state.get("drawn_area_key") != new_key:
                st.session_state.osm_layers = None
                st.session_state.osm_area_key = None
                st.session_state.candidates = None

            st.session_state.drawn_area = usable_area
            st.session_state.drawn_area_geojson = drawing
            st.session_state.drawn_area_key = new_key

    # Start/exit selection uses a plain Leaflet map click rather than a Draw
    # marker. This avoids the transient marker layer that was disappearing on
    # the first Streamlit rerun.
    elif point_selection_mode in ("Startpunkt", "Utgångspunkt"):
        clicked = map_data.get("last_clicked")

        if clicked is not None:
            point_lat = clicked.get("lat")
            point_lon = clicked.get("lng")

            if point_lat is not None and point_lon is not None:
                px, py = to_xy.transform(float(point_lon), float(point_lat))
                selected_point = Point(px, py)

                if point_selection_mode == "Startpunkt":
                    if (
                        st.session_state.drawn_area is not None
                        and st.session_state.drawn_area.covers(selected_point)
                    ):
                        old_start = st.session_state.get("preferred_start")
                        changed = (
                            old_start is None
                            or old_start.distance(selected_point) > 0.5
                        )
                        if changed:
                            st.session_state.preferred_start = selected_point
                            st.session_state.preferred_start_geojson = {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [
                                        float(point_lon),
                                        float(point_lat),
                                    ],
                                },
                            }
                            st.session_state.candidates = None

                            # Create a fresh map instance containing the SAVED
                            # green marker. Unlike Draw markers, the click does
                            # not need to survive the rerun itself.
                            st.session_state.map_revision += 1
                            st.rerun()
                    else:
                        st.warning(
                            "Startpunkten måste ligga inom det användbara området."
                        )

                else:
                    old_exit = st.session_state.get("preferred_exit")
                    changed = (
                        old_exit is None
                        or old_exit.distance(selected_point) > 0.5
                    )
                    if changed:
                        st.session_state.preferred_exit = selected_point
                        st.session_state.preferred_exit_geojson = {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    float(point_lon),
                                    float(point_lat),
                                ],
                            },
                        }
                        st.session_state.candidates = None
                        st.session_state.map_revision += 1
                        st.rerun()

if st.session_state.drawn_area is not None:
    area_ha = st.session_state.drawn_area.area / 10_000

    st.metric(
        "Användbart område",
        f"{area_ha:.1f} ha",
    )

    if st.session_state.preferred_start is None:
        st.caption(
            "Valfri startpunkt: välj 'Startpunkt' ovan och klicka en gång inom området."
        )
    else:
        st.success("Önskad startpunkt vald.")
        if st.button(
            "📍 Rensa vald startpunkt",
            use_container_width=True,
        ):
            st.session_state.preferred_start = None
            st.session_state.preferred_start_geojson = None
            st.session_state.candidates = None
            st.rerun()

    if st.session_state.preferred_exit is None:
        st.caption(
            "Valfri utgångspunkt: välj 'Utgångspunkt' ovan och placera "
            "en markör där du vill lämna området."
        )
    else:
        st.success("Önskad utgångspunkt vald.")
        if st.button(
            "🚶 Rensa vald utgångspunkt",
            use_container_width=True,
        ):
            st.session_state.preferred_exit = None
            st.session_state.preferred_exit_geojson = None
            st.session_state.candidates = None
            st.rerun()

    if st.button(
        "🗑️ Rensa ritat område",
        use_container_width=True,
    ):
        st.session_state.drawn_area = None
        st.session_state.drawn_area_geojson = None
        st.session_state.drawn_area_key = None
        st.session_state.preferred_start = None
        st.session_state.preferred_start_geojson = None
        st.session_state.preferred_exit = None
        st.session_state.preferred_exit_geojson = None
        st.session_state.osm_layers = None
        st.session_state.osm_area_key = None
        st.session_state.candidates = None
        st.session_state.map_revision += 1
        st.rerun()
else:
    st.info(
        "Rita ett polygonområde på kartan med polygonverktyget."
    )


with st.expander("⚙️ Spårinställningar", expanded=True):
    basic_tab, advanced_tab = st.tabs(["Grundinställningar", "Advanced"])

    with basic_tab:
        profile_name = st.selectbox(
            "Regelprofil",
            list(RULE_PROFILES.keys()),
        )
        profile = RULE_PROFILES[profile_name]

        target_length = st.number_input(
            "Spårlängd (m)",
            min_value=100,
            max_value=5000,
            value=profile.target_length,
            step=100,
        )

        num_angles = st.number_input(
            "Antal vinklar",
            min_value=1,
            max_value=20,
            value=profile.angles,
            step=1,
        )

        num_objects = st.number_input(
            "Antal objekt",
            min_value=0,
            max_value=20,
            value=profile.objects,
            step=1,
        )

    with advanced_tab:
        boundary_margin = st.number_input(
            "Marginal från områdesgräns (m)",
            min_value=0,
            max_value=100,
            value=profile.boundary_margin,
            step=5,
        )

        min_leg = st.number_input(
            "Minsta benlängd (m)",
            min_value=20,
            max_value=500,
            value=profile.min_leg,
            step=10,
        )

        max_leg = st.number_input(
            "Största benlängd (m)",
            min_value=20,
            max_value=500,
            value=profile.max_leg,
            step=10,
        )

        preferred_separation = st.number_input(
            "Önskat avstånd mellan spårben (m)",
            min_value=10,
            max_value=100,
            value=int(DEFAULT_PREFERRED_LEG_SEPARATION_M),
            step=5,
            help="Generatorn premierar minst detta avstånd mellan icke angränsande spårben.",
        )

        minimum_separation = st.number_input(
            "Minsta tillåtna avstånd mellan spårben (m)",
            min_value=5,
            max_value=50,
            value=int(DEFAULT_MIN_LEG_SEPARATION_M),
            step=1,
            help="Absolut gräns. Spårben får aldrig komma närmare än detta.",
        )

        start_radius = st.number_input(
            "Tolerans kring vald startpunkt (m)",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            help="Spåret försöker starta nära vald punkt inom denna radie.",
        )

        exit_clearance = st.number_input(
            "Önskat fritt avstånd vid utgång (m)",
            min_value=10,
            max_value=100,
            value=int(DEFAULT_EXIT_CLEARANCE_M),
            step=5,
            help=(
                "Generatorn premierar en utgångsväg som håller minst detta "
                "avstånd till tidigare spårben."
            ),
        )

        start_clearance = st.number_input(
            "Fri zon runt start efter inledningen (m)",
            min_value=20,
            max_value=100,
            value=int(DEFAULT_START_CLEARANCE_M),
            step=5,
            help=(
                "Efter de två första spårbenen får senare ben inte återvända "
                "in i denna zon runt starten. Motverkar att spåret ringlar "
                "runt och stänger in startområdet."
            ),
        )

        seed = st.number_input(
            "Slumpfrö",
            min_value=0,
            max_value=999999,
            value=12345,
            step=1,
        )


# Validate geometry settings before doing any OSM work or candidate search.
n_legs = int(num_angles) + 1
leg_bounds_valid = (
    int(target_length) >= n_legs * int(min_leg)
    and int(target_length) <= n_legs * int(max_leg)
)

if not leg_bounds_valid:
    st.error(
        f"Benlängderna är inte möjliga: {n_legs} ben × "
        f"{int(min_leg)}–{int(max_leg)} m kan inte ge totalt "
        f"{int(target_length)} m."
    )

# Apply the selected boundary margin only after the options have been rendered.
# This keeps the map-before-options layout without referencing an undefined
# setting during polygon selection.
generation_area = None
if st.session_state.drawn_area is not None:
    generation_area = st.session_state.drawn_area.buffer(
        -float(boundary_margin)
    )
    if generation_area.is_empty:
        st.warning(
            "Området blev för litet efter den valda säkerhetsmarginalen."
        )
        generation_area = None

    elif (
        st.session_state.preferred_start is not None
        and not generation_area.covers(st.session_state.preferred_start)
    ):
        st.warning(
            "Den valda startpunkten ligger innanför det ritade området men "
            "utanför området efter vald kantmarginal. Flytta startpunkten "
            "eller minska marginalen."
        )

st.divider()

if st.session_state.drawn_area is not None:
    bounds_area = (
        generation_area
        if generation_area is not None
        else st.session_state.drawn_area
    )
    minx, miny, maxx, maxy = bounds_area.bounds

    lat1, lon1 = xy_ll(minx, miny, to_ll)
    lat2, lon2 = xy_ll(maxx, maxy, to_ll)

    south = min(lat1, lat2)
    north = max(lat1, lat2)
    west = min(lon1, lon2)
    east = max(lon1, lon2)

    area_key = (
        round(south, 5),
        round(west, 5),
        round(north, 5),
        round(east, 5),
    )

    current_key = (
        profile_name,
        int(target_length),
        int(num_angles),
        int(num_objects),
        int(seed),
        int(min_leg),
        int(max_leg),
        int(boundary_margin),
        int(preferred_separation),
        int(minimum_separation),
        int(start_radius),
        int(exit_clearance),
        int(start_clearance),
        (
            round(st.session_state.preferred_start.x, 1),
            round(st.session_state.preferred_start.y, 1),
        ) if st.session_state.preferred_start is not None else None,
        (
            round(st.session_state.preferred_exit.x, 1),
            round(st.session_state.preferred_exit.y, 1),
        ) if st.session_state.preferred_exit is not None else None,
        area_key,
    )

    generation_ready = generation_area is not None and leg_bounds_valid

    if st.button(
        "🐕 Generera 5 spåralternativ",
        type="primary",
        use_container_width=True,
        disabled=not generation_ready,
    ):
        try:
            # OSM data is fetched automatically when needed. The Overpass
            # request itself is cached, and the processed layers are also
            # retained in session state while the selected area is unchanged.
            if (
                st.session_state.osm_layers is None
                or st.session_state.osm_area_key != area_key
            ):
                with st.spinner("Hämtar kartdata och analyserar området..."):
                    # Round to ~1 m. Tiny projection/float differences
                    # should not cause a new Overpass request.
                    data = overpass_query(
                        round(south, 5),
                        round(west, 5),
                        round(north, 5),
                        round(east, 5),
                    )
                    st.session_state.osm_layers = build_osm_layers(
                        data, to_xy
                    )
                    st.session_state.osm_area_key = area_key
                    st.session_state.osm_count = len(
                        data.get("elements", [])
                    )
                    st.session_state.osm_reduced = bool(
                        data.get("_reduced_osm_query", False)
                    )

            layers = st.session_state.osm_layers

            if st.session_state.get("osm_reduced", False):
                st.info(
                    "Overpass var långsamt. En reducerad kartfråga användes "
                    "för denna körning; de viktigaste hindren finns kvar, men "
                    "vissa terrängkategorier kan saknas."
                )

            with st.spinner("Genererar och utvärderar spår..."):
                _, scored = generate_best(
                    area=generation_area,
                    forest=layers["forest"],
                    soft=layers["soft"],
                    hard_geometry=layers["hard"],
                    target_length=int(target_length),
                    num_angles=int(num_angles),
                    num_objects=int(num_objects),
                    seed=int(seed),
                    n_candidates=(
                        90 if int(num_angles) <= 2
                        else 140 if int(num_angles) <= 5
                        else 180
                    ),
                    min_leg=int(min_leg),
                    max_leg=int(max_leg),
                    preferred_start=st.session_state.preferred_start,
                    start_radius_m=float(start_radius),
                    preferred_exit=st.session_state.preferred_exit,
                    preferred_separation_m=float(preferred_separation),
                    minimum_separation_m=float(minimum_separation),
                    exit_clearance_m=float(exit_clearance),
                    start_clearance_m=float(start_clearance),
                    appell_mode=(profile_name == "Appell"),
                )

                selected = select_diverse_candidates(
                    scored,
                    max_count=5,
                    min_shape_distance=35.0,
                )

                rng = np.random.default_rng(int(seed))
                candidates = []

                for rank, (score, candidate) in enumerate(
                    selected, start=1
                ):
                    objects = choose_objects(
                        candidate,
                        int(num_objects),
                        rng,
                    )

                    candidates.append({
                        "rank": rank,
                        "score": score,
                        "candidate": candidate,
                        "objects": objects,
                    })

                st.session_state.candidates = candidates
                st.session_state.candidate_key = current_key

                if not candidates:
                    st.error(
                        "Kunde inte generera något giltigt spår "
                        "inom det valda området."
                    )
                else:
                    st.success(
                        f"{len(candidates)} spåralternativ genererade."
                    )

        except Exception as exc:
            st.error(
                "Kunde inte hämta kartdata eller generera spår: "
                f"{exc}"
            )


if (
    st.session_state.candidates
    and st.session_state.candidate_key == current_key
):
    candidates = st.session_state.candidates

    labels = [
        f"Alternativ {item['rank']}"
        for item in candidates
    ]

    selected_label = st.radio(
        "Välj spåralternativ",
        labels,
        horizontal=True,
    )

    selected = candidates[labels.index(selected_label)]
    candidate = selected["candidate"]
    objects = selected["objects"]

    if not st.session_state.drawn_area.covers(
        candidate["geometry"]
    ):
        st.error(
            "Det valda spåret ligger inte helt inom området. "
            "Generera spåren igen."
        )
        st.stop()

    st.divider()

    track_map = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        control_scale=True,
    )

    selected_bounds = polygon_leaflet_bounds(
        st.session_state.drawn_area, to_ll
    )
    if selected_bounds is not None:
        track_map.fit_bounds(selected_bounds, padding=(25, 25))

    if st.session_state.drawn_area_geojson:
        folium.GeoJson(
            st.session_state.drawn_area_geojson,
            name="Valt område",
            style_function=lambda feature: {
                "fillOpacity": 0.05,
                "weight": 2,
            },
        ).add_to(track_map)

    layers = st.session_state.osm_layers

    add_geom_to_map(
        track_map, layers["forest"], to_ll,
        "Skog", 0.12
    )
    add_geom_to_map(
        track_map, layers["soft"], to_ll,
        "Övrig natur", 0.08
    )
    add_lines_to_map(
        track_map, layers["roads"], to_ll,
        "Väg", 3
    )
    add_lines_to_map(
        track_map, layers["paths"], to_ll,
        "Stig", 2
    )
    add_track_to_map(
        track_map, candidate, to_ll, objects
    )

    if candidate.get("exit_segment") is not None:
        exit_locations = [
            xy_ll(x, y, to_ll)
            for x, y in candidate["exit_segment"].coords
        ]
        folium.PolyLine(
            locations=exit_locations,
            tooltip="Föreslagen utgångsväg",
            weight=4,
            dash_array="8, 8",
        ).add_to(track_map)
        folium.Marker(
            exit_locations[-1],
            tooltip="Utgångspunkt",
            icon=folium.Icon(color="red", icon="sign-out"),
        ).add_to(track_map)

    st_folium(
        track_map,
        width=None,
        height=520,
        returned_objects=[],
    )

    track = candidate["geometry"]

    forest_length = (
        track.intersection(layers["forest"]).length
        if not layers["forest"].is_empty else 0
    )

    nature = layers["forest"].union(layers["soft"])

    nature_length = (
        track.intersection(nature).length
        if not nature.is_empty else 0
    )

    forest_percent = (
        forest_length / track.length * 100
        if track.length else 0
    )

    nature_percent = (
        nature_length / track.length * 100
        if track.length else 0
    )

    cols = st.columns(4)

    cols[0].metric(
        "Längd", f"{track.length:.0f} m"
    )
    cols[1].metric(
        "Skog", f"{forest_percent:.0f} %"
    )
    cols[2].metric(
        "Natur", f"{nature_percent:.0f} %"
    )
    cols[3].metric(
        "Vinklar", str(int(num_angles))
    )

    extra_cols = st.columns(2)
    extra_cols[0].metric(
        "Min. benavstånd",
        f"{candidate.get('min_separation', 0.0):.0f} m",
    )
    if st.session_state.preferred_start is not None:
        extra_cols[1].metric(
            "Start från vald punkt",
            f"{candidate.get('start_distance', 0.0):.0f} m",
        )

    openness_cols = st.columns(2)
    openness_cols[0].metric(
        "Start–slut fågelväg",
        f"{candidate.get('start_to_end_distance', 0.0):.0f} m",
    )
    openness_cols[1].metric(
        "Öppenhet",
        f"{candidate.get('openness_ratio', 0.0) * 100:.0f} %",
    )

    if st.session_state.preferred_exit is not None:
        exit_cols = st.columns(2)
        exit_cols[0].metric(
            "Slut till utgångspunkt",
            f"{candidate.get('exit_distance', 0.0):.0f} m",
        )
        exit_cols[1].metric(
            "Min. avstånd på utgångsväg",
            f"{candidate.get('exit_clearance', 0.0):.0f} m",
        )

    if objects:
        st.subheader("Objekt")

        rows = []

        for obj in objects:
            lat_obj, lon_obj = xy_ll(
                obj["x"], obj["y"], to_ll
            )

            rows.append({
                "Objekt": obj["object"],
                "Avstånd": f"{obj['distance_m']:.0f} m",
                "Lat": f"{lat_obj:.6f}",
                "Lon": f"{lon_obj:.6f}",
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "⬇️ GPX",
        data=gpx_text(candidate, to_ll, objects),
        file_name="sbk_spår.gpx",
        mime="application/gpx+xml",
        use_container_width=True,
    )

    st.download_button(
        "⬇️ CSV",
        data=csv_bytes(candidate, to_ll, objects),
        file_name="sbk_spår.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if (
        st.session_state.candidate_key is not None
        and current_key != st.session_state.candidate_key
    ):
        st.warning(
            "Spårinställningarna har ändrats sedan spåren "
            "genererades. Generera spåren igen."
        )


