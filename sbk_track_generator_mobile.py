#!/usr/bin/env python3
"""
SBK Spårhundsgruppen - OSM-aware track generator

Run:
    pip install -r requirements_osm.txt
    streamlit run sbk_track_generator_osm.py

Features:
* Google Maps pin / lat,lon input
* Editable rectangular planning area
* OpenStreetMap/Overpass terrain and obstacle data
* Forest preference, hard avoidance of buildings/water/roads/etc.
* Adjustable buffers around obstacles
* Candidate generation + scoring
* SBK-oriented track geometry and object placement
* GPX/CSV export

The OSM data is a planning aid. It does not establish land ownership,
public access, terrain condition, recent forestry work, fences, hunting,
private restrictions, or competition suitability.
"""
from __future__ import annotations

import html
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium.plugins import Draw, Fullscreen
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.prepared import prep
from pyproj import Transformer
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
CACHE_TTL = 900


@dataclass(frozen=True)
class RuleProfile:
    name: str
    default_length: int
    default_angles: int
    default_objects: int
    first_angle_min: float
    first_object_min: float
    object_angle_clearance: float


RULES = {
    "2023-2026 / lägre": RuleProfile("2023-2026 / lägre", 1000, 5, 8, 100, 100, 10),
    "2023-2026 / högre": RuleProfile("2023-2026 / högre", 1200, 6, 8, 100, 100, 10),
    "2023-2026 / elit": RuleProfile("2023-2026 / elit", 1500, 7, 8, 100, 100, 10),
    "2027 / lägre": RuleProfile("2027 / lägre", 1000, 5, 8, 60, 100, 10),
    "2027 / högre": RuleProfile("2027 / högre", 1200, 6, 8, 60, 100, 10),
    "2027 / elit": RuleProfile("2027 / elit", 1500, 7, 8, 60, 100, 10),
}


def parse_google_maps_pin(text: str) -> Optional[Tuple[float, float]]:
    text = text.strip()
    patterns = [
        r"@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)",
        r"(?:[?&](?:q|query|ll)=)(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)",
        r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)",
        r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def transformers(lat: float, lon: float):
    epsg = utm_epsg(lat, lon)
    return (
        Transformer.from_crs(4326, epsg, always_xy=True),
        Transformer.from_crs(epsg, 4326, always_xy=True),
    )


def ll_xy(lat, lon, t):
    return t.transform(lon, lat)


def xy_ll(x, y, t):
    lon, lat = t.transform(x, y)
    return lat, lon


def safe_union(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else GeometryCollection()


def polygonal(geom):
    if geom is None or geom.is_empty:
        return GeometryCollection()
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            if isinstance(g, (Polygon, MultiPolygon)):
                polys.append(g)
    return safe_union(polys)


def line_buffer_union(lines, width):
    return safe_union([g.buffer(width) for g in lines if not g.is_empty])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def overpass_query(south: float, west: float, north: float, east: float) -> dict:
    # Keep the query deliberately focused so a few-km planning box remains usable.
    bbox = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    query = f"""
    [out:json][timeout:45];
    (
      way[landuse=forest]({bbox});
      way[natural=wood]({bbox});
      way[natural=scrub]({bbox});
      way[landuse=meadow]({bbox});
      way[landuse=grass]({bbox});
      way[landuse=farmland]({bbox});
      way[landuse=residential]({bbox});
      way[landuse=industrial]({bbox});
      way[landuse=commercial]({bbox});
      way[landuse=quarry]({bbox});
      way[natural=water]({bbox});
      way[waterway=riverbank]({bbox});
      way[highway]({bbox});
      way[building]({bbox});
      way[barrier]({bbox});
      relation[landuse=forest]({bbox});
      relation[natural=wood]({bbox});
      relation[natural=water]({bbox});
    );
    out geom tags;
    """
    last_error = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data=query, timeout=60, headers={"User-Agent": "SBK-Spårgenerator/1.0"})
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Kunde inte hämta OSM-data från Overpass: {last_error}")


def elements_to_geometries(data: dict, to_xy: Transformer) -> Dict[str, List]:
    groups = {"forest": [], "soft": [], "hard_area": [], "water": [], "roads": [], "paths": [], "barriers": []}
    highway_soft = {"path", "footway", "track", "bridleway", "cycleway", "steps", "pedestrian", "service"}
    highway_hard = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified", "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

    for el in data.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        coords = [to_xy.transform(p["lon"], p["lat"]) for p in geom]
        tags = el.get("tags", {})
        closed = coords[0] == coords[-1]
        try:
            if closed and len(coords) >= 4:
                g = Polygon(coords)
                if not g.is_valid:
                    g = g.buffer(0)
            else:
                g = LineString(coords)
        except Exception:
            continue
        if g.is_empty:
            continue

        landuse = tags.get("landuse")
        natural = tags.get("natural")
        highway = tags.get("highway")
        waterway = tags.get("waterway")

        if landuse == "forest" or natural == "wood":
            groups["forest"].append(g)
        elif natural == "scrub" or landuse in {"meadow", "grass"}:
            groups["soft"].append(g)
        elif natural == "water" or waterway == "riverbank":
            groups["water"].append(g)
        elif landuse in {"residential", "industrial", "commercial", "quarry"}:
            groups["hard_area"].append(g)
        elif highway:
            if highway in highway_hard:
                groups["roads"].append(g)
            elif highway in highway_soft:
                groups["paths"].append(g)
            else:
                groups["paths"].append(g)
        elif "building" in tags:
            groups["hard_area"].append(g)
        elif "barrier" in tags:
            groups["barriers"].append(g)
    return groups


def build_osm_layers(data, to_xy, bbox: Polygon, road_buffer: float, building_buffer: float, water_buffer: float, barrier_buffer: float):
    g = elements_to_geometries(data, to_xy)
    forest = polygonal(safe_union([x for x in g["forest"] if isinstance(x, (Polygon, MultiPolygon))]))
    soft = polygonal(safe_union([x for x in g["soft"] if isinstance(x, (Polygon, MultiPolygon))]))
    water = safe_union(g["water"])
    hard_area = safe_union(g["hard_area"])
    roads = safe_union(g["roads"])
    paths = safe_union(g["paths"])
    barriers = safe_union(g["barriers"])

    hard = safe_union([
        hard_area.buffer(building_buffer),
        water.buffer(water_buffer),
        roads.buffer(road_buffer),
        barriers.buffer(barrier_buffer),
    ])
    # Clip all planning layers to keep geometry manageable.
    return {
        "forest": forest.intersection(bbox),
        "soft": soft.intersection(bbox),
        "water": water.intersection(bbox),
        "hard_area": hard_area.intersection(bbox),
        "roads": roads.intersection(bbox),
        "paths": paths.intersection(bbox),
        "barriers": barriers.intersection(bbox),
        "hard": hard.intersection(bbox),
    }


def random_segment_lengths(total, n_segments, first_min, min_segment=55):
    mins = [max(first_min, min_segment)] + [min_segment] * (n_segments - 1)
    if sum(mins) >= total:
        raise ValueError(f"För kort spår för {n_segments-1} vinklar. Ungefärlig minsta längd: {sum(mins):.0f} m.")
    rem = total - sum(mins)
    # Slightly variable but not excessively jagged.
    weights = np.random.default_rng().dirichlet(np.ones(n_segments) * 2.2)
    return [mins[i] + rem * float(weights[i]) for i in range(n_segments)]


def sample_point_in_area(area, rng: random.Random, max_attempts=1000):
    if area.is_empty:
        return None
    minx, miny, maxx, maxy = area.bounds
    for _ in range(max_attempts):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if area.covers(p):
            return p
    return area.representative_point()


def segment_allowed(seg: LineString, layers, bbox, min_path_gap, forest_fraction_min):
    if not bbox.covers(seg):
        return False
    if seg.intersects(layers["hard"]):
        return False
    if seg.length < 1:
        return False
    # A training track may use non-forest terrain, but a candidate is rejected
    # if it spends too much of its length outside mapped woodland/soft ground.
    preferred = safe_union([layers["forest"], layers["soft"]])
    if not preferred.is_empty:
        inside = seg.intersection(preferred).length / seg.length
        if inside < forest_fraction_min:
            return False
    # Avoid long sections immediately beside an existing path/road unless
    # the user deliberately lowers this threshold.
    if not layers["paths"].is_empty and seg.distance(layers["paths"]) < min_path_gap:
        return False
    return True


def generate_candidate(start, total_length, n_angles, bbox, layers, rules, rng, forest_fraction_min, min_path_gap):
    lengths = random_segment_lengths(total_length, n_angles + 1, rules.first_angle_min)
    safe_start = bbox.difference(layers["hard"])
    if not safe_start.covers(Point(start)):
        p = sample_point_in_area(safe_start, rng)
        if p is None:
            return None
        start = p.coords[0]

    for attempt in range(120):
        heading = rng.uniform(0, 360)
        points = [start]
        ok = True
        for i, length in enumerate(lengths):
            if i == 0:
                new_heading = heading
            else:
                # Realistic training tracks: predominantly clear direction changes,
                # with some shallower bends. Prevent 180-degree reversals.
                turn = rng.choice([
                    rng.uniform(65, 115), rng.uniform(-115, -65),
                    rng.uniform(35, 60), rng.uniform(-60, -35),
                ])
                new_heading = heading + turn
            r = math.radians(new_heading)
            q = (points[-1][0] + length * math.cos(r), points[-1][1] + length * math.sin(r))
            seg = LineString([points[-1], q])
            if not segment_allowed(seg, layers, bbox, min_path_gap, forest_fraction_min):
                ok = False
                break
            if len(points) >= 2:
                existing = LineString(points)
                if seg.crosses(existing) or seg.distance(existing) < 25:
                    ok = False
                    break
            points.append(q)
            heading = new_heading
        if ok:
            line = LineString(points)
            if line.length >= total_length * 0.995:
                return line
    return None


def candidate_score(line, layers, start, min_path_gap):
    preferred = safe_union([layers["forest"], layers["soft"]])
    forest_ratio = line.intersection(preferred).length / max(line.length, 1) if not preferred.is_empty else 0
    path_penalty = 0 if layers["paths"].is_empty else max(0, 1 - line.distance(layers["paths"]) / max(min_path_gap, 1))
    edge_penalty = line.distance(layers["hard"])
    # Score primarily by suitable terrain, then by avoiding paths.
    return 100 * forest_ratio - 20 * path_penalty + min(edge_penalty, 50) / 10


def generate_best(start, total_length, n_angles, bbox, layers, rules, seed, candidates_n, forest_fraction_min, min_path_gap):
    rng = random.Random(seed)
    candidates = []
    for i in range(candidates_n):
        line = generate_candidate(start, total_length, n_angles, bbox, layers, rules, rng, forest_fraction_min, min_path_gap)
        if line:
            candidates.append((candidate_score(line, layers, start, min_path_gap), line))
    if not candidates:
        raise RuntimeError("Inget giltigt spår hittades. Prova större område, lägre krav på skogsandel eller mindre hinderbuffertar.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates


def select_diverse_candidates(scored_candidates, max_count=5, min_shape_distance=35.0):
    """Select high-scoring but geometrically different candidates."""
    selected = []
    for score, line in scored_candidates:
        if not selected:
            selected.append((score, line))
        else:
            # Hausdorff distance is useful here because all tracks share roughly
            # the same start point but should otherwise look different.
            if all(line.hausdorff_distance(other) >= min_shape_distance for _, other in selected):
                selected.append((score, line))
        if len(selected) >= max_count:
            break
    # If the geometry is unusually constrained, fill the remaining slots with
    # the best unused candidates rather than failing.
    if len(selected) < max_count:
        chosen_ids = {id(line) for _, line in selected}
        for item in scored_candidates:
            if id(item[1]) not in chosen_ids:
                selected.append(item)
            if len(selected) >= max_count:
                break
    return selected


def choose_objects(line, n_objects, rules, seed):
    rng = random.Random(seed)
    total = line.length
    if n_objects == 1:
        return [(total, line.interpolate(total))]
    first = max(rules.first_object_min, 1)
    last = total
    available_end = max(first + 20, last - 1)
    if available_end <= first:
        return [(last, line.interpolate(last))]
    # Last object is fixed at the finish; the others are distributed with jitter.
    ds = sorted(rng.uniform(first, available_end - 20) for _ in range(n_objects - 1))
    out = []
    for d in ds:
        if out and d - out[-1][0] < 45:
            d = out[-1][0] + 45
        if d < total - 30:
            out.append((d, line.interpolate(d)))
    out.append((total, line.interpolate(total)))
    return out[:n_objects]


def gpx_text(track_ll, objects):
    pts = "\n".join(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}" />' for lat, lon in track_ll)
    wpts = "\n".join(f'  <wpt lat="{lat:.7f}" lon="{lon:.7f}"><name>Objekt {i}</name></wpt>' for i, (lat, lon) in enumerate(objects, 1))
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="SBK Spårgenerator OSM" xmlns="http://www.topografix.com/GPX/1/1">
{wpts}
  <trk><name>SBK spårförslag</name><trkseg>
{pts}
  </trkseg></trk>
</gpx>'''


def csv_bytes(line, to_ll, objects):
    rows = []
    for i, (x, y) in enumerate(line.coords):
        lat, lon = xy_ll(x, y, to_ll)
        rows.append({"type": "track", "index": i + 1, "distance_m": round(line.project(Point(x, y)), 1), "lat": lat, "lon": lon})
    for i, (d, p) in enumerate(objects, 1):
        lat, lon = xy_ll(p.x, p.y, to_ll)
        rows.append({"type": "object", "index": i, "distance_m": round(d, 1), "lat": lat, "lon": lon})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def add_geom_to_map(m, geom, to_ll, color, fill=False, weight=2, fill_opacity=0.08, tooltip=None):
    if geom is None or geom.is_empty:
        return
    polys = []
    if isinstance(geom, Polygon): polys = [geom]
    elif isinstance(geom, MultiPolygon): polys = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
    for poly in polys:
        coords = [xy_ll(x, y, to_ll) for x, y in poly.exterior.coords]
        folium.Polygon([(a, b) for a, b in coords], color=color, fill=not fill, fill_opacity=fill_opacity, weight=weight, tooltip=tooltip).add_to(m)


def add_lines_to_map(m, geom, to_ll, color, weight=3, opacity=0.6):
    if geom is None or geom.is_empty:
        return
    lines = [geom] if isinstance(geom, LineString) else [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]
    for line in lines:
        coords = [xy_ll(x, y, to_ll) for x, y in line.coords]
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity).add_to(m)


# ------------------------------- UI ---------------------------------------
st.set_page_config(
    page_title="SBK Spårgenerator",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {padding-top: .65rem; padding-left: .65rem; padding-right: .65rem; max-width: 900px;}
.stButton > button, .stDownloadButton > button {min-height: 3.1rem; font-size: 1rem; width: 100%;}
input, textarea, [role="combobox"] {font-size: 16px !important;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
.small-note {font-size: .82rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

st.title("🐕 SBK Spårgenerator")
st.caption("Mobilanpassad OSM-baserad planering för Spårhundsgruppen")

# Session state keeps the expensive OSM request and generated candidates alive
# while the user changes display options or selects another candidate.
def ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

ss_default("location", None)
ss_default("drawn_bounds", None)
ss_default("osm_layers", None)
ss_default("osm_area_key", None)
ss_default("osm_count", 0)
ss_default("candidates", None)
ss_default("candidate_key", None)

with st.expander("📍 Plats", expanded=True):
    loc_col1, loc_col2 = st.columns([3, 2])
    with loc_col1:
        maps_input = st.text_input(
            "Google Maps-pin",
            placeholder="Klistra in länk eller lat, lon",
            key="maps_input",
        )
    with loc_col2:
        gps_clicked = st.button("📍 Min position", use_container_width=True)

    # Browser geolocation works on HTTPS (including Streamlit Community Cloud).
    # The location is only read when the user explicitly requests it.
    if gps_clicked:
        loc = streamlit_geolocation()
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            st.session_state["location"] = (float(loc["latitude"]), float(loc["longitude"]))
            st.session_state["maps_input"] = f"{loc['latitude']:.6f}, {loc['longitude']:.6f}"
            st.session_state["drawn_bounds"] = None
            st.session_state["candidates"] = None
            st.session_state["candidate_key"] = None
            st.rerun()
        else:
            st.info("Tillåt platsåtkomst i webbläsaren och tryck sedan på knappen igen.")

    parsed = parse_google_maps_pin(maps_input) if maps_input else None
    if parsed:
        if st.session_state.get("location") != parsed:
            st.session_state["location"] = parsed
            st.session_state["drawn_bounds"] = None
            st.session_state["candidates"] = None
            st.session_state["candidate_key"] = None
        lat, lon = parsed
        st.success(f"Startpunkt: {lat:.6f}, {lon:.6f}")
    elif st.session_state.get("location"):
        lat, lon = st.session_state["location"]
        st.info(f"Använd senast valda position: {lat:.6f}, {lon:.6f}")
    else:
        lat, lon = 59.3293, 18.0686
        st.caption("Ingen plats vald — kartan startar i Stockholm.")

with st.expander("⚙️ Spårinställningar", expanded=True):
    profile_name = st.selectbox("Regelprofil", list(RULES), index=3, key="profile")
    rules = RULES[profile_name]
    c1, c2 = st.columns(2)
    with c1:
        length = st.number_input("Mållängd (m)", 200, 5000, rules.default_length, 100, key="length")
        angles = st.number_input("Antal vinklar", 1, 20, rules.default_angles, 1, key="angles")
    with c2:
        objects = st.number_input("Antal föremål", 1, 20, rules.default_objects, 1, key="objects")
        seed = st.number_input("Slumpfrö", 0, 9999999, 12345, 1, key="seed")

    st.markdown("**OSM / terräng**")
    c1, c2 = st.columns(2)
    with c1:
        forest_min = st.slider("Min. skog/öppen natur", 0.0, 1.0, 0.55, 0.05, format="%.0f%%", key="forest_min")
        road_buffer = st.slider("Vägbuffert (m)", 0, 80, 15, 5, key="road_buffer")
        water_buffer = st.slider("Vattenbuffert (m)", 0, 80, 10, 5, key="water_buffer")
    with c2:
        building_buffer = st.slider("Bebyggelsebuffert (m)", 0, 80, 15, 5, key="building_buffer")
        barrier_buffer = st.slider("Barriärbuffert (m)", 0, 40, 3, 1, key="barrier_buffer")
        path_gap = st.slider("Min. avstånd till OSM-stig (m)", 0, 50, 8, 2, key="path_gap")

    st.markdown("**Standardområde runt positionen**")
    c1, c2 = st.columns(2)
    with c1:
        half_w = st.slider("Halv bredd (m)", 100, 3000, 800, 50, key="half_w")
    with c2:
        half_h = st.slider("Halv höjd (m)", 100, 3000, 800, 50, key="half_h")

# Work in local UTM coordinates for metre-accurate geometry.
to_xy, to_ll = transformers(lat, lon)
cx, cy = ll_xy(lat, lon, to_xy)
lon_half = half_w / (111000 * max(math.cos(math.radians(lat)), 0.1))
lat_half = half_h / 111000
default_bounds = [[lat - lat_half, lon - lon_half], [lat + lat_half, lon + lon_half]]

st.subheader("1. 📐 Välj område")
st.caption("Tryck på rektangelverktyget och dra runt området. På Android fungerar detta med finger.")

m = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True, width="100%", height=430)
folium.Marker([lat, lon], tooltip="Startpunkt", icon=folium.Icon(color="blue", icon="flag")).add_to(m)
if st.session_state.get("drawn_bounds"):
    db = st.session_state["drawn_bounds"]
    shown_bounds = [[db[1], db[0]], [db[3], db[2]]]
else:
    shown_bounds = default_bounds
folium.Rectangle(shown_bounds, color="blue", fill=False, weight=2).add_to(m)
Draw(
    export=False,
    draw_options={"polyline": False, "polygon": False, "circle": False, "circlemarker": False, "marker": False, "rectangle": True},
    edit_options={"edit": True, "remove": True},
).add_to(m)
Fullscreen(position="topleft").add_to(m)
map_data = st_folium(m, width="100%", height=430, returned_objects=["all_drawings"], key="planning_map")

if map_data and map_data.get("all_drawings"):
    for d in reversed(map_data["all_drawings"]):
        if d.get("geometry", {}).get("type") == "Polygon":
            pts = d["geometry"]["coordinates"][0]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            new_bounds = (min(xs), min(ys), max(xs), max(ys))
            if new_bounds != st.session_state.get("drawn_bounds"):
                st.session_state["drawn_bounds"] = new_bounds
                st.session_state["candidates"] = None
                st.session_state["candidate_key"] = None
            break

if st.session_state.get("drawn_bounds"):
    minlon, minlat, maxlon, maxlat = st.session_state["drawn_bounds"]
    bbox = Polygon([
        ll_xy(minlat, minlon, to_xy), ll_xy(minlat, maxlon, to_xy),
        ll_xy(maxlat, maxlon, to_xy), ll_xy(maxlat, minlon, to_xy)
    ])
    area_source = "ritad rektangel"
else:
    bbox = Polygon([(cx-half_w, cy-half_h), (cx+half_w, cy-half_h), (cx+half_w, cy+half_h), (cx-half_w, cy+half_h)])
    area_source = "standardområde"

current_area_key = tuple(round(v, 1) for v in bbox.bounds)
st.info(f"{area_source} · {bbox.bounds[2]-bbox.bounds[0]:.0f} × {bbox.bounds[3]-bbox.bounds[1]:.0f} m")

st.subheader("2. 🗺️ OSM-underlag")
fc1, fc2 = st.columns(2)
with fc1:
    fetch = st.button("Hämta OSM-data", use_container_width=True)
with fc2:
    clear = st.button("Rensa genererade spår", use_container_width=True)

if clear:
    st.session_state["candidates"] = None
    st.session_state["candidate_key"] = None
    st.rerun()

def fetch_osm_for_bbox():
    bounds = bbox.bounds
    a_lat, a_lon = xy_ll(bounds[0], bounds[1], to_ll)
    b_lat, b_lon = xy_ll(bounds[2], bounds[3], to_ll)
    south, north = min(a_lat, b_lat), max(a_lat, b_lat)
    west, east = min(a_lon, b_lon), max(a_lon, b_lon)
    with st.spinner("Hämtar OSM-data …"):
        data = overpass_query(south, west, north, east)
    layers_ = build_osm_layers(data, to_xy, bbox, road_buffer, building_buffer, water_buffer, barrier_buffer)
    st.session_state["osm_layers"] = layers_
    st.session_state["osm_area_key"] = current_area_key
    st.session_state["osm_count"] = len(data.get("elements", []))
    st.session_state["candidates"] = None
    st.session_state["candidate_key"] = None

if fetch:
    try:
        fetch_osm_for_bbox()
        st.success(f"OSM-data hämtad: {st.session_state['osm_count']} objekt.")
    except Exception as exc:
        st.error(f"OSM-fel: {exc}")

layers = st.session_state.get("osm_layers")
if layers is not None and st.session_state.get("osm_area_key") != current_area_key:
    st.warning("Området har ändrats. Hämta OSM-data igen innan du genererar spår.")
    layers = None

if layers is not None:
    with st.expander("👁️ Visa OSM-underlag", expanded=False):
        om = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True, width="100%", height=400)
        add_geom_to_map(om, layers["forest"], to_ll, "green", fill=False, tooltip="OSM skog")
        add_geom_to_map(om, layers["water"], to_ll, "blue", fill=False, tooltip="OSM vatten")
        add_geom_to_map(om, layers["hard_area"], to_ll, "red", fill=False, tooltip="Bebyggelse/markslag")
        add_lines_to_map(om, layers["roads"], to_ll, "black", 4, 0.7)
        add_lines_to_map(om, layers["paths"], to_ll, "orange", 2, 0.8)
        folium.Marker([lat, lon], tooltip="Startpunkt").add_to(om)
        st_folium(om, width="100%", height=400, returned_objects=[], key="osm_map")

# Candidate generation is deliberately separated from the parameter widgets.
# Changing a parameter no longer causes an OSM request or silently replaces the
# five generated alternatives. Press Generate again when you want new tracks.
gen_key = (
    current_area_key, profile_name, int(length), int(angles), int(objects), int(seed),
    round(float(forest_min), 2), int(road_buffer), int(water_buffer), int(building_buffer),
    int(barrier_buffer), int(path_gap),
)

st.subheader("3. 🐾 Generera fem alternativ")
generate = st.button("🐾 Generera 5 spåralternativ", type="primary", use_container_width=True)

if generate:
    try:
        if layers is None:
            st.error("Hämta OSM-data för det aktuella området först.")
        else:
            with st.spinner("Genererar och jämför spår …"):
                # 100 candidates is enough for a responsive phone UI; only the
                # five selected alternatives are retained for display.
                _, scored = generate_best(
                    (cx, cy), int(length), int(angles), bbox, layers, rules,
                    int(seed), 100, float(forest_min), float(path_gap)
                )
                selected = select_diverse_candidates(scored, max_count=5, min_shape_distance=35.0)
                records = []
                for idx, (score, line) in enumerate(selected, 1):
                    objs = choose_objects(line, int(objects), rules, int(seed) + 1001 + idx)
                    records.append({"line": line, "score": score, "objects": objs, "rank": idx})
                st.session_state["candidates"] = records
                st.session_state["candidate_key"] = gen_key
            st.success(f"{len(records)} spåralternativ skapade.")
    except Exception as exc:
        st.error(f"Kunde inte generera spåren: {exc}")

records = st.session_state.get("candidates")
if records:
    if st.session_state.get("candidate_key") != gen_key:
        st.warning("Parametrarna har ändrats sedan spåren skapades. Tryck på **Generera 5 spåralternativ** för att skapa nya spår med de nya parametrarna.")
    labels = []
    for r in records:
        line = r["line"]
        preferred = safe_union([layers["forest"], layers["soft"]])
        pref_ratio = line.intersection(preferred).length / line.length if not preferred.is_empty else 0
        labels.append(f"Alternativ {r['rank']} · {line.length:.0f} m · {pref_ratio*100:.0f}% skog/natur")

    selected_label = st.radio("Välj spår att visa/exportera", labels, index=0, key="selected_candidate")
    selected_idx = labels.index(selected_label)
    selected_record = records[selected_idx]
    track = selected_record["line"]
    objs = selected_record["objects"]

    out = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True, width="100%", height=520)
    area_ll = [xy_ll(x, y, to_ll) for x, y in bbox.exterior.coords]
    folium.Polygon(area_ll, color="blue", fill=False, weight=2).add_to(out)
    if layers is not None:
        add_geom_to_map(out, layers["forest"], to_ll, "green", fill=False, weight=1, fill_opacity=0.05)
        add_lines_to_map(out, layers["roads"], to_ll, "black", 3, 0.45)
        add_lines_to_map(out, layers["paths"], to_ll, "orange", 2, 0.5)
    track_ll = [xy_ll(x, y, to_ll) for x, y in track.coords]
    folium.PolyLine(track_ll, color="red", weight=5, opacity=0.9, tooltip=f"Spår {track.length:.0f} m").add_to(out)
    slat, slon = track_ll[0]
    flat, flon = track_ll[-1]
    folium.Marker([slat, slon], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(out)
    folium.Marker([flat, flon], tooltip="Mål", icon=folium.Icon(color="red", icon="stop")).add_to(out)
    coords = list(track.coords)
    for i, (x, y) in enumerate(coords[1:-1], 1):
        alat, alon = xy_ll(x, y, to_ll)
        d = track.project(Point(x, y))
        folium.CircleMarker([alat, alon], radius=5, color="black", fill=True, fill_opacity=1, tooltip=f"Vinkel {i} · {d:.0f} m").add_to(out)
    for i, (d, p) in enumerate(objs, 1):
        olat, olon = xy_ll(p.x, p.y, to_ll)
        folium.Marker([olat, olon], tooltip=f"Föremål {i} · {d:.0f} m", icon=folium.Icon(color="purple" if i == len(objs) else "orange", icon="star")).add_to(out)

    st.subheader("4. 📱 Vald spår")
    st_folium(out, width="100%", height=520, returned_objects=[], key=f"generated_{selected_idx}_{gen_key}")

    preferred = safe_union([layers["forest"], layers["soft"]])
    pref_ratio = track.intersection(preferred).length / track.length if not preferred.is_empty else 0
    c1, c2 = st.columns(2)
    c1.metric("Längd", f"{track.length:.0f} m")
    c2.metric("Skog/natur", f"{pref_ratio*100:.0f}%")
    c1, c2 = st.columns(2)
    c1.metric("Vinklar", str(len(coords)-2))
    c2.metric("Föremål", str(len(objs)))

    rows = []
    for i, (d, p) in enumerate(objs, 1):
        olat, olon = xy_ll(p.x, p.y, to_ll)
        rows.append({"Nr": i, "Avstånd (m)": round(d), "Lat": round(olat, 6), "Lon": round(olon, 6), "Typ": "Slutföremål" if i == len(objs) else "Föremål"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    gpx = gpx_text(track_ll, [xy_ll(p.x, p.y, to_ll) for _, p in objs]).encode()
    csv = csv_bytes(track, to_ll, objs)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ GPX", gpx, "sbk_sparforslag.gpx", "application/gpx+xml", use_container_width=True)
    with c2:
        st.download_button("⬇️ CSV", csv, "sbk_sparforslag.csv", "text/csv", use_container_width=True)

    st.info("OSM är endast planeringsunderlag. Kontrollera alltid terrängen, markägare, tillträde, jakt, avverkningar och stängsel på plats.")

st.markdown("---")
st.markdown("<div class='small-note'>Android: öppna appens URL i Chrome → ⋮ → <b>Lägg till på startskärmen</b>. GPS kräver att du tillåter platsåtkomst.</div>", unsafe_allow_html=True)
st.caption("Regelprofilerna är planeringshjälp och ersätter inte aktuell SBK-regelbok eller lokala tävlingsanvisningar.")
