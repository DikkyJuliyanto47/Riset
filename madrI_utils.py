# madrI_utils.py
import os
import json
import math
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import transform

from madrI_config import BASE_URL_OSRM

# -------- OSRM caching simple ----------
_osrm_cache = {}


def load_osrm_cache(path="./madrI_osrm_cache.json"):
    global _osrm_cache
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                _osrm_cache = json.load(f)
            print(f"Loaded OSRM cache ({len(_osrm_cache)} entries).")
        except Exception:
            _osrm_cache = {}
    else:
        _osrm_cache = {}


def save_osrm_cache(path="./madrI_osrm_cache.json"):
    with open(path, "w") as f:
        json.dump(_osrm_cache, f, indent=2)


def get_osrm_distance_cached(lat1, lon1, lat2, lon2):
    """Return distance in kilometers. Uses simple JSON cache."""
    key = f"{lat1:.6f},{lon1:.6f}__{lat2:.6f},{lon2:.6f}"
    if key in _osrm_cache:
        return float(_osrm_cache[key])
    try:
        url = f"{BASE_URL_OSRM}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        dist_km = data["routes"][0]["distance"] / 1000.0
        _osrm_cache[key] = dist_km
        return float(dist_km)
    except Exception:
        # fallback: haversine approximate (not routing)
        return _haversine_km(lat1, lon1, lat2, lon2)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -------- Data loaders (minimal, self-contained) ----------
def load_flood_polygons(path):
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        print(f"Loaded flood polygons: {len(gdf)}")
        return gdf
    except Exception as e:
        print("Error load_flood_polygons:", e)
        return None


def load_evac_candidates_geojson(path):
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        out = []
        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            name = (
                row.get("name") or row.get("Nama") or row.get("Evakuasi") or f"evac_{i}"
            )
            out.append({"coord": (geom.y, geom.x), "name": str(name)})
        print(f"Loaded evac candidates: {len(out)}")
        return out
    except Exception as e:
        print("Error load_evac_candidates_geojson:", e)
        return []


def sample_users_from_shp(shp_path, n):
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        bounds = gdf.total_bounds  # xmin,ymin,xmax,ymax in projected units
        xmin, ymin, xmax, ymax = bounds
        pts = []
        import random

        while len(pts) < n:
            rx = random.uniform(xmin, xmax)
            ry = random.uniform(ymin, ymax)
            p = Point(rx, ry)
            # check within any polygon
            if gdf.contains(p).any():
                pts.append((p.y, p.x))
        return pts
    except Exception as e:
        print("Error sample_users_from_shp:", e)
        return []


def get_elevation_at_point(tif_path, lat, lon):
    try:
        with rasterio.open(tif_path) as src:
            if src.crs.to_string() != "EPSG:4326":
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat
            row, col = src.index(x, y)
            data = src.read(1)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                elev = float(data[row, col])
                if elev < -100:
                    return 0.0
                return elev
            return 0.0
    except Exception:
        return 0.0


def is_coord_in_flood_zone(flood_gdf, lat, lon):
    if flood_gdf is None or flood_gdf.empty:
        return False
    pt = Point(lon, lat)
    return flood_gdf.geometry.contains(pt).any()
