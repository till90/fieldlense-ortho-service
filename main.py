#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_file,
)
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

# Raster stack
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask

# -------------------------------
# Config (Cloud Run friendly)
# -------------------------------

APP_TITLE = os.getenv("APP_TITLE", "Hessen DOP20 – AOI Snapshot (WCS)")

# Try multiple endpoints (because some environments resolve one but not the other)
WCS_URLS = [
    u.strip()
    for u in os.getenv(
        "WCS_URLS",
        "https://inspire.hessen.de/raster/dop20/ows,https://inspire-hessen.de/raster/dop20/ows",
    ).split(",")
    if u.strip()
]

WCS_VERSION = os.getenv("WCS_VERSION", "2.0.1")

# CoverageId: best guess; code also tries to auto-detect from capabilities at runtime
WCS_COVERAGE_ID = os.getenv("WCS_COVERAGE_ID", "he_dop20")

# CRS used for subsetting/buffering (metric). HVBG lists EPSG:25832 for the service scope.
WCS_EPSG = int(os.getenv("WCS_EPSG", "25832"))
WCS_SUBSET_AXES = os.getenv("WCS_SUBSET_AXES", "e,n")  # default per HVBG examples

# Buffer around AOI (meters). Buffer is used as overall raster extent; DOP pixels outside AOI become transparent/nodata.
DEFAULT_BUFFER_M = float(os.getenv("DEFAULT_BUFFER_M", "200"))

# Safety limits (important, DOP20 at 20 cm can explode quickly)
MAX_AOI_AREA_KM2 = float(os.getenv("MAX_AOI_AREA_KM2", "1.0"))  # AOI polygon area limit
MAX_RASTER_DIM_PX = int(os.getenv("MAX_RASTER_DIM_PX", "2048"))  # target max width/height requested via scaling
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

# Temp cache in Cloud Run (ephemeral)
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")) / "aoi_cache"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "40"))

# -------------------------------
# Flask
# -------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# -------------------------------
# Helpers
# -------------------------------

@dataclass
class RenderResult:
    job_id: str
    bounds_wgs84: Tuple[Tuple[float, float], Tuple[float, float]]  # ((south, west), (north, east))
    aoi_area_km2: float
    buffer_m: float
    png_path: Path
    tif_path: Path


def _cleanup_cache() -> None:
    """Best-effort cleanup (Cloud Run /tmp)."""
    try:
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)  # newest first

        # TTL cleanup
        now = time.time()
        for mtime, p in items:
            if now - mtime > CACHE_TTL_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        # size cap cleanup
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)
        if len(items) > MAX_CACHE_ITEMS:
            for _, p in items[MAX_CACHE_ITEMS:]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        # never fail request due to cleanup
        pass


def _parse_geojson(payload: Any) -> Dict[str, Any]:
    if payload is None:
        raise ValueError("Kein GeoJSON übergeben.")
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError("Leerer GeoJSON-String.")
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("GeoJSON muss ein JSON-Objekt oder String sein.")


def _extract_single_geometry(gj: Dict[str, Any]):
    """Accept Feature, FeatureCollection(1), or plain Geometry."""
    t = gj.get("type")
    if t == "Feature":
        geom = gj.get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        if len(feats) != 1:
            raise ValueError("FeatureCollection muss genau 1 Feature enthalten.")
        geom = feats[0].get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t in ("Polygon", "MultiPolygon"):
        return shape(gj)
    raise ValueError(f"Nicht unterstützter GeoJSON-Typ: {t}. Erlaubt: Feature, FeatureCollection(1), Polygon, MultiPolygon.")


def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def _geom_to_epsg(geom, src_epsg: int, dst_epsg: int):
    tr = _transformer(src_epsg, dst_epsg)
    return shp_transform(lambda x, y: tr.transform(x, y), geom)


def _bounds_epsg_to_wgs84(minx: float, miny: float, maxx: float, maxy: float, src_epsg: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    tr = _transformer(src_epsg, 4326)
    west, south = tr.transform(minx, miny)
    east, north = tr.transform(maxx, maxy)
    # Leaflet bounds: [[south, west], [north, east]]
    return (south, west), (north, east)


def _compute_scaled_dims(width_m: float, height_m: float, max_dim_px: int) -> Tuple[int, int]:
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Ungültige Ausdehnung (Breite/Höhe <= 0).")
    # Keep aspect ratio, cap by max_dim_px
    if width_m >= height_m:
        w = max_dim_px
        h = max(1, int(round(max_dim_px * (height_m / width_m))))
    else:
        h = max_dim_px
        w = max(1, int(round(max_dim_px * (width_m / height_m))))
    return w, h


def _pick_working_base_url() -> str:
    # Prefer first; if request fails, try next per-call.
    return WCS_URLS[0]


def _try_get_capabilities(base_url: str) -> Optional[str]:
    # Some servers insist on VERSION; others accept without.
    params = {"SERVICE": "WCS", "REQUEST": "GetCapabilities", "VERSION": WCS_VERSION}
    try:
        r = requests.get(base_url, params=params, timeout=HTTP_TIMEOUT)
        if r.ok and ("Coverage" in r.text or "Capabilities" in r.text):
            return r.text
    except Exception:
        return None
    return None


def _detect_coverage_id_from_caps(xml_text: str) -> Optional[str]:
    # Minimal, robust string-based fallback (no heavy XML libs)
    # Strategy: find tokens that look like coverage ids and contain "dop20"
    low = xml_text.lower()
    if "dop20" not in low:
        return None

    # Naive scan for common tags (<wcs:CoverageId>, <CoverageId>, <gml:identifier>, etc.)
    candidates = []
    for tag in ["coverageid", "wcs:coverageid", "gml:identifier", "identifier"]:
        start = 0
        while True:
            i = low.find(f"<{tag}", start)
            if i == -1:
                break
            j = low.find(">", i)
            k = low.find(f"</{tag}>", j)
            if j != -1 and k != -1:
                val = xml_text[j + 1 : k].strip()
                if val and ("dop20" in val.lower()):
                    candidates.append(val)
            start = k + 1 if k != -1 else j + 1

    # Prefer shortest (often the actual id)
    if candidates:
        candidates.sort(key=lambda s: (len(s), s))
        return candidates[0]

    return None


def _build_getcoverage_params(
    coverage_id: str,
    bbox: Tuple[float, float, float, float],
    subset_axes: Tuple[str, str],
    out_w: Optional[int] = None,
    out_h: Optional[int] = None,
) -> Dict[str, str]:
    minx, miny, maxx, maxy = bbox
    ax_x, ax_y = subset_axes

    params: Dict[str, str] = {
        "SERVICE": "WCS",
        "REQUEST": "GetCoverage",
        "VERSION": WCS_VERSION,
        "COVERAGEID": coverage_id,
        # HVBG examples use FORMAT=GTIFF
        "FORMAT": "GTIFF",
        # CRS URN form used by many INSPIRE services:
        "SUBSETTINGCRS": f"http://www.opengis.net/def/crs/EPSG/0/{WCS_EPSG}",
        "OUTPUTCRS": f"http://www.opengis.net/def/crs/EPSG/0/{WCS_EPSG}",
        # Axis subsetting (e/n per HVBG sample)
        "SUBSET": [
            f"{ax_x}({minx},{maxx})",
            f"{ax_y}({miny},{maxy})",
        ],
    }

    # Scaling extension (best effort). If server rejects, we retry without.
    if out_w and out_h:
        params["SCALESIZE"] = f"i({out_w}),j({out_h})"

    return params


def _request_wcs_geotiff(base_url: str, params: Dict[str, Any]) -> bytes:
    # requests cannot send list values via params dict cleanly unless we pass list-of-tuples
    # Convert params to list-of-tuples, preserving multiple SUBSET keys.
    q = []
    for k, v in params.items():
        if isinstance(v, list):
            for item in v:
                q.append((k, item))
        else:
            q.append((k, v))

    r = requests.get(base_url, params=q, timeout=HTTP_TIMEOUT)
    ct = (r.headers.get("Content-Type") or "").lower()

    if not r.ok:
        # Try to parse meaningful message
        msg = r.text[:1200] if r.text else f"HTTP {r.status_code}"
        raise ValueError(f"WCS-Request fehlgeschlagen ({r.status_code}). Antwort: {msg}")

    # Many servers return application/xml for exceptions; GTIFF is usually image/tiff or application/octet-stream.
    if ("xml" in ct) or ("text" in ct):
        txt = r.text
        # keep it short
        raise ValueError(f"WCS lieferte keine Rasterdaten (Content-Type={ct}). Antwort (Auszug): {txt[:1200]}")

    return r.content


def _render_from_geojson(geojson: Dict[str, Any], buffer_m: float) -> RenderResult:
    _cleanup_cache()

    geom_wgs84 = _extract_single_geometry(geojson)

    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    # area check in EPSG:25832
    geom_utm = _geom_to_epsg(geom_wgs84, 4326, WCS_EPSG)
    aoi_area_m2 = float(geom_utm.area)
    aoi_area_km2 = aoi_area_m2 / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    # buffered bbox determines raster extent; we mask to AOI later
    geom_buffered = geom_utm.buffer(buffer_m)
    minx, miny, maxx, maxy = geom_buffered.bounds
    bbox = (minx, miny, maxx, maxy)

    width_m = maxx - minx
    height_m = maxy - miny
    out_w, out_h = _compute_scaled_dims(width_m, height_m, MAX_RASTER_DIM_PX)

    # choose coverage id: env default, but try capabilities autodetect
    coverage_id = WCS_COVERAGE_ID
    subset_axes = tuple([s.strip() for s in WCS_SUBSET_AXES.split(",")][:2])
    if len(subset_axes) != 2:
        subset_axes = ("e", "n")

    # Try endpoints until one works
    last_err = None
    geotiff_bytes = None
    used_base = None

    for base in WCS_URLS:
        used_base = base
        # try to autodetect coverage id if possible (optional)
        try:
            caps = _try_get_capabilities(base)
            if caps:
                det = _detect_coverage_id_from_caps(caps)
                if det:
                    coverage_id = det
        except Exception:
            pass

        # 1) try with scaling
        try:
            params = _build_getcoverage_params(coverage_id, bbox, subset_axes, out_w=out_w, out_h=out_h)
            geotiff_bytes = _request_wcs_geotiff(base, params)
            break
        except Exception as e:
            last_err = e

        # 2) retry without scaling (some servers don't support SCALESIZE)
        try:
            params = _build_getcoverage_params(coverage_id, bbox, subset_axes, out_w=None, out_h=None)
            geotiff_bytes = _request_wcs_geotiff(base, params)
            break
        except Exception as e:
            last_err = e

    if geotiff_bytes is None:
        raise ValueError(f"WCS konnte nicht abgefragt werden. Letzter Fehler: {last_err}")

    job_id = uuid.uuid4().hex[:12]
    raw_tif = TMP_DIR / f"{job_id}.raw.tif"
    out_tif = TMP_DIR / f"{job_id}.aoi.tif"
    out_png = TMP_DIR / f"{job_id}.aoi.png"

    raw_tif.write_bytes(geotiff_bytes)

    # Mask to AOI (keep full buffered extent, transparent/nodata outside AOI)
    with MemoryFile(geotiff_bytes) as memfile:
        with memfile.open() as src:
            # Use AOI in same CRS as raster
            aoi_geom = geom_utm
            # rasterio.mask expects geojson mappings
            geoms = [mapping(aoi_geom)]

            # Keep extent (crop=False), but mask outside AOI
            masked, out_transform = rio_mask(src, geoms, crop=False, filled=False)

            # Prefer RGB from first 3 bands; if fewer, replicate
            data = masked.data
            msk = masked.mask

            if data.ndim != 3:
                raise ValueError("Unerwartete Raster-Dimensionen.")

            bands = data.shape[0]
            if bands >= 3:
                rgb = data[:3].copy()
                rgb_mask = msk[:3]
            elif bands == 1:
                rgb = np.repeat(data[:1], 3, axis=0).copy()
                rgb_mask = np.repeat(msk[:1], 3, axis=0)
            else:
                raise ValueError(f"Unerwartete Bandanzahl: {bands}")

            # Build alpha: valid if NOT masked in all RGB bands
            valid = ~np.all(rgb_mask, axis=0)
            alpha = (valid.astype(np.uint8) * 255)

            # Fill masked pixels in rgb with 0
            for b in range(3):
                rgb[b][rgb_mask[b]] = 0

            # Convert to uint8 if needed
            if rgb.dtype != np.uint8:
                # robust scaling by dtype range
                rgb = rgb.astype(np.float32)
                # heuristic: clamp 0..255
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            # Write GeoTIFF as RGBA (4 bands)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=4,
                dtype=rasterio.uint8,
                nodata=0,
                compress="deflate",
                tiled=True,
                interleave="pixel",
            )

            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(rgb.astype(np.uint8), indexes=[1, 2, 3])
                dst.write(alpha.astype(np.uint8), indexes=4)

            # Write PNG (RGBA)
            rgba = np.dstack(
                [
                    np.transpose(rgb, (1, 2, 0)),
                    alpha,
                ]
            )
            Image.fromarray(rgba, mode="RGBA").save(out_png)

    # bounds for leaflet overlay in WGS84
    bounds_wgs84 = _bounds_epsg_to_wgs84(minx, miny, maxx, maxy, WCS_EPSG)

    return RenderResult(
        job_id=job_id,
        bounds_wgs84=bounds_wgs84,
        aoi_area_km2=aoi_area_km2,
        buffer_m=buffer_m,
        png_path=out_png,
        tif_path=out_tif,
    )


# -------------------------------
# Routes
# -------------------------------

INDEX_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />

  <style>
    :root{
      --bg:#0b0f19;
      --card:#111a2e;
      --text:#e6eaf2;
      --muted:#a8b3cf;
      --border: rgba(255,255,255,.10);
      --primary:#6ea8fe;
      --focus: rgba(110,168,254,.45);
      --radius: 16px;
      --container: 1200px;
      --gap: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    body{
      margin:0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
    }
    .wrap{
      max-width: var(--container);
      margin: 18px auto;
      padding: 0 14px 24px;
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: var(--gap);
    }
    header{
      max-width: var(--container);
      margin: 18px auto 0;
      padding: 0 14px;
      display:flex;
      align-items:baseline;
      justify-content:space-between;
      gap: 12px;
    }
    h1{
      font-size: 18px;
      margin:0;
      letter-spacing: .2px;
    }
    .hint{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }
    .card{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 18px 60px rgba(0,0,0,.35);
      overflow: hidden;
    }
    #map{
      height: 70vh;
      min-height: 520px;
    }
    .panel{
      padding: 12px;
      display:flex;
      flex-direction:column;
      gap: 10px;
    }
    label{
      color: var(--muted);
      font-size: 12px;
    }
    textarea{
      width: 100%;
      min-height: 220px;
      resize: vertical;
      background: rgba(255,255,255,.04);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      outline: none;
    }
    textarea:focus{
      border-color: var(--primary);
      box-shadow: 0 0 0 4px var(--focus);
    }
    .row{
      display:flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    button{
      appearance:none;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary{
      border-color: rgba(110,168,254,.35);
      background: rgba(110,168,254,.16);
    }
    button:disabled{
      opacity:.55;
      cursor:not-allowed;
    }
    .status{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(0,0,0,.18);
      border: 1px solid var(--border);
    }
    .status b{ color: var(--text); }
    .err{
      border-color: rgba(255,100,100,.35);
      background: rgba(255,100,100,.10);
      color: #ffd1d1;
    }
    .ok{
      border-color: rgba(120,220,160,.35);
      background: rgba(120,220,160,.08);
    }
    .small{
      font-size: 12px;
      color: var(--muted);
    }
    a{ color: var(--primary); text-decoration: none; }
    a:hover{ text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ title }}</h1>
      <div class="hint">Zeichne ein Polygon oder Rechteck. Es ist immer nur <b>ein</b> Feature aktiv – neues Feature ersetzt das alte. DOP20 wird nur <b>innerhalb</b> der AOI angezeigt; außerhalb (Buffer) bleibt OSM sichtbar.</div>
    </div>
    <div class="small">API: <code>/api/geotiff</code> oder <code>/api/png</code></div>
  </header>

  <div class="wrap">
    <div class="card">
      <div id="map"></div>
    </div>

    <div class="card">
      <div class="panel">
        <div class="row">
          <button id="btn-clear">AOI löschen</button>
          <button class="primary" id="btn-render" disabled>Vorschau laden</button>
        </div>

        <div class="row">
          <label>Buffer (m): <input id="buf" type="number" min="0" step="10" value="{{ default_buffer }}" style="width:120px; margin-left:8px; padding:8px; border-radius:10px; border:1px solid var(--border); background: rgba(255,255,255,.04); color:var(--text);"></label>
        </div>

        <div id="status" class="status">Noch keine AOI.</div>

        <div class="row">
          <button id="btn-geojson" disabled>GeoJSON herunterladen</button>
          <button id="btn-png" disabled>PNG herunterladen</button>
          <button id="btn-tif" disabled>GeoTIFF herunterladen</button>
        </div>

        <label>GeoJSON (aktuelles Feature, EPSG:4326)</label>
        <textarea id="geojson" spellcheck="false" placeholder="Hier erscheint das GeoJSON…"></textarea>

        <div class="small">
          Hinweis: Große AOIs können vom WCS abgelehnt werden oder sehr lange dauern. Limit serverseitig: <b>{{ max_area_km2 }} km²</b>.
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <script>
    const map = L.map('map', { preferCanvas: true }).setView([50.55, 9.0], 8);

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const drawn = new L.FeatureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      draw: {
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
        polygon: { allowIntersection: false, showArea: true },
        rectangle: true
      },
      edit: {
        featureGroup: drawn,
        edit: true,
        remove: false
      }
    });
    map.addControl(drawControl);

    let overlay = null;
    let currentFeature = null;
    let currentJob = null;

    const elGeo = document.getElementById('geojson');
    const elStatus = document.getElementById('status');
    const btnRender = document.getElementById('btn-render');
    const btnClear = document.getElementById('btn-clear');
    const btnGJ = document.getElementById('btn-geojson');
    const btnPNG = document.getElementById('btn-png');
    const btnTIF = document.getElementById('btn-tif');
    const elBuf = document.getElementById('buf');

    function setStatus(html, cls){
      elStatus.className = 'status' + (cls ? (' ' + cls) : '');
      elStatus.innerHTML = html;
    }

    function setButtons(hasFeature, hasJob){
      btnRender.disabled = !hasFeature;
      btnGJ.disabled = !hasFeature;
      btnPNG.disabled = !hasJob;
      btnTIF.disabled = !hasJob;
    }

    function clearAll(){
      drawn.clearLayers();
      currentFeature = null;
      currentJob = null;
      elGeo.value = '';
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      setButtons(false, false);
      setStatus('Noch keine AOI.', '');
    }

    function featureToGeoJSON(layer){
    return {
        type: "Feature",
        properties: { epsg: 4326 },
        geometry: layer.toGeoJSON().geometry
    };
}

    function downloadText(filename, text){
      const blob = new Blob([text], {type: 'application/json;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    map.on(L.Draw.Event.CREATED, function (e) {
      // enforce single feature
      drawn.clearLayers();
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      currentJob = null;

      const layer = e.layer;
      drawn.addLayer(layer);
      currentFeature = layer;

      const gj = featureToGeoJSON(layer);
      elGeo.value = JSON.stringify(gj, null, 2);

      setButtons(true, false);
      setStatus('AOI gesetzt. Klicke <b>Vorschau laden</b>, um DOP20 zu holen.', 'ok');
    });

    map.on('draw:edited', function(){
      if(!currentFeature) return;
      const layers = drawn.getLayers();
      if(layers.length < 1) return;
      currentFeature = layers[0];
      const gj = featureToGeoJSON(currentFeature);
      elGeo.value = JSON.stringify(gj, null, 2);
      currentJob = null;
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      setButtons(true, false);
      setStatus('AOI geändert. Bitte <b>Vorschau laden</b> erneut ausführen.', 'ok');
    });

    btnClear.addEventListener('click', clearAll);

    btnGJ.addEventListener('click', () => {
      if(!currentFeature) return;
      downloadText('aoi.geojson', elGeo.value);
    });

    async function renderPreview(){
      if(!currentFeature) return;
      if (btnRender.dataset.busy === "1") return;
      btnRender.dataset.busy = "1";
      btnRender.disabled = true;
      setButtons(true, false);
      setStatus('Lade DOP20 via WCS…', '');

      const buffer_m = Number(elBuf.value || 0);

      let gj;
      try{
        gj = JSON.parse(elGeo.value);
      }catch(err){
        setStatus('GeoJSON ist ungültig.', 'err');
        return;
      }finally {
        btnRender.dataset.busy = "0";
        btnRender.disabled = !currentFeature;
      }

      try{
        const res = await fetch('/api/preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ geojson: gj, buffer_m })
        });

        const ct = (res.headers.get('content-type') || '').toLowerCase();
        const raw = await res.text();

        let data = null;
        if (ct.includes('application/json')) {
          data = raw ? JSON.parse(raw) : {};
        } else {
          // Hier siehst du künftig den echten Cloud-Run/Proxy-Fehlerauszug
          throw new Error(`Server lieferte kein JSON (HTTP ${res.status}, Content-Type=${ct}). Antwort-Auszug: ${raw.slice(0, 240)}`);
        }

        if (!res.ok) {
          throw new Error((data && data.error) ? data.error : (`HTTP ${res.status}`));
        }

        currentJob = data.job_id;

        if(overlay){
          map.removeLayer(overlay);
          overlay = null;
        }
        const b = data.overlay.bounds; // [[south, west],[north, east]]
        overlay = L.imageOverlay(data.overlay.url, b, { opacity: 1.0, interactive: false });
        overlay.addTo(map);

        const fit = L.latLngBounds(b);
        map.fitBounds(fit.pad(0.15));

        setButtons(true, true);
        setStatus(
          `DOP20 geladen. <b>AOI</b>: ${data.aoi_area_km2.toFixed(3)} km² · <b>Buffer</b>: ${data.buffer_m.toFixed(0)} m`,
          'ok'
        );

        // wire download buttons
        btnPNG.onclick = () => { window.location = data.download.png; };
        btnTIF.onclick = () => { window.location = data.download.geotiff; };

      }catch(err){
        setButtons(true, false);
        setStatus('Fehler: ' + (err && err.message ? err.message : String(err)), 'err');
      }
    }

    btnRender.addEventListener('click', renderPreview);

    // init
    clearAll();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        default_buffer=int(DEFAULT_BUFFER_M),
        max_area_km2=MAX_AOI_AREA_KM2,
    )


@app.post("/api/preview")
def api_preview():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        buffer_m = float(body.get("buffer_m", DEFAULT_BUFFER_M))

        rr = _render_from_geojson(gj, buffer_m)

        sw, ne = rr.bounds_wgs84  # ((south, west), (north, east))
        return jsonify(
            {
                "job_id": rr.job_id,
                "aoi_area_km2": rr.aoi_area_km2,
                "buffer_m": rr.buffer_m,
                "overlay": {
                    "url": f"/r/{rr.job_id}/overlay.png",
                    "bounds": [[sw[0], sw[1]], [ne[0], ne[1]]],
                },
                "download": {
                    "png": f"/r/{rr.job_id}/aoi.png",
                    "geotiff": f"/r/{rr.job_id}/aoi.tif",
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/geotiff")
def api_geotiff():
    """
    API: GeoJSON rein -> GeoTIFF (RGBA, Buffer-Extent, DOP nur innerhalb AOI; außerhalb nodata/transparent)
    Body: { "geojson": <Feature|FeatureCollection(1)|Polygon>, "buffer_m": 200 }
    """
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        buffer_m = float(body.get("buffer_m", DEFAULT_BUFFER_M))

        rr = _render_from_geojson(gj, buffer_m)
        return send_file(
            rr.tif_path,
            mimetype="image/tiff",
            as_attachment=True,
            download_name=f"aoi_{rr.job_id}_epsg{WCS_EPSG}.tif",
            conditional=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/png")
def api_png():
    """
    API: GeoJSON rein -> PNG (RGBA Overlay; gleiche Logik wie Web)
    """
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        buffer_m = float(body.get("buffer_m", DEFAULT_BUFFER_M))

        rr = _render_from_geojson(gj, buffer_m)
        return send_file(
            rr.png_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"aoi_{rr.job_id}.png",
            conditional=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/r/<job_id>/overlay.png")
def job_overlay(job_id: str):
    p = TMP_DIR / f"{job_id}.aoi.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    # overlay should be cacheable per job_id
    return send_file(p, mimetype="image/png", as_attachment=False, conditional=True)


@app.get("/r/<job_id>/aoi.png")
def job_png(job_id: str):
    p = TMP_DIR / f"{job_id}.aoi.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=True, download_name=f"aoi_{job_id}.png", conditional=True)


@app.get("/r/<job_id>/aoi.tif")
def job_tif(job_id: str):
    p = TMP_DIR / f"{job_id}.aoi.tif"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(
        p,
        mimetype="image/tiff",
        as_attachment=True,
        download_name=f"aoi_{job_id}_epsg{WCS_EPSG}.tif",
        conditional=True,
    )

@app.get("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
