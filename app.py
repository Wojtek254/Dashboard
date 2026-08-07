# app.py
# pip install streamlit folium earthengine-api streamlit-folium pandas altair google-auth numpy

import datetime as dt
import io
import re
import urllib.request

import numpy as np

import altair as alt
import ee
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Draw, SideBySideLayers
from google.oauth2 import service_account
from PIL import Image, ImageDraw, ImageFont
from streamlit_folium import st_folium

# ---------------------------------------------
# STREAMLIT PAGE SETUP
# ---------------------------------------------
st.set_page_config(
    page_title="CYGNSS – Regional Viewer (5-band inundation & anomalies)",
    layout="wide",
)

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
PROJECT_ID = st.secrets["project_id"]
ASSET_FOLDER = f"projects/{PROJECT_ID}/assets"

# Asset naming convention produced by upload_to_gee.py:
#   inundation_5bands_{YEAR}_{DAY_OF_YEAR}
# e.g. inundation_5bands_2025_305
ASSET_NAME_RE = re.compile(r"^inundation_5bands_(\d{4})_(\d{1,3})$")

CENTER = [0.5, 108.0]
ZOOM = 6

# Color palette for inundation
PALETTE_INUND = [
    "#e3f2fd",
    "#bbdefb",
    "#90caf9",
    "#64b5f6",
    "#42a5f5",
    "#1e88e5",
    "#0d47a1",
]

# Color palette for anomalies
PALETTE_ANOM = [
    "#8c2d04",
    "#fe9929",
    "#fee6ce",
    "#e3f2fd",
    "#90caf9",
    "#42a5f5",
    "#0d47a1",
]

BAND_OPTIONS = {
    1: "Daily observations",
    2: "3-daily interpolated data",
    3: "Interpolation flux",
    4: "1-day inundation anomalies",
    5: "Interpolated inundation anomalies",
}

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
NDVI_COLLECTION = "MODIS/061/MOD13Q1"
POP_COLLECTION = "CIESIN/GPWv411/GPW_Population_Density"
ELEVATION_IMAGE = "USGS/SRTMGL1_003"

LAYER_OPTIONS = {
    "none": "None",
    "cygnss_1": "CYGNSS – Daily observations",
    "cygnss_2": "CYGNSS – 3-daily interpolated data",
    "cygnss_3": "CYGNSS – Interpolation flux",
    "cygnss_4": "CYGNSS – 1-day inundation anomalies",
    "cygnss_5": "CYGNSS – Interpolated inundation anomalies",
    "chirps": "CHIRPS precipitation",
    "ndvi": "NDVI",
    "population_density": "Population density",
    "elevation": "Elevation a.s.l.",
}

OVERLAY_LEGENDS = {
    "chirps": {
        "title": "CHIRPS precipitation",
        "palette": ["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
        "min": 0,
        "max": 20,
        "unit": "mm",
    },
    "ndvi": {
        "title": "NDVI",
        "palette": ["#f7fcf5", "#a1d99b", "#31a354", "#006d2c"],
        "min": 0.0,
        "max": 0.8,
        "unit": "-",
    },
    "population_density": {
        "title": "Population density",
        "palette": ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
        "min": 0,
        "max": 1000,
        "unit": "people/km²",
    },
    "elevation": {
        "title": "Elevation",
        "palette": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
        "min": 0,
        "max": 3000,
        "unit": "m a.s.l.",
    },
}
def band_kind(band_number: int) -> str:
    return "anomaly" if band_number in (4, 5) else "inundation"


# ---------------------------------------------
# INITIALIZE GOOGLE EARTH ENGINE
# ---------------------------------------------
def ensure_ee():
    """
    Initialize Earth Engine using service account credentials
    stored in Streamlit secrets.
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(credentials=credentials, project=PROJECT_ID)
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {e}")
        st.stop()


@st.cache_data(ttl=600, show_spinner="Skanowanie dostepnych assetow w Earth Engine...")
def discover_available_days(_ee_ready_marker=None):
    """
    List all assets in ASSET_FOLDER, parse names matching
    inundation_5bands_{YEAR}_{DOY}, and return sorted info for
    every day that is actually available (across any number of years).

    A composite "day_key" (year * 1000 + doy) is used everywhere downstream
    instead of the raw day-of-year, so that data spanning multiple years
    never collides (e.g. 2020 doy 10 vs 2021 doy 10).
    """
    try:
        asset_ids = []
        page_token = None
        while True:
            params = {"parent": ASSET_FOLDER, "pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            listing = ee.data.listAssets(params)
            asset_ids.extend(a["id"] for a in listing.get("assets", []))
            page_token = listing.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        return [], f"Nie udalo sie wylistowac assetow w {ASSET_FOLDER}: {e}"

    days_info = []
    for asset_id in asset_ids:
        short_name = asset_id.split("/")[-1]
        m = ASSET_NAME_RE.match(short_name)
        if not m:
            continue
        year, doy = int(m.group(1)), int(m.group(2))
        try:
            date_obj = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            continue
        days_info.append(
            {
                "year": year,
                "doy": doy,
                "day_key": year * 1000 + doy,
                "date": date_obj.isoformat(),
                "asset_id": asset_id,
            }
        )

    days_info.sort(key=lambda d: d["day_key"])
    return days_info, None


def test_asset_access(days_info):
    """
    Test whether the app can read one known asset.
    This helps separate IAM/auth errors from rendering errors.
    """
    if not days_info:
        st.error(
            f"Nie znaleziono zadnych assetow pasujacych do wzorca "
            f"'inundation_5bands_{{YEAR}}_{{DOY}}' w folderze {ASSET_FOLDER}."
        )
        st.stop()

    test_info = days_info[0]
    test_path = test_info["asset_id"]
    try:
        img = ee.Image(test_path)
        info = img.getInfo()
        bands = [b.get("id", f"band_{i}") for i, b in enumerate(info.get("bands", []))]
        st.success(
            f"Earth Engine initialized. Found {len(days_info)} available day(s), "
            f"from {days_info[0]['date']} to {days_info[-1]['date']}."
        )
        with st.expander("Debug: Earth Engine connection"):
            st.write("PROJECT_ID:", PROJECT_ID)
            st.write("ASSET_FOLDER:", ASSET_FOLDER)
            st.write("Test asset:", test_path)
            st.write("Bands:", bands)
            st.write("Total days found:", len(days_info))
    except Exception as e:
        st.error(f"Asset access failed for {test_path}: {e}")
        st.stop()


ensure_ee()
DAYS_INFO, discovery_error = discover_available_days()
if discovery_error:
    st.error(discovery_error)
    st.stop()
if not st.session_state.get("ee_asset_access_tested", False):
    test_asset_access(DAYS_INFO)
    st.session_state.ee_asset_access_tested = True

# Lookup helpers built from the discovered assets
DAY_KEY_TO_INFO = {info["day_key"]: info for info in DAYS_INFO}
DATE_TO_DAY_KEY = {dt.date.fromisoformat(info["date"]): info["day_key"] for info in DAYS_INFO}
DAY_KEY_TO_LABEL = {info["day_key"]: info["date"] for info in DAYS_INFO}

# Available date range in the dataset (spans all years found)
MIN_DATE = dt.date.fromisoformat(DAYS_INFO[0]["date"])
MAX_DATE = dt.date.fromisoformat(DAYS_INFO[-1]["date"])


# ---------------------------------------------
# IMAGE COLLECTIONS
# ---------------------------------------------
def build_inund_collection():
    """
    Build image collection from all discovered asset days.
    """
    imgs = []
    for info in DAYS_INFO:
        img = ee.Image(info["asset_id"]).set("day_key", info["day_key"])
        imgs.append(img)
    return ee.ImageCollection(imgs)


IC_INUND = build_inund_collection()


def get_collection(kind: str):
    """
    Return the correct collection for the selected data type.
    """
    return IC_INUND


# ---------------------------------------------
# HELPER FUNCTIONS – CYGNSS BANDS
# ---------------------------------------------
def cygnss_valid_raw_band(img, band_index):
    """Return a CYGNSS band masked where the dataset uses 255 as no-data."""
    band = img.select(band_index)
    return band.updateMask(band.lt(255))


def cygnss_scaled_band(img, band_index):
    """
    Return a display/analysis-ready CYGNSS band.

    Bands 4 and 5 are anomaly bands encoded with +100 offset,
    so here they are converted back to real anomaly values by subtracting 100.
    Other bands are kept unchanged.
    """
    band_number = band_index + 1
    band = cygnss_valid_raw_band(img, band_index)

    if band_number in (4, 5):
        band = band.subtract(100)

    return band.rename("value")


def cygnss_thresholded_band(img, band_index, thr_min, thr_max):
    band = cygnss_scaled_band(img, band_index)
    mask = band.gte(thr_min).And(band.lte(thr_max))
    return band.updateMask(mask)


# Backward-compatible names used later in the script
def mask_inund_band(img, band_index, thr_min, thr_max):
    return cygnss_thresholded_band(img, band_index, thr_min, thr_max)


def inund_valid_band(img, band_index):
    return cygnss_valid_raw_band(img, band_index)


def anomaly_valid_band(img, band_index):
    return cygnss_scaled_band(img, band_index)


def anomaly_thresholded(img, band_index, thr_min, thr_max):
    return cygnss_thresholded_band(img, band_index, thr_min, thr_max)


# ---------------------------------------------
# DATE HELPERS
# ---------------------------------------------
def parse_date_range(date_value):
    if isinstance(date_value, tuple):
        if len(date_value) != 2 or date_value[0] is None or date_value[1] is None:
            return None, None
        start_date, end_date = date_value
    else:
        start_date = end_date = date_value

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start_date = max(start_date, MIN_DATE)
    end_date = min(end_date, MAX_DATE)
    return start_date, end_date


def dates_to_doys(start_date, end_date):
    """
    Expand a calendar date range into the list of calendar dates, plus the
    sorted list of "day_key" values that actually have a matching asset
    (gaps in the data are simply skipped). Works across any number of years.
    """
    selected_dates = []
    current_date = start_date
    while current_date <= end_date:
        selected_dates.append(current_date)
        current_date += dt.timedelta(days=1)

    day_keys = [
        DATE_TO_DAY_KEY[d] for d in selected_dates if d in DATE_TO_DAY_KEY
    ]
    return selected_dates, sorted(day_keys)


# ---------------------------------------------
# BUILD MEAN IMAGE FOR MAP DISPLAY
# ---------------------------------------------
def build_mean_image(selected_days, thr_min, thr_max, kind, band_index):
    """
    Compute pixel-wise mean image over selected days after masking.
    """
    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day_key", selected_days))

    # This uses real values for all bands. For anomaly bands (4/5),
    # the encoded +100 offset is removed before thresholding.
    ic_proc = ic_sel.map(lambda img: cygnss_thresholded_band(img, band_index, thr_min, thr_max))

    stacked = ic_proc.toBands()
    pixel_mean = stacked.reduce(ee.Reducer.mean())
    return pixel_mean


def is_cygnss_layer(layer_name: str) -> bool:
    return isinstance(layer_name, str) and layer_name.startswith("cygnss_")


def cygnss_band_number(layer_name: str) -> int:
    return int(layer_name.split("_")[1])


def cygnss_layer_kind(layer_name: str) -> str:
    return band_kind(cygnss_band_number(layer_name))


def cygnss_layer_label(layer_name: str) -> str:
    return BAND_OPTIONS[cygnss_band_number(layer_name)]


def cygnss_unit(kind: str) -> str:
    # CYGNSS inundation and anomaly values are shown on the same percentage scale.
    # Anomaly bands are decoded first by subtracting the +100 storage offset.
    return "%"


def cygnss_legend(layer_name, thr_min, thr_max):
    kind = cygnss_layer_kind(layer_name)
    return {
        "title": f"CYGNSS – {cygnss_layer_label(layer_name)}",
        "palette": PALETTE_ANOM if kind == "anomaly" else PALETTE_INUND,
        "min": thr_min,
        "max": thr_max,
        "unit": cygnss_unit(kind),
    }


def get_layer_legend(layer_name, thr_min=None, thr_max=None):
    if layer_name == "none":
        return None
    if is_cygnss_layer(layer_name):
        return cygnss_legend(layer_name, thr_min, thr_max)
    return OVERLAY_LEGENDS[layer_name]


def build_contours(img, vis, n_levels=10):
    """
    Build simple value-based contour lines. Each contour level gets a color
    sampled from the same palette as the shading layer.
    """
    vmin = float(vis["min"])
    vmax = float(vis["max"])
    palette = vis["palette"]

    if vmax <= vmin:
        vmax = vmin + 1

    step = (vmax - vmin) / n_levels
    contour_imgs = []

    for i in range(n_levels + 1):
        level = vmin + i * step
        palette_idx = round(i * (len(palette) - 1) / n_levels)
        color = palette[min(palette_idx, len(palette) - 1)]

        contour = (
            img.subtract(level)
            .zeroCrossing()
            .selfMask()
            .visualize(
                min=0,
                max=1,
                palette=[color],
                opacity=0.9,
            )
        )
        contour_imgs.append(contour)

    return ee.ImageCollection(contour_imgs).mosaic()


def build_cygnss_image(layer_name, selected_days, thr_min, thr_max, mode="shading"):
    band_number = cygnss_band_number(layer_name)
    kind = band_kind(band_number)
    band_index = band_number - 1

    img = build_mean_image(selected_days, thr_min, thr_max, kind, band_index)
    vis = cygnss_legend(layer_name, thr_min, thr_max)

    if mode == "contour":
        return build_contours(img, vis)

    return img.select(0).visualize(
        min=vis["min"],
        max=vis["max"],
        palette=vis["palette"],
        opacity=0.75,
    )


def build_external_layer_image(layer_name, start_date, end_date, mode="shading"):
    """
    Build non-CYGNSS layer image.
    mode="shading" returns a semi-transparent raster.
    mode="contour" returns value-based contour lines colored with the same scale.
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_exclusive = (end_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    if layer_name == "none":
        return None

    if layer_name == "chirps":
        img = (
            ee.ImageCollection(CHIRPS_COLLECTION)
            .filterDate(start_str, end_exclusive)
            .select("precipitation")
            .sum()
        )
        vis = OVERLAY_LEGENDS["chirps"]

    elif layer_name == "ndvi":
        img = (
            ee.ImageCollection(NDVI_COLLECTION)
            .filterDate(start_str, end_exclusive)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
        )
        vis = OVERLAY_LEGENDS["ndvi"]

    elif layer_name == "population_density":
        img = (
            ee.ImageCollection(POP_COLLECTION)
            .sort("system:time_start", False)
            .first()
            .select("population_density")
        )
        vis = OVERLAY_LEGENDS["population_density"]

    elif layer_name == "elevation":
        img = ee.Image(ELEVATION_IMAGE).select("elevation")
        vis = OVERLAY_LEGENDS["elevation"]

    else:
        return None

    if mode == "contour":
        return build_contours(img, vis)

    return img.visualize(
        min=vis["min"],
        max=vis["max"],
        palette=vis["palette"],
        opacity=0.55,
    )


def empty_visual_image():
    return ee.Image(0).selfMask().visualize(
        min=0,
        max=1,
        palette=["#000000"],
        opacity=0.0,
    )


def build_layer_image(
    layer_name,
    selected_days,
    thr_min,
    thr_max,
    start_date,
    end_date,
    mode="shading",
):
    if layer_name == "none":
        return None
    if is_cygnss_layer(layer_name):
        return build_cygnss_image(layer_name, selected_days, thr_min, thr_max, mode=mode)
    return build_external_layer_image(layer_name, start_date, end_date, mode=mode)


def build_side_visual_image(
    selected_days,
    thr_min,
    thr_max,
    start_date,
    end_date,
    shading_layer,
    contour_layer,
):
    """
    Build one side of the map. CYGNSS is no longer mandatory:
    it can be selected as shading, contour, or not selected at all.
    """
    base = build_layer_image(
        shading_layer,
        selected_days,
        thr_min,
        thr_max,
        start_date,
        end_date,
        mode="shading",
    )

    if base is None:
        base = empty_visual_image()

    contour = build_layer_image(
        contour_layer,
        selected_days,
        thr_min,
        thr_max,
        start_date,
        end_date,
        mode="contour",
    )

    if contour is not None:
        base = base.blend(contour)

    return base


def selected_cygnss_layer(shading_layer, contour_layer):
    """Return the first CYGNSS layer selected on a side, used for statistics."""
    if is_cygnss_layer(shading_layer):
        return shading_layer
    if is_cygnss_layer(contour_layer):
        return contour_layer
    return None
# LOCAL BBOX STATISTICS (CACHE)
# ---------------------------------------------
# Statistics are intentionally calculated locally. Earth Engine is used only
# to crop/download the selected CYGNSS band for the user's rectangle. This
# avoids running reduceRegion repeatedly for every day.
STATS_SCALE_M = 3000
DOWNLOAD_DAYS_PER_CHUNK = 20


def _download_ee_npy(image, region):
    """Download a small EE image as a NumPy array."""
    url = image.getDownloadURL(
        {
            "region": region,
            "scale": STATS_SCALE_M,
            "crs": "EPSG:6933",
            "format": "NPY",
        }
    )
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _npy_to_band_arrays(arr, expected_count):
    """Normalize EE's NPY response into an ordered list of 2-D arrays."""
    if arr.dtype.names:
        arrays = [np.asarray(arr[name]) for name in arr.dtype.names]
    elif arr.ndim == 2:
        arrays = [arr]
    elif arr.ndim == 3:
        # Earth Engine normally returns structured NPY for multiband images,
        # but support both common plain-array layouts as a safeguard.
        if arr.shape[-1] == expected_count:
            arrays = [arr[..., i] for i in range(expected_count)]
        elif arr.shape[0] == expected_count:
            arrays = [arr[i, ...] for i in range(expected_count)]
        else:
            raise ValueError(f"Unexpected NPY shape from Earth Engine: {arr.shape}")
    else:
        raise ValueError(f"Unexpected NPY response from Earth Engine: shape={arr.shape}")

    if len(arrays) != expected_count:
        raise ValueError(
            f"Earth Engine returned {len(arrays)} band(s), expected {expected_count}."
        )
    return arrays


@st.cache_data(show_spinner=False)
def compute_region_stats_local(
    selected_days_tuple,
    thr_min,
    thr_max,
    xmin,
    ymin,
    xmax,
    ymax,
    kind,
    band_index,
):
    """
    Download only the selected bbox and compute every regional statistic in
    NumPy. Earth Engine does no per-day reduceRegion work here.

    Returns:
      (summary_min, summary_max, summary_mean,
       period_inrange_count, period_valid_count, daily_rows)
    """
    selected_days = sorted(int(d) for d in selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326", geodesic=False)

    # Running per-pixel accumulators reproduce EE ImageCollection.mean() after
    # threshold masking, without keeping the full time x y cube in memory.
    pixel_sum = None
    pixel_n = None
    valid_any = None
    inrange_any = None
    daily_rows = []

    for chunk_start in range(0, len(selected_days), DOWNLOAD_DAYS_PER_CHUNK):
        chunk_days = selected_days[chunk_start:chunk_start + DOWNLOAD_DAYS_PER_CHUNK]

        images = []
        for day in chunk_days:
            info = DAY_KEY_TO_INFO.get(day)
            if info is None:
                continue
            images.append(
                ee.Image(info["asset_id"])
                .select(band_index)
                .rename(f"d_{day}")
            )

        if not images:
            continue

        # The rectangle is applied before download, so only the small requested
        # subset is transferred to the Streamlit server.
        stack_img = ee.Image.cat(images).clip(region)
        npy = _download_ee_npy(stack_img, region)
        arrays = _npy_to_band_arrays(npy, len(images))

        for day, raw in zip(chunk_days, arrays):
            raw = np.asarray(raw)
            valid = raw < 255

            values = raw.astype(np.float32, copy=False)
            if kind == "anomaly":
                values = values - 100.0

            inrange = valid & (values >= thr_min) & (values <= thr_max)

            if pixel_sum is None:
                shape = raw.shape
                pixel_sum = np.zeros(shape, dtype=np.float64)
                pixel_n = np.zeros(shape, dtype=np.uint16)
                valid_any = np.zeros(shape, dtype=bool)
                inrange_any = np.zeros(shape, dtype=bool)

            # All chunks use the same region/projection/scale. Guard against an
            # unexpected service-side grid change rather than silently misaligning.
            if raw.shape != pixel_sum.shape:
                raise ValueError(
                    "Earth Engine returned different raster dimensions between chunks "
                    f"({raw.shape} vs {pixel_sum.shape})."
                )

            valid_any |= valid
            inrange_any |= inrange

            cnt_total = int(np.count_nonzero(valid))
            cnt_in = int(np.count_nonzero(inrange))

            if cnt_in:
                vals = values[inrange]
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                vmean = float(np.mean(vals))
                pixel_sum[inrange] += values[inrange]
                pixel_n[inrange] += 1
            else:
                vmin = vmax = vmean = 0.0

            daily_rows.append(
                {
                    "date": DAY_KEY_TO_LABEL.get(day, str(day)),
                    "min": vmin,
                    "max": vmax,
                    "mean": vmean,
                    "count_total": cnt_total,
                    "count_inrange": cnt_in,
                }
            )

    if pixel_sum is None:
        return None, None, None, 0, 0, []

    has_mean = pixel_n > 0
    if np.any(has_mean):
        pixel_mean = np.empty(pixel_sum.shape, dtype=np.float64)
        pixel_mean.fill(np.nan)
        pixel_mean[has_mean] = pixel_sum[has_mean] / pixel_n[has_mean]
        vals = pixel_mean[has_mean]
        summary_min = float(np.min(vals))
        summary_max = float(np.max(vals))
        summary_mean = float(np.mean(vals))
    else:
        summary_min = summary_max = summary_mean = None

    period_valid_count = int(np.count_nonzero(valid_any))
    period_inrange_count = int(np.count_nonzero(inrange_any))

    daily_rows.sort(key=lambda row: row["date"])
    return (
        summary_min,
        summary_max,
        summary_mean,
        period_inrange_count,
        period_valid_count,
        daily_rows,
    )


# ---------------------------------------------
# MAP / DRAWING HELPERS
# ---------------------------------------------
def extract_feature_from_map_state(map_state):
    feature = None
    if map_state is not None:
        feature = map_state.get("last_active_drawing")
        if feature is None:
            drawings = map_state.get("all_drawings")
            if drawings:
                feature = drawings[-1]
    return feature


def add_colorbar(
    m,
    title,
    palette,
    vmin,
    vmax,
    unit="",
    position="left",
    bottom="40px",
):
    n = len(palette)
    labels = [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]

    rows = ""
    unit_txt = f" {unit}" if unit else ""
    for val, col in zip(labels, palette):
        rows += (
            f"<i style='background:{col}; width:18px; height:10px; "
            f"float:left; margin-right:4px;'></i> {val:.1f}{unit_txt}<br>"
        )

    side = "left:40px;" if position == "left" else "right:40px;"

    html = f"""
    <div style='
        position: fixed;
        bottom: {bottom};
        {side}
        width: 280px;
        background-color: white;
        color: black;
        padding: 10px;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 13px;
    '>
    <b>{title}</b><br>
    {rows}
    </div>
    """

    m.get_root().html.add_child(folium.Element(html))


def add_layer_colorbar(m, side_name, layer_name, thr_min, thr_max, position, bottom, role):
    if layer_name == "none":
        return
    cfg = get_layer_legend(layer_name, thr_min, thr_max)
    if cfg is None:
        return
    add_colorbar(
        m,
        title=f"{side_name} {role}: {cfg['title']}",
        palette=cfg["palette"],
        vmin=cfg["min"],
        vmax=cfg["max"],
        unit=cfg.get("unit", ""),
        position=position,
        bottom=bottom,
    )


def ee_tile_url(visual_image):
    """Resolve an Earth Engine visualization to a tile URL once."""
    map_id = visual_image.getMapId({})
    return map_id["tile_fetcher"].url_format


def build_map_from_tiles(
    left_tile_url,
    left_label,
    map_center=None,
    map_zoom=None,
    right_tile_url=None,
    right_label=None,
    left_shading_layer="none",
    left_contour_layer="none",
    right_shading_layer="none",
    right_contour_layer="none",
    left_thr_min=None,
    left_thr_max=None,
    right_thr_min=None,
    right_thr_max=None,
):
    """Build a fresh Folium object from already-resolved tile URLs.

    A fresh Folium object is intentional. st_folium mutates Folium objects while
    generating its Leaflet script, so reusing the same folium.Map from
    session_state can change the component hash and remount the map.
    """
    try:
        if map_center is None:
            map_center = CENTER
        if map_zoom is None:
            map_zoom = ZOOM

        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="Esri.WorldImagery")

        left_layer = folium.TileLayer(
            tiles=left_tile_url,
            attr="Google Earth Engine",
            name=left_label,
            overlay=True,
            control=True,
        )
        left_layer.add_to(m)

        if right_tile_url is not None:
            if right_label is None:
                right_label = "SECONDARY layer"

            right_layer = folium.TileLayer(
                tiles=right_tile_url,
                attr="Google Earth Engine",
                name=right_label,
                overlay=True,
                control=True,
            )
            right_layer.add_to(m)
            SideBySideLayers(left_layer, right_layer).add_to(m)

        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "rectangle": {
                    "shapeOptions": {
                        "color": "#ff8800",
                        "fillColor": "#ff8800",
                        "fillOpacity": 0.2,
                    }
                },
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        add_layer_colorbar(
            m, "MAIN", left_shading_layer, left_thr_min, left_thr_max,
            position="left", bottom="40px", role="shading"
        )
        add_layer_colorbar(
            m, "MAIN", left_contour_layer, left_thr_min, left_thr_max,
            position="left", bottom="260px", role="contour"
        )

        if right_tile_url is not None:
            add_layer_colorbar(
                m, "SECONDARY", right_shading_layer, right_thr_min, right_thr_max,
                position="right", bottom="40px", role="shading"
            )
            add_layer_colorbar(
                m, "SECONDARY", right_contour_layer, right_thr_min, right_thr_max,
                position="right", bottom="260px", role="contour"
            )

        folium.LayerControl().add_to(m)
        return m

    except Exception as e:
        st.error(f"Earth Engine map rendering failed: {e}")
        st.stop()


def saved_region_feature_group(feature):
    """Dynamic overlay for the saved rectangle.

    Passed through st_folium(feature_group_to_add=...), so changing the
    rectangle does not remount/reload the base Leaflet map.
    """
    if feature is None:
        return None

    fg = folium.FeatureGroup(name="Selected region", show=True)
    folium.GeoJson(
        feature,
        name="Selected region",
        style_function=lambda x: {
            "color": "#ff8800",
            "weight": 2,
            "fillColor": "#ff8800",
            "fillOpacity": 0.15,
        },
    ).add_to(fg)
    return fg
# ALTAIR PLOT – MIN / MAX / MEAN
# ---------------------------------------------
def plot_timeseries(df, title, kind, thr_max):
    if df.empty:
        return

    df_plot = df.reset_index()
    df_plot["date_str"] = df_plot["date"]

    if kind == "anomaly":
        ymin = float(df_plot[["min", "mean", "max"]].min().min())
        ymax = float(df_plot[["min", "mean", "max"]].max().max())
        pad = 0.1 * max(1.0, abs(ymin) + abs(ymax))
        y_lower = ymin - pad
        y_upper = ymax + pad
        y_title = "Anomaly value"
    else:
        y_lower = 0.0
        y_upper = thr_max + 5
        y_title = "Value"

    chart = (
        alt.Chart(df_plot)
        .transform_fold(
            ["min", "max", "mean"],
            as_=["stat", "value"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date_str:N",
                title="Date",
                sort=df_plot["date_str"].tolist(),
            ),
            y=alt.Y(
                "value:Q",
                title=y_title,
                scale=alt.Scale(domain=[y_lower, y_upper]),
            ),
            color=alt.Color("stat:N", title="Statistic"),
            tooltip=[
                alt.Tooltip("date_str:N", title="Date"),
                alt.Tooltip("stat:N", title="Statistic"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
        .properties(title=title, height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------
# ALTAIR PLOT – PIXEL COUNTS PER DAY
# ---------------------------------------------
def plot_pixelcount_timeseries(df, title):
    required_cols = {"count_total", "count_inrange"}
    if df.empty or not required_cols.issubset(df.columns):
        return

    df_plot = df.reset_index()
    df_plot["date_str"] = df_plot["date"]

    df_plot["out_of_range"] = df_plot["count_total"] - df_plot["count_inrange"]
    df_plot.loc[df_plot["out_of_range"] < 0, "out_of_range"] = 0

    max_count = df_plot["count_total"].max()
    y_upper = max_count * 1.2 if max_count > 0 else 1

    chart_bars = (
        alt.Chart(df_plot)
        .transform_fold(
            ["count_inrange", "out_of_range"],
            as_=["type", "count"],
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "date_str:N",
                title="Date",
                sort=df_plot["date_str"].tolist(),
            ),
            y=alt.Y(
                "count:Q",
                title="Pixel count",
                scale=alt.Scale(domain=[0, y_upper]),
                stack="zero",
            ),
            color=alt.Color(
                "type:N",
                title="Pixel type",
                sort=["count_inrange", "out_of_range"],
                legend=alt.Legend(
                    labelExpr="datum.value == 'count_inrange' ? 'In-range' : 'Out-of-range'"
                ),
            ),
            tooltip=[
                alt.Tooltip("date_str:N", title="Date"),
                alt.Tooltip("count_total:Q", title="Total pixels"),
                alt.Tooltip("count_inrange:Q", title="In-range pixels"),
                alt.Tooltip("out_of_range:Q", title="Out-of-range pixels"),
            ],
        )
    )

    chart_text = (
        alt.Chart(df_plot)
        .mark_text(dy=-6)
        .encode(
            x=alt.X("date_str:N", sort=df_plot["date_str"].tolist()),
            y=alt.Y("count_total:Q", scale=alt.Scale(domain=[0, y_upper])),
            text=alt.Text("count_total:Q", format="d"),
        )
    )

    chart = (chart_bars + chart_text).properties(
        title=title,
        height=400,
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def build_png_report(view_info, stats_info):
    width, height = 2400, 1400
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    y = 40
    left_margin = 60
    line_h = 38

    draw.text((left_margin, y), "CYGNSS Viewer — Snapshot report", fill="black", font=font_title)
    y += line_h * 2

    header_lines = [
        f"MAIN: {view_info['left_label']}",
        f"SECONDARY: {view_info['right_label']}",
        f"Map center: {view_info['map_center'][0]:.4f}, {view_info['map_center'][1]:.4f} | zoom: {view_info['map_zoom']}",
        f"MAIN shading: {view_info['left_shading']}",
        f"MAIN contour: {view_info['left_contour']}",
        f"SECONDARY shading: {view_info['right_shading']}",
        f"SECONDARY contour: {view_info['right_contour']}",
    ]
    for line in header_lines:
        draw.text((left_margin, y), line, fill="black", font=font_body)
        y += line_h

    y += line_h
    draw.text((left_margin, y), "MAIN statistics", fill="black", font=font_title)
    y += line_h * 2

    rows = [
        ("Stats source", "MAIN asset layer only"),
        ("Threshold range", f"{stats_info['thr_min']} → {stats_info['thr_max']}"),
        ("Region drawn", stats_info["region_drawn"]),
        ("Min", stats_info["min"]),
        ("Max", stats_info["max"]),
        ("Mean", stats_info["mean"]),
        ("In-range pixels", stats_info["count_inrange"]),
        ("Total valid pixels", stats_info["count_total"]),
    ]

    col1_x = left_margin
    col2_x = 800
    table_top = y - 14
    row_h = 48
    table_w = 2000
    table_h = row_h * (len(rows) + 1)

    draw.rectangle(
        [col1_x - 20, table_top, col1_x - 20 + table_w, table_top + table_h],
        outline="black",
        width=2,
    )
    draw.line(
        [col2_x - 20, table_top, col2_x - 20, table_top + table_h],
        fill="black",
        width=2,
    )
    for i in range(1, len(rows) + 1):
        yline = table_top + i * row_h
        draw.line([col1_x - 20, yline, col1_x - 20 + table_w, yline], fill="black", width=1)

    draw.text((col1_x, table_top + 10), "Metric", fill="black", font=font_title)
    draw.text((col2_x, table_top + 10), "Value", fill="black", font=font_title)

    for idx, (k, v) in enumerate(rows, start=1):
        yrow = table_top + idx * row_h + 10
        draw.text((col1_x, yrow), str(k), fill="black", font=font_body)
        draw.text((col2_x, yrow), str(v), fill="black", font=font_body)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def render_fullpage_screenshot_button():
    components.html(
        """
        <div style="padding:8px 0;">
          <button id="capture_full_page_png"
            style="background:#0b57d0;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer;font-weight:600;">
            Download FULL page PNG (printscreen)
          </button>
          <span id="capture_status" style="margin-left:10px;font-family:sans-serif;font-size:12px;color:#333;"></span>
        </div>
        <script>
          async function loadHtml2Canvas() {
            if (window.html2canvas) return window.html2canvas;
            return new Promise((resolve, reject) => {
              const s = document.createElement("script");
              s.src = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";
              s.onload = () => resolve(window.html2canvas);
              s.onerror = reject;
              document.head.appendChild(s);
            });
          }

          async function capture() {
            const status = document.getElementById("capture_status");
            try {
              status.textContent = "Preparing screenshot...";
              const h2c = await loadHtml2Canvas();
              const target = window.parent.document.body;
              const desiredScale = Math.max(2, window.parent.devicePixelRatio || 2);
              const maxDim = 8000; // keep file safe for viewers
              const rawW = target.scrollWidth * desiredScale;
              const rawH = target.scrollHeight * desiredScale;
              const limiter = Math.max(rawW / maxDim, rawH / maxDim, 1);
              const scale = desiredScale / limiter;
              const canvas = await h2c(target, {
                useCORS: true,
                allowTaint: true,
                backgroundColor: "#ffffff",
                scale: scale,
                windowWidth: target.scrollWidth,
                windowHeight: target.scrollHeight,
                width: target.scrollWidth,
                height: target.scrollHeight,
                scrollX: 0,
                scrollY: 0
              });
              const a = document.createElement("a");
              a.download = "dashboard_fullpage_screenshot.png";
              canvas.toBlob((blob) => {
                if (!blob) {
                  status.textContent = "Screenshot failed: empty PNG blob.";
                  return;
                }
                const url = URL.createObjectURL(blob);
                a.href = url;
                a.click();
                setTimeout(() => URL.revokeObjectURL(url), 2000);
                status.textContent = "PNG downloaded.";
              }, "image/png");
            } catch (e) {
              status.textContent = "Screenshot failed (browser CORS/security).";
            }
          }

          document.getElementById("capture_full_page_png").addEventListener("click", capture);
        </script>
        """,
        height=80,
    )


# ---------------------------------------------
# ---------------------------------------------
# APP HEADER
# ---------------------------------------------
st.title("CYGNSS – Regional Viewer")
st.caption(
    "Compare MAIN/SECONDARY map layers over independent date ranges. "
    "Each side can use one shading layer and one contour layer. "
    "CYGNSS bands are selectable as ordinary layers."
)

# ---------------------------------------------
# SESSION STATE
# ---------------------------------------------
if "saved_feature" not in st.session_state:
    st.session_state.saved_feature = None

if "map_center" not in st.session_state:
    st.session_state.map_center = CENTER

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = ZOOM

# Cache only Earth Engine tile URLs, never the Folium map itself.
if "map_tile_signature" not in st.session_state:
    st.session_state.map_tile_signature = None
if "left_tile_url" not in st.session_state:
    st.session_state.left_tile_url = None
if "right_tile_url" not in st.session_state:
    st.session_state.right_tile_url = None
if "map_revision" not in st.session_state:
    st.session_state.map_revision = 0

# st_folium copies its newest component value into session_state[key] in its
# internal on_change callback BEFORE the script reruns. Read that value now,
# before constructing/rendering the map, so the rectangle is already known on
# the drawing-triggered rerun.
component_key = f"cygnss_map_{st.session_state.map_revision}"
_previous_map_state = st.session_state.get(component_key)
_previous_feature = extract_feature_from_map_state(_previous_map_state)
if _previous_feature and "geometry" in _previous_feature:
    st.session_state.saved_feature = _previous_feature

if st.button("Clear selected region"):
    st.session_state.saved_feature = None
    st.session_state.map_revision += 1
    # Changing the component key intentionally remounts the map only for Clear,
    # which also clears Leaflet.Draw's internal drawnItems layer.
    component_key = f"cygnss_map_{st.session_state.map_revision}"

split_view = st.checkbox(
    "Enable split-view map comparison (MAIN vs SECONDARY)",
    value=True,
)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### MAIN panel")

    left_shading_layer = st.selectbox(
        "MAIN shading layer:",
        list(LAYER_OPTIONS.keys()),
        index=list(LAYER_OPTIONS.keys()).index("cygnss_1"),
        format_func=lambda k: LAYER_OPTIONS[k],
        key="left_shading_layer",
    )

    left_contour_layer = st.selectbox(
        "MAIN contour layer:",
        list(LAYER_OPTIONS.keys()),
        index=0,
        format_func=lambda k: LAYER_OPTIONS[k],
        key="left_contour_layer",
    )

    left_date_range = st.date_input(
        "MAIN date range (from–to):",
        value=(MIN_DATE, MIN_DATE),
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        format="YYYY-MM-DD",
        key="left_date_range",
    )

with right_col:
    st.markdown("### SECONDARY panel")

    right_shading_layer = st.selectbox(
        "SECONDARY shading layer:",
        list(LAYER_OPTIONS.keys()),
        index=list(LAYER_OPTIONS.keys()).index("cygnss_1"),
        format_func=lambda k: LAYER_OPTIONS[k],
        key="right_shading_layer",
        disabled=not split_view,
    )

    right_contour_layer = st.selectbox(
        "SECONDARY contour layer:",
        list(LAYER_OPTIONS.keys()),
        index=0,
        format_func=lambda k: LAYER_OPTIONS[k],
        key="right_contour_layer",
        disabled=not split_view,
    )

    right_date_range = st.date_input(
        "SECONDARY date range (from–to):",
        value=(MIN_DATE, MIN_DATE),
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        format="YYYY-MM-DD",
        key="right_date_range",
        disabled=not split_view,
    )

left_start_date, left_end_date = parse_date_range(left_date_range)
if left_start_date is None:
    st.warning("Invalid MAIN date range.")
    st.stop()

left_selected_dates, left_sel_days = dates_to_doys(left_start_date, left_end_date)
if not left_sel_days:
    st.warning("No valid MAIN dataset days found in selected range.")
    st.stop()

left_cygnss_layer = selected_cygnss_layer(left_shading_layer, left_contour_layer)
left_kind_for_thr = cygnss_layer_kind(left_cygnss_layer) if left_cygnss_layer else "inundation"

if left_kind_for_thr == "anomaly":
    left_thr_min, left_thr_max = st.slider(
        "MAIN CYGNSS threshold range:",
        min_value=-100,
        max_value=100,
        value=(-100, 100),
        step=1,
        key="left_thr",
    )
else:
    left_thr_min, left_thr_max = st.slider(
        "MAIN CYGNSS threshold range:",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1,
        key="left_thr",
    )

if left_thr_min >= left_thr_max:
    st.error("MAIN lower threshold must be smaller than upper threshold.")
    st.stop()

if is_cygnss_layer(left_shading_layer) and is_cygnss_layer(left_contour_layer):
    if cygnss_layer_kind(left_shading_layer) != cygnss_layer_kind(left_contour_layer):
        st.warning(
            "MAIN uses one CYGNSS threshold slider for both CYGNSS layers. "
            "You selected one normal band and one anomaly band, so the same threshold range may not fit both."
        )

left_label = (
    f"MAIN | shading: {LAYER_OPTIONS[left_shading_layer]} | "
    f"contour: {LAYER_OPTIONS[left_contour_layer]} | "
    f"{left_start_date.strftime('%Y-%m-%d')}→{left_end_date.strftime('%Y-%m-%d')}"
)

if split_view:
    right_start_date, right_end_date = parse_date_range(right_date_range)
    if right_start_date is None:
        st.warning("Invalid SECONDARY date range.")
        st.stop()

    right_selected_dates, right_sel_days = dates_to_doys(right_start_date, right_end_date)
    if not right_sel_days:
        st.warning("No valid SECONDARY dataset days found in selected range.")
        st.stop()

    right_cygnss_layer = selected_cygnss_layer(right_shading_layer, right_contour_layer)
    right_kind_for_thr = cygnss_layer_kind(right_cygnss_layer) if right_cygnss_layer else "inundation"

    if right_kind_for_thr == "anomaly":
        right_thr_min, right_thr_max = st.slider(
            "SECONDARY CYGNSS threshold range:",
            min_value=-100,
            max_value=100,
            value=(-100, 100),
            step=1,
            key="right_thr",
        )
    else:
        right_thr_min, right_thr_max = st.slider(
            "SECONDARY CYGNSS threshold range:",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1,
            key="right_thr",
        )

    if right_thr_min >= right_thr_max:
        st.error("SECONDARY lower threshold must be smaller than upper threshold.")
        st.stop()

    if is_cygnss_layer(right_shading_layer) and is_cygnss_layer(right_contour_layer):
        if cygnss_layer_kind(right_shading_layer) != cygnss_layer_kind(right_contour_layer):
            st.warning(
                "SECONDARY uses one CYGNSS threshold slider for both CYGNSS layers. "
                "You selected one normal band and one anomaly band, so the same threshold range may not fit both."
            )

    right_label = (
        f"SECONDARY | shading: {LAYER_OPTIONS[right_shading_layer]} | "
        f"contour: {LAYER_OPTIONS[right_contour_layer]} | "
        f"{right_start_date.strftime('%Y-%m-%d')}→{right_end_date.strftime('%Y-%m-%d')}"
    )
else:
    right_sel_days = None
    right_start_date = None
    right_end_date = None
    right_thr_min = None
    right_thr_max = None
    right_label = None
    right_cygnss_layer = None

st.write("MAIN dates used:", ", ".join(d.strftime("%Y-%m-%d") for d in left_selected_dates))
if split_view and right_start_date is not None:
    st.write("SECONDARY dates used:", ", ".join(d.strftime("%Y-%m-%d") for d in right_selected_dates))
# ---------------------------------------------
# BUILD / DISPLAY MAP
# ---------------------------------------------
# Resolve Earth Engine tile URLs only when map-defining controls change.
# Drawing a rectangle does NOT change this signature, so no new EE map request
# is made on the drawing-triggered rerun.
map_signature = (
    bool(split_view),
    left_shading_layer,
    left_contour_layer,
    tuple(left_sel_days),
    float(left_thr_min),
    float(left_thr_max),
    right_shading_layer if split_view else "none",
    right_contour_layer if split_view else "none",
    tuple(right_sel_days) if split_view and right_sel_days is not None else (),
    float(right_thr_min) if right_thr_min is not None else None,
    float(right_thr_max) if right_thr_max is not None else None,
)

if (
    st.session_state.left_tile_url is None
    or st.session_state.map_tile_signature != map_signature
):
    try:
        left_visual_image = build_side_visual_image(
            selected_days=left_sel_days,
            thr_min=left_thr_min,
            thr_max=left_thr_max,
            start_date=left_start_date,
            end_date=left_end_date,
            shading_layer=left_shading_layer,
            contour_layer=left_contour_layer,
        )
        st.session_state.left_tile_url = ee_tile_url(left_visual_image)
    except Exception as e:
        st.error(f"Failed to build MAIN image: {e}")
        st.stop()

    st.session_state.right_tile_url = None
    if split_view and right_sel_days is not None:
        try:
            right_visual_image = build_side_visual_image(
                selected_days=right_sel_days,
                thr_min=right_thr_min,
                thr_max=right_thr_max,
                start_date=right_start_date,
                end_date=right_end_date,
                shading_layer=right_shading_layer,
                contour_layer=right_contour_layer,
            )
            st.session_state.right_tile_url = ee_tile_url(right_visual_image)
        except Exception as e:
            st.error(f"Failed to build SECONDARY image: {e}")
            st.stop()

    st.session_state.map_tile_signature = map_signature

# Build a fresh Folium wrapper from stable tile URLs. Because its generated
# base script is unchanged on a drawing-only rerun, st_folium keeps the same
# frontend Leaflet map instead of remounting it.
m = build_map_from_tiles(
    left_tile_url=st.session_state.left_tile_url,
    left_label=left_label,
    map_center=st.session_state.map_center,
    map_zoom=st.session_state.map_zoom,
    right_tile_url=st.session_state.right_tile_url if split_view else None,
    right_label=right_label,
    left_shading_layer=left_shading_layer,
    left_contour_layer=left_contour_layer,
    right_shading_layer=right_shading_layer if split_view else "none",
    right_contour_layer=right_contour_layer if split_view else "none",
    left_thr_min=left_thr_min,
    left_thr_max=left_thr_max,
    right_thr_min=right_thr_min,
    right_thr_max=right_thr_max,
)

# The saved region is sent separately from the base map. streamlit-folium
# updates feature_group_to_add dynamically without reloading the map.
region_fg = saved_region_feature_group(st.session_state.saved_feature)

map_state = st_folium(
    m,
    height=650,
    width=None,
    key=component_key,
    returned_objects=["last_active_drawing", "all_drawings"],
    feature_group_to_add=region_fg,
)

current_feature = extract_feature_from_map_state(map_state)
if current_feature and "geometry" in current_feature:
    st.session_state.saved_feature = current_feature
    feature = current_feature
else:
    feature = st.session_state.saved_feature

st.markdown("---")
# STATS & COUNTS FOR SELECTED REGION (MAIN CYGNSS ONLY, IF SELECTED)
# ---------------------------------------------
st.subheader("Statistics and pixel counts for the drawn area")

user_min = user_max = user_mean = None
left_sel_days_tuple = tuple(left_sel_days)
pixel_count_inrange = None
pixel_count_total = None
region_drawn = "No"

if left_cygnss_layer is None:
    st.info("No CYGNSS layer is selected on MAIN. Statistics are currently calculated only for a selected MAIN CYGNSS layer.")
else:
    stats_band_number = cygnss_band_number(left_cygnss_layer)
    stats_kind = band_kind(stats_band_number)
    stats_band_index = stats_band_number - 1

    st.caption(f"Stats source: MAIN {LAYER_OPTIONS[left_cygnss_layer]}")

    if feature and "geometry" in feature:
        geom = feature["geometry"]
        coords = geom.get("coordinates", [])

        if coords and isinstance(coords[0], list):
            region_drawn = "Yes"
            ring = coords[0]
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            xmin, xmax = min(lons), max(lons)
            ymin, ymax = min(lats), max(lats)

            with st.spinner("Loading selected pixels and calculating statistics..."):
                (
                    user_min,
                    user_max,
                    user_mean,
                    pixel_count_inrange,
                    pixel_count_total,
                    region_ts,
                ) = compute_region_stats_local(
                    left_sel_days_tuple,
                    left_thr_min,
                    left_thr_max,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    stats_kind,
                    stats_band_index,
                )

            if any(v is None for v in (user_min, user_max, user_mean)) or pixel_count_total == 0:
                st.info(
                    "There are no valid pixels in the selected area "
                    "for the chosen MAIN thresholds/scale. Try a larger area or different thresholds."
                )
            else:
                c1, c2, c3, c4, c5 = st.columns(5)

                if stats_kind == "anomaly":
                    c1.metric("Min anomaly (area)", f"{user_min:.4f}")
                    c2.metric("Max anomaly (area)", f"{user_max:.4f}")
                    c3.metric("Mean anomaly (area)", f"{user_mean:.4f}")
                    c4.metric("In-range pixels", f"{pixel_count_inrange}")
                    c5.metric("Total valid pixels", f"{pixel_count_total}")
                else:
                    c1.metric("Min (area)", f"{user_min:.4f}")
                    c2.metric("Max (area)", f"{user_max:.4f}")
                    c3.metric("Mean (area)", f"{user_mean:.4f}")
                    c4.metric("In-range pixels", f"{pixel_count_inrange}")
                    c5.metric("Total valid pixels", f"{pixel_count_total}")

                if region_ts:
                    df_r = pd.DataFrame(region_ts)
                    df_r = df_r.sort_values("date").set_index("date")

                    col_ts, col_cnt = st.columns(2)

                    with col_ts:
                        title_ts = (
                            f"Min / Max / Mean anomaly time series (MAIN area, {cygnss_layer_label(left_cygnss_layer)})"
                            if stats_kind == "anomaly"
                            else f"Min / Max / Mean time series (MAIN area, {cygnss_layer_label(left_cygnss_layer)})"
                        )
                        plot_timeseries(df_r, title_ts, stats_kind, left_thr_max)

                    with col_cnt:
                        title_cnt = f"Daily pixel counts in MAIN area ({cygnss_layer_label(left_cygnss_layer)})"
                        plot_pixelcount_timeseries(df_r, title_cnt)
                else:
                    st.info("No data available to draw MAIN time series for the selected area after masking.")
        else:
            st.info("Draw a rectangular area on the map using the drawing tool.")
    else:
        st.info("Draw a rectangular area on the map using the drawing tool (rectangle icon in the top-left corner).")

st.markdown("---")
st.subheader("Export PNG")
st.caption("Use full-page screenshot first (captures map + charts + stats). If browser blocks it, use fallback report PNG.")

render_fullpage_screenshot_button()

view_info = {
    "left_label": left_label,
    "right_label": right_label if right_label is not None else "Split view disabled",
    "map_center": st.session_state.map_center,
    "map_zoom": st.session_state.map_zoom,
    "left_shading": LAYER_OPTIONS[left_shading_layer],
    "left_contour": LAYER_OPTIONS[left_contour_layer],
    "right_shading": LAYER_OPTIONS[right_shading_layer] if split_view else "None",
    "right_contour": LAYER_OPTIONS[right_contour_layer] if split_view else "None",
}

stats_info = {
    "thr_min": left_thr_min,
    "thr_max": left_thr_max,
    "region_drawn": region_drawn,
    "min": f"{user_min:.4f}" if user_min is not None else "N/A",
    "max": f"{user_max:.4f}" if user_max is not None else "N/A",
    "mean": f"{user_mean:.4f}" if user_mean is not None else "N/A",
    "count_inrange": str(pixel_count_inrange) if pixel_count_inrange is not None else "N/A",
    "count_total": str(pixel_count_total) if pixel_count_total is not None else "N/A",
}

png_bytes = build_png_report(view_info=view_info, stats_info=stats_info)
st.download_button(
    label="Fallback: Download report PNG (metadata + stats)",
    data=png_bytes,
    file_name=f"cygnss_snapshot_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
    mime="image/png",
)
