# app.py
# pip install streamlit folium earthengine-api streamlit-folium pandas altair google-auth

import streamlit as st
import ee
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import datetime as dt
import pandas as pd
import altair as alt
from google.oauth2 import service_account

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

# Julian days
START_DOY = 305
END_DOY = 336
YEAR = 2025

# Build list: [(doy, "YYYY-MM-DD"), ...]
BASE_DATE = dt.date(YEAR, 1, 1) + dt.timedelta(days=START_DOY - 1)
DAYS_INFO = [
    {
        "doy": START_DOY + i,
        "label": (BASE_DATE + dt.timedelta(days=i)).strftime("%Y-%m-%d"),
    }
    for i in range(END_DOY - START_DOY + 1)
]
DOY_TO_LABEL = {info["doy"]: info["label"] for info in DAYS_INFO}

# Available date range in the dataset
MIN_DATE = BASE_DATE
MAX_DATE = BASE_DATE + dt.timedelta(days=END_DOY - START_DOY)

CENTER = [0, 0]
ZOOM = 2

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

# Supported data modes
DATA_MODES = {
    "Daily observations (band 1)": {"kind": "inundation", "band_index": 0},
    "3-daily interpolated data (band 2)": {"kind": "inundation", "band_index": 1},
    "Interpolation flux (band 3)": {"kind": "inundation", "band_index": 2},
    "1-day inundation anomalies (band 4)": {"kind": "anomaly", "band_index": 3},
    "Interpolated inundation anomalies (band 5)": {"kind": "anomaly", "band_index": 4},
}

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


def test_asset_access():
    """
    Test whether the app can read one known asset.
    This helps separate IAM/auth errors from rendering errors.
    """
    test_path = f"{ASSET_FOLDER}/inundation_5bands_{YEAR}_{START_DOY}"
    try:
        img = ee.Image(test_path)
        info = img.getInfo()
        bands = [b.get("id", f"band_{i}") for i, b in enumerate(info.get("bands", []))]
        st.success("Earth Engine initialized and test asset is readable.")
        with st.expander("Debug: Earth Engine connection"):
            st.write("PROJECT_ID:", PROJECT_ID)
            st.write("ASSET_FOLDER:", ASSET_FOLDER)
            st.write("Test asset:", test_path)
            st.write("Bands:", bands)
    except Exception as e:
        st.error(f"Asset access failed for {test_path}: {e}")
        st.stop()


ensure_ee()
test_asset_access()

# ---------------------------------------------
# IMAGE COLLECTIONS
# ---------------------------------------------
def build_inund_collection():
    """
    Build image collection from all configured asset days.
    """
    imgs = []
    for info in DAYS_INFO:
        day = info["doy"]
        path = f"{ASSET_FOLDER}/inundation_5bands_{YEAR}_{day}"
        img = ee.Image(path).set("day", day)
        imgs.append(img)
    return ee.ImageCollection(imgs)


IC_INUND = build_inund_collection()


def get_collection(kind: str):
    """
    Return the correct collection for the selected data type.
    """
    return IC_INUND


# ---------------------------------------------
# HELPER FUNCTIONS – INUNDATION
# ---------------------------------------------
def mask_inund_band(img, band_index, thr_min, thr_max):
    band = img.select(band_index)
    mask = band.gte(thr_min).And(band.lte(thr_max)).And(band.lt(255))
    return band.updateMask(mask)


def inund_valid_band(img, band_index):
    band = img.select(band_index)
    return band.updateMask(band.lt(255))


# ---------------------------------------------
# HELPER FUNCTIONS – ANOMALIES
# ---------------------------------------------
def anomaly_valid_band(img, band_index):
    band = img.select(band_index)
    return band.updateMask(band.lt(255))


def anomaly_thresholded(img, band_index, thr_min, thr_max):
    band = anomaly_valid_band(img, band_index)
    thr_mask = band.gte(thr_min).And(band.lte(thr_max))
    return band.updateMask(thr_mask)


# ---------------------------------------------
# BUILD MEAN IMAGE FOR MAP DISPLAY
# ---------------------------------------------
def build_mean_image(selected_days, thr_min, thr_max, kind, band_index):
    """
    Compute pixel-wise mean image over selected days after masking.
    """
    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    size = ic_sel.size().getInfo()
    if size == 0:
        raise ValueError(f"No images found for selected days: {selected_days}")

    if kind == "inundation":
        ic_proc = ic_sel.map(lambda img: mask_inund_band(img, band_index, thr_min, thr_max))
    else:
        ic_proc = ic_sel.map(lambda img: anomaly_thresholded(img, band_index, thr_min, thr_max))

    stacked = ic_proc.toBands()
    pixel_mean = stacked.reduce(ee.Reducer.mean())
    return pixel_mean


# ---------------------------------------------
# TIME SERIES FOR AREA (CACHE)
# ---------------------------------------------
@st.cache_data
def compute_region_ts_for_bbox(
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
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    results = []

    ic = get_collection(kind)

    def region_stat(img, reducer):
        d = img.reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=3000,
            maxPixels=1e13,
        ).getInfo()
        if not d:
            return None
        val = list(d.values())[0]
        if val is None:
            return None
        return float(val)

    def region_count_inrange(img_thr):
        d = img_thr.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=3000,
            maxPixels=1e13,
        ).getInfo()
        if not d:
            return 0
        val = list(d.values())[0]
        if val is None:
            return 0
        return int(val)

    def region_count_total_inund(img):
        band_valid = inund_valid_band(img, band_index)
        d = band_valid.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=3000,
            maxPixels=1e13,
        ).getInfo()
        if not d:
            return 0
        val = list(d.values())[0]
        if val is None:
            return 0
        return int(val)

    def region_count_total_anom(img):
        band_valid = anomaly_valid_band(img, band_index)
        d = band_valid.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=3000,
            maxPixels=1e13,
        ).getInfo()
        if not d:
            return 0
        val = list(d.values())[0]
        if val is None:
            return 0
        return int(val)

    for day in sorted(selected_days):
        img = ic.filter(ee.Filter.eq("day", day)).first()

        if kind == "inundation":
            img_thr = mask_inund_band(img, band_index, thr_min, thr_max)
            cnt_tot = region_count_total_inund(img)
        else:
            img_thr = anomaly_thresholded(img, band_index, thr_min, thr_max)
            cnt_tot = region_count_total_anom(img)

        vmin = region_stat(img_thr, ee.Reducer.min())
        vmax = region_stat(img_thr, ee.Reducer.max())
        vmean = region_stat(img_thr, ee.Reducer.mean())
        cnt_in = region_count_inrange(img_thr)

        vmin = vmin if vmin is not None else 0.0
        vmax = vmax if vmax is not None else 0.0
        vmean = vmean if vmean is not None else 0.0

        results.append(
            {
                "date": DOY_TO_LABEL.get(day, str(day)),
                "min": vmin,
                "max": vmax,
                "mean": vmean,
                "count_total": cnt_tot,
                "count_inrange": cnt_in,
            }
        )

    return results


# ---------------------------------------------
# SUMMARY STATS FOR MEAN IMAGE OVER AREA (CACHE)
# ---------------------------------------------
@st.cache_data
def compute_region_summary_for_bbox(
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
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        ic_proc = ic_sel.map(lambda img: mask_inund_band(img, band_index, thr_min, thr_max))
    else:
        ic_proc = ic_sel.map(lambda img: anomaly_thresholded(img, band_index, thr_min, thr_max))

    stacked = ic_proc.toBands()
    pixel_mean = stacked.reduce(ee.Reducer.mean())

    def region_stat(img, reducer):
        d = img.reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=3000,
            maxPixels=1e13,
        ).getInfo()
        if not d:
            return None
        val = list(d.values())[0]
        if val is None:
            return None
        return float(val)

    rmin = region_stat(pixel_mean, ee.Reducer.min())
    rmax = region_stat(pixel_mean, ee.Reducer.max())
    rmean = region_stat(pixel_mean, ee.Reducer.mean())
    return rmin, rmax, rmean


# ---------------------------------------------
# PIXEL COUNTS FOR SELECTED PERIOD (CACHE)
# ---------------------------------------------
@st.cache_data
def compute_region_pixel_count(
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
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        def valid_mask(img):
            band = img.select(band_index)
            return band.lt(255).toInt()

        def inrange_mask(img):
            band = img.select(band_index)
            return band.gte(thr_min).And(band.lte(thr_max)).And(band.lt(255)).toInt()
    else:
        def valid_mask(img):
            band = img.select(band_index)
            return band.lt(255).toInt()

        def inrange_mask(img):
            band = img.select(band_index)
            return band.lt(255).And(band.gte(thr_min)).And(band.lte(thr_max)).toInt()

    valid_any = ic_sel.map(valid_mask).max()
    inrange_any = ic_sel.map(inrange_mask).max()

    d_tot = valid_any.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=3000,
        maxPixels=1e13,
    ).getInfo()

    d_in = inrange_any.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=3000,
        maxPixels=1e13,
    ).getInfo()

    total_count = int(list(d_tot.values())[0]) if d_tot and list(d_tot.values())[0] is not None else 0
    in_range_count = int(list(d_in.values())[0]) if d_in and list(d_in.values())[0] is not None else 0

    return in_range_count, total_count


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


def build_map(
    image,
    thr_min,
    thr_max,
    kind,
    mode_label,
    saved_feature=None,
    map_center=None,
    map_zoom=None,
):
    try:
        band = image.select(0)

        palette = PALETTE_ANOM if kind == "anomaly" else PALETTE_INUND
        vis = {
            "min": thr_min,
            "max": thr_max,
            "palette": palette,
        }

        if map_center is None:
            map_center = CENTER
        if map_zoom is None:
            map_zoom = ZOOM

        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="Esri.WorldImagery")

        map_id = band.getMapId(vis)
        tile_url = map_id["tile_fetcher"].url_format

        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name=f"Mean {mode_label} of selected days",
            overlay=True,
            control=True,
        ).add_to(m)

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

        if saved_feature is not None:
            folium.GeoJson(
                saved_feature,
                name="Selected region",
                style_function=lambda x: {
                    "color": "#ff8800",
                    "weight": 2,
                    "fillColor": "#ff8800",
                    "fillOpacity": 0.15,
                },
            ).add_to(m)

        if kind == "anomaly":
            num_classes = len(PALETTE_ANOM)
            step = (thr_max - thr_min) / (num_classes - 1) if num_classes > 1 else 1
            ticks = [thr_min + i * step for i in range(num_classes)]
            colors = PALETTE_ANOM
            width = 260
        else:
            num_classes = len(PALETTE_INUND)
            step = (thr_max - thr_min) / (num_classes - 1) if num_classes > 1 else 1
            ticks = [thr_min + i * step for i in range(num_classes)]
            colors = PALETTE_INUND
            width = 220

        legend_rows = ""
        for val, col in zip(ticks, colors):
            legend_rows += (
                f"<i style='background:{col}; width:18px; height:10px; "
                f"float:left; margin-right:4px;'></i> {val:.1f}<br>"
            )

        legend_html = f"""
         <div style='position: fixed; bottom: 40px; left: 40px; width: {width}px;
             background-color: white; color: black; padding: 10px; border:2px solid grey; z-index:9999;'>
         <b>{mode_label} ({thr_min}–{thr_max})</b><br>
         {legend_rows}
         </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)
        return m

    except Exception as e:
        st.error(f"Earth Engine map rendering failed: {e}")
        st.stop()


# ---------------------------------------------
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


# ---------------------------------------------
# APP HEADER
# ---------------------------------------------
st.title("CYGNSS – Regional Viewer")
st.caption(
    "Explore CYGNSS inundation products (bands 1-5) from Google Earth Engine. "
    "The map shows the mean of selected days for the chosen data type after applying thresholds. "
    "Draw an area (rectangle) on the map to compute regional statistics and pixel counts."
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

if st.button("Clear selected region"):
    st.session_state.saved_feature = None


# ---------------------------------------------
# 0) DATA TYPE SELECTION
# ---------------------------------------------
mode_label = st.selectbox(
    "Data type:",
    list(DATA_MODES.keys()),
    index=0,
)
mode_cfg = DATA_MODES[mode_label]
kind = mode_cfg["kind"]
band_index = mode_cfg["band_index"]

# ---------------------------------------------
# 1) DATE RANGE SELECTION
# ---------------------------------------------
st.markdown("### Select date range")

date_range = st.date_input(
    "Date range (from–to):",
    value=(MIN_DATE, MIN_DATE),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    format="YYYY-MM-DD",
)

if isinstance(date_range, tuple):
    if len(date_range) != 2 or date_range[0] is None or date_range[1] is None:
        st.stop()
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

if start_date > end_date:
    start_date, end_date = end_date, start_date

start_date = max(start_date, MIN_DATE)
end_date = min(end_date, MAX_DATE)

selected_dates = []
current_date = start_date
while current_date <= end_date:
    selected_dates.append(current_date)
    current_date += dt.timedelta(days=1)

if not selected_dates:
    st.stop()

year_start = dt.date(YEAR, 1, 1)
sel_days = [
    (d - year_start).days + 1
    for d in selected_dates
    if START_DOY <= (d - year_start).days + 1 <= END_DOY
]

if not sel_days:
    st.warning("No valid dataset days found in the selected range.")
    st.stop()

sel_days_tuple = tuple(sorted(sel_days))

st.write("Dates used:", ", ".join(d.strftime("%Y-%m-%d") for d in selected_dates))

# ---------------------------------------------
# 2) THRESHOLD SELECTION
# ---------------------------------------------
if kind == "anomaly":
    thr_min, thr_max = st.slider(
        "Anomaly range (lower and upper threshold):",
        min_value=-100,
        max_value=100,
        value=(-20, 20),
        step=1,
    )
else:
    thr_min, thr_max = st.slider(
        f"Value range (lower and upper threshold, {mode_label}):",
        min_value=0,
        max_value=100,
        value=(20, 100),
        step=1,
    )

if thr_min >= thr_max:
    st.error("Lower threshold must be smaller than upper threshold.")
    st.stop()

# ---------------------------------------------
# BUILD MEAN IMAGE FOR MAP
# ---------------------------------------------
try:
    mean_image = build_mean_image(sel_days, thr_min, thr_max, kind, band_index)
except Exception as e:
    st.error(f"Failed to build mean image: {e}")
    st.stop()

# ---------------------------------------------
# BUILD / DISPLAY MAP
# ---------------------------------------------
m = build_map(
    mean_image,
    thr_min,
    thr_max,
    kind,
    mode_label,
    saved_feature=st.session_state.saved_feature,
    map_center=st.session_state.map_center,
    map_zoom=st.session_state.map_zoom,
)

map_state = st_folium(
    m,
    height=650,
    width=None,
    key="cygnss_map",
)

if map_state is not None:
    if map_state.get("center") is not None:
        center_dict = map_state["center"]
        st.session_state.map_center = [center_dict["lat"], center_dict["lng"]]

    if map_state.get("zoom") is not None:
        st.session_state.map_zoom = map_state["zoom"]

current_feature = extract_feature_from_map_state(map_state)

if current_feature and "geometry" in current_feature:
    st.session_state.saved_feature = current_feature

feature = st.session_state.saved_feature

st.markdown("---")

# ---------------------------------------------
# STATS & COUNTS FOR SELECTED REGION
# ---------------------------------------------
st.subheader("Statistics and pixel counts for the drawn area")

user_min = user_max = user_mean = None

if feature and "geometry" in feature:
    geom = feature["geometry"]
    coords = geom.get("coordinates", [])

    if coords and isinstance(coords[0], list):
        ring = coords[0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        xmin, xmax = min(lons), max(lons)
        ymin, ymax = min(lats), max(lats)

        user_min, user_max, user_mean = compute_region_summary_for_bbox(
            sel_days_tuple,
            thr_min,
            thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            kind,
            band_index,
        )

        pixel_count_inrange, pixel_count_total = compute_region_pixel_count(
            sel_days_tuple,
            thr_min,
            thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            kind,
            band_index,
        )

        region_ts = compute_region_ts_for_bbox(
            sel_days_tuple,
            thr_min,
            thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            kind,
            band_index,
        )

        if any(v is None for v in (user_min, user_max, user_mean)) or pixel_count_total == 0:
            st.info(
                "There are no valid pixels in the selected area "
                "for the chosen thresholds/scale. Try a larger area or different thresholds."
            )
        else:
            c1, c2, c3, c4, c5 = st.columns(5)

            if kind == "anomaly":
                c1.metric("Min anomaly (area)", f"{user_min:.4f}")
                c2.metric("Max anomaly (area)", f"{user_max:.4f}")
                c3.metric("Mean anomaly (area)", f"{user_mean:.4f}")
                c4.metric("In-range pixels (at least once in selected period)", f"{pixel_count_inrange}")
                c5.metric("Total valid pixels (at least once in selected period)", f"{pixel_count_total}")
            else:
                c1.metric("Min (area)", f"{user_min:.4f}")
                c2.metric("Max (area)", f"{user_max:.4f}")
                c3.metric("Mean (area)", f"{user_mean:.4f}")
                c4.metric("In-range pixels (at least once in selected period)", f"{pixel_count_inrange}")
                c5.metric("Total valid pixels (at least once in selected period)", f"{pixel_count_total}")

            if region_ts:
                df_r = pd.DataFrame(region_ts)
                df_r = df_r.sort_values("date").set_index("date")

                col_ts, col_cnt = st.columns(2)

                with col_ts:
                    title_ts = (
                        f"Min / Max / Mean anomaly time series (area, {mode_label})"
                        if kind == "anomaly"
                        else f"Min / Max / Mean time series (area, {mode_label})"
                    )
                    plot_timeseries(df_r, title_ts, kind, thr_max)

                with col_cnt:
                    title_cnt = f"Daily pixel counts in area ({mode_label})"
                    plot_pixelcount_timeseries(df_r, title_cnt)
            else:
                st.info("No data available to draw time series for the selected area (after masking).")
    else:
        st.info("Draw a rectangular area on the map using the drawing tool.")
else:
    st.info("Draw a rectangular area on the map using the drawing tool (rectangle icon in the top-left corner).")
