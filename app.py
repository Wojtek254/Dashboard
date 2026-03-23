# cygnss_regional_dashboard_combined.py
# pip install streamlit folium earthengine-api streamlit-folium pandas altair

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
# CONFIG
# ---------------------------------------------
PROJECT_ID = "dahsboard-streamlit-1"
ASSET_FOLDER = "projects/dahsboard-streamlit-1/assets"

# Julian days (example window used in the current app)
START_DOY = 182
END_DOY   = 212
YEAR      = 2021

# Build list: [(doy, "2021-07-01"), ...]
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

CENTER = [0, 0]   # default map center
ZOOM   = 2

# Color palette for inundation
PALETTE_INUND = [
    "#e3f2fd",  # 0 – very light blue
    "#bbdefb",
    "#90caf9",
    "#64b5f6",
    "#42a5f5",
    "#1e88e5",
    "#0d47a1",  # maximum – very dark blue
]

# Color palette for anomalies
# Negative -> orange, zero -> light blue, positive -> darker blue
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
    "Inundation – band 3": {"kind": "inundation", "band_index": 2},
    "Inundation – band 1": {"kind": "inundation", "band_index": 0},
    "Anomaly – band 3":    {"kind": "anomaly",    "band_index": 0},  # single-band anomaly asset
}

# ---------------------------------------------
# INITIALIZE GOOGLE EARTH ENGINE
# ---------------------------------------------
def ensure_ee():
    """
    Initialize the Earth Engine session using credentials
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

ensure_ee()

# ---------------------------------------------
# IMAGE COLLECTIONS
# ---------------------------------------------
def build_inund_collection():
    """
    Build an image collection for inundation assets.
    Each image is tagged with its day-of-year metadata.
    """
    imgs = []
    for info in DAYS_INFO:
        day = info["doy"]
        img = ee.Image(f"{ASSET_FOLDER}/inundation_CYGNSS_3bands_2021_{day}").set("day", day)
        imgs.append(img)
    return ee.ImageCollection(imgs)

def build_anom_collection():
    """
    Build an image collection for anomaly assets.
    Assumed asset pattern:
        {ASSET_FOLDER}/2021_{DOY}_anomaly
    """
    imgs = []
    for info in DAYS_INFO:
        day = info["doy"]
        asset_id = f"{ASSET_FOLDER}/2021_{day}_anomaly"
        img = ee.Image(asset_id).set("day", day)
        imgs.append(img)
    return ee.ImageCollection(imgs)

IC_INUND = build_inund_collection()
IC_ANOM  = build_anom_collection()

def get_collection(kind: str):
    """
    Return the correct image collection for the selected data type.
    """
    return IC_INUND if kind == "inundation" else IC_ANOM

# ---------------------------------------------
# HELPER FUNCTIONS – INUNDATION
# ---------------------------------------------
def mask_inund_band(img, band_index, thr_min, thr_max):
    """
    Select one inundation band and mask:
    - values outside [thr_min, thr_max]
    - pixels equal to 255 (no data)
    """
    band = img.select(band_index)
    mask = band.gte(thr_min).And(band.lte(thr_max)).And(band.lt(255))
    return band.updateMask(mask)

def inund_valid_band(img, band_index):
    """
    Return the selected inundation band with only no-data (255) masked out.
    """
    band = img.select(band_index)
    return band.updateMask(band.lt(255))

# ---------------------------------------------
# HELPER FUNCTIONS – ANOMALIES
# ---------------------------------------------
def anomaly_band3_corrected(img):
    """
    Use anomaly band 3 (single band: index 0),
    treat 255 as no data, and subtract 100
    because anomaly values were stored with an offset.
    """
    raw = img.select(0)
    valid_mask = raw.neq(255)
    corrected = raw.subtract(100).updateMask(valid_mask)
    return corrected

def anomaly_band3_thresholded(img, thr_min, thr_max):
    """
    Correct anomalies first, then apply the [thr_min, thr_max] threshold.
    """
    corr = anomaly_band3_corrected(img)
    thr_mask = corr.gte(thr_min).And(corr.lte(thr_max))
    return corr.updateMask(thr_mask)

# ---------------------------------------------
# BUILD MEAN IMAGE FOR MAP DISPLAY
# ---------------------------------------------
def build_mean_image(selected_days, thr_min, thr_max, kind, band_index):
    """
    Compute a pixel-wise mean image over the selected days
    after applying thresholding and no-data masking.
    This image is only used for map visualization.
    """
    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        ic_proc = ic_sel.map(lambda img: mask_inund_band(img, band_index, thr_min, thr_max))
    else:
        ic_proc = ic_sel.map(lambda img: anomaly_band3_thresholded(img, thr_min, thr_max))

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
    """
    Compute time series of min, max, mean, and pixel counts
    for a rectangular region.

    Important:
    - min/max/mean are computed AFTER thresholding
    - count_total is the number of valid pixels for each day
    - count_inrange is the number of pixels within the threshold range for each day

    This is a day-by-day diagnostic.
    """
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
        corr = anomaly_band3_corrected(img)
        d = corr.reduceRegion(
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
        if img is None:
            continue

        if kind == "inundation":
            img_thr = mask_inund_band(img, band_index, thr_min, thr_max)
            cnt_tot = region_count_total_inund(img)
        else:
            img_thr = anomaly_band3_thresholded(img, thr_min, thr_max)
            cnt_tot = region_count_total_anom(img)

        vmin   = region_stat(img_thr, ee.Reducer.min())
        vmax   = region_stat(img_thr, ee.Reducer.max())
        vmean  = region_stat(img_thr, ee.Reducer.mean())
        cnt_in = region_count_inrange(img_thr)

        vmin  = vmin  if vmin  is not None else 0.0
        vmax  = vmax  if vmax  is not None else 0.0
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
    """
    Compute min, max, and mean for the map mean-image
    inside the selected rectangular region.
    """
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        ic_proc = ic_sel.map(lambda img: mask_inund_band(img, band_index, thr_min, thr_max))
    else:
        ic_proc = ic_sel.map(lambda img: anomaly_band3_thresholded(img, thr_min, thr_max))

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

    rmin  = region_stat(pixel_mean, ee.Reducer.min())
    rmax  = region_stat(pixel_mean, ee.Reducer.max())
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
    """
    Return two period-based diagnostics for the selected region:

    1) in_range_count:
       Number of unique pixels that were within the selected threshold
       at least once during the selected period.

    2) total_count:
       Number of unique pixels that had valid data at least once
       during the selected period.

    This is NOT computed from the mean image.
    This is NOT computed from only the first day.
    Instead, it uses a temporal OR / union logic across all selected days.
    """
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        def valid_mask(img):
            """
            Binary mask:
            1 where a pixel is valid (value != 255), 0 otherwise.
            """
            band = img.select(band_index)
            return band.lt(255).toInt()

        def inrange_mask(img):
            """
            Binary mask:
            1 where a pixel is valid and inside the selected threshold range,
            0 otherwise.
            """
            band = img.select(band_index)
            return band.gte(thr_min).And(band.lte(thr_max)).And(band.lt(255)).toInt()

    else:
        def valid_mask(img):
            """
            Binary mask:
            1 where anomaly data are valid (value != 255), 0 otherwise.
            """
            raw = img.select(0)
            return raw.neq(255).toInt()

        def inrange_mask(img):
            """
            Binary mask:
            1 where anomaly data are valid and corrected anomaly values
            fall within the selected threshold range, 0 otherwise.
            """
            raw = img.select(0)
            corr = raw.subtract(100)
            return raw.neq(255).And(corr.gte(thr_min)).And(corr.lte(thr_max)).toInt()

    # Temporal OR / union:
    # max() over a stack of 0/1 images returns 1 if the condition was met at least once.
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
    """
    Extract the latest user-drawn geometry from streamlit-folium state.
    If no actively edited geometry is present, fall back to the last drawing.
    """
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
    """
    Build the Folium map used in the dashboard.

    The map shows:
    - the mean image over the selected days
    - the drawing tool for region selection
    - the previously saved region, if available
    """
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

    # Draw plugin: only rectangles are enabled.
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

    # Re-display the saved region so that changing thresholds/dates
    # does not visually remove the selected rectangle.
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

    # Legend
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

# ---------------------------------------------
# ALTAIR PLOT – MIN / MAX / MEAN
# ---------------------------------------------
def plot_timeseries(df, title, kind, thr_max):
    """
    Plot min, max, and mean values over time.
    These statistics are computed after thresholding.
    """
    if df.empty:
        return

    df_plot = df.reset_index()
    df_plot["date_str"] = df_plot["date"]

    if kind == "anomaly":
        ymin = float(df_plot[["min", "mean", "max"]].min().min())
        ymax = float(df_plot[["min", "mean", "max"]].max().max())
        pad  = 0.1 * max(1.0, abs(ymin) + abs(ymax))
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
    """
    Plot a stacked bar chart:
    - in-range pixels
    - out-of-range pixels
    for each individual day.

    This is a daily diagnostic and is different from the top summary metrics,
    which now use a temporal union over the selected period.
    """
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
            x=alt.X(
                "date_str:N",
                sort=df_plot["date_str"].tolist(),
            ),
            y=alt.Y(
                "count_total:Q",
                scale=alt.Scale(domain=[0, y_upper]),
            ),
            text=alt.Text("count_total:Q", format="d"),
        )
    )

    chart = (chart_bars + chart_text).properties(
        title=title,
        height=400,
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------
# STREAMLIT PAGE SETUP
# ---------------------------------------------
st.set_page_config(
    page_title="CYGNSS – Regional Viewer (Inundation & Anomalies)",
    layout="wide",
)

st.title("CYGNSS – Regional Viewer")
st.caption(
    "Explore CYGNSS inundation (bands 1 & 3) and anomaly band 3 from Google Earth Engine. "
    "The map shows the mean of selected days for the chosen data type after applying thresholds. "
    "Draw an area (rectangle) on the map to compute regional statistics and pixel counts."
)

# ---------------------------------------------
# SESSION STATE FOR SAVED REGION
# ---------------------------------------------
if "saved_feature" not in st.session_state:
    st.session_state.saved_feature = None

if "map_center" not in st.session_state:
    st.session_state.map_center = CENTER

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = ZOOM

# Optional button to clear the saved region
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
    value=(MIN_DATE, MIN_DATE),   # default: single day
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    format="YYYY-MM-DD",
)

# Streamlit can return either a single date or a tuple
if isinstance(date_range, tuple):
    if len(date_range) != 2 or date_range[0] is None or date_range[1] is None:
        st.stop()
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# Sort the dates if selected in reverse order
if start_date > end_date:
    start_date, end_date = end_date, start_date

# Clip to valid data availability
start_date = max(start_date, MIN_DATE)
end_date   = min(end_date, MAX_DATE)

# Build the list of selected dates
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
    st.stop()

sel_days_tuple = tuple(sorted(sel_days))

st.write(
    "Dates used:",
    ", ".join(d.strftime("%Y-%m-%d") for d in selected_dates),
)

# ---------------------------------------------
# 2) THRESHOLD SELECTION
# ---------------------------------------------
if kind == "anomaly":
    thr_min, thr_max = st.slider(
        "Anomaly range (lower and upper threshold, band 3 after -100 offset):",
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
mean_image = build_mean_image(sel_days, thr_min, thr_max, kind, band_index)

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
    key="cygnss_map",   # constant key -> region is not lost across reruns
)

if map_state is not None:
    if map_state.get("center") is not None:
        center_dict = map_state["center"]
        st.session_state.map_center = [center_dict["lat"], center_dict["lng"]]

    if map_state.get("zoom") is not None:
        st.session_state.map_zoom = map_state["zoom"]

# Extract the current drawing, if any.
current_feature = extract_feature_from_map_state(map_state)

# Save the most recent valid geometry to session state.
if current_feature and "geometry" in current_feature:
    st.session_state.saved_feature = current_feature

# Always use the saved region if available.
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

    # Rectangle coordinates come as a polygon ring.
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

        # Display a message if there are no valid pixels at all in the selected period.
        if (
            any(v is None for v in (user_min, user_max, user_mean))
            or pixel_count_total == 0
        ):
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
                        "Min / Max / Mean anomaly time series (area, band 3)"
                        if kind == "anomaly"
                        else f"Min / Max / Mean time series (area, {mode_label})"
                    )
                    plot_timeseries(df_r, title_ts, kind, thr_max)

                with col_cnt:
                    title_cnt = (
                        "Daily pixel counts in area (anomaly band 3)"
                        if kind == "anomaly"
                        else f"Daily pixel counts in area ({mode_label})"
                    )
                    plot_pixelcount_timeseries(df_r, title_cnt)
            else:
                st.info("No data available to draw time series for the selected area (after masking).")
    else:
        st.info("Draw a rectangular area on the map using the drawing tool.")
else:
    st.info("Draw a rectangular area on the map using the drawing tool (rectangle icon in the top-left corner).")
