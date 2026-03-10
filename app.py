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

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
PROJECT_ID = "cygnss-dashboard"
ASSET_FOLDER = f"projects/{PROJECT_ID}/assets/CYGNSS"

# Julian days (2021_182 ... 2021_212)
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

CENTER = [0, 0]   # default map center (global)
ZOOM   = 2

# Palettes
PALETTE_INUND = [
    "#e3f2fd",  # 0 – very light blue
    "#bbdefb",
    "#90caf9",
    "#64b5f6",
    "#42a5f5",
    "#1e88e5",
    "#0d47a1",  # max – very dark blue
]


# ANOMALIES (negative → orange, 0 → light blue, positive → darker blue)
PALETTE_ANOM = [
    "#8c2d04",
    "#fe9929",
    "#fee6ce",
    "#e3f2fd",  # zero → light blue
    "#90caf9",
    "#42a5f5",
    "#0d47a1",
]

# Data modes
DATA_MODES = {
    "Inundation – band 3": {"kind": "inundation", "band_index": 2},
    "Inundation – band 1": {"kind": "inundation", "band_index": 0},
    "Anomaly – band 3":    {"kind": "anomaly",    "band_index": 0},  # single band
}

# ---------------------------------------------
# INITIALIZE GEE
# ---------------------------------------------
def ensure_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate(auth_mode="localhost", project=PROJECT_ID)
        ee.Initialize(project=PROJECT_ID)

ensure_ee()

# ---------------------------------------------
# IMAGE COLLECTIONS
# ---------------------------------------------
def build_inund_collection():
    imgs = []
    for info in DAYS_INFO:
        day = info["doy"]
        img = ee.Image(f"{ASSET_FOLDER}/inundation_CYGNSS_3bands_2021_{day}").set(
            "day", day
        )
        imgs.append(img)
    return ee.ImageCollection(imgs)

def build_anom_collection():
    """
    Assumed anomaly asset pattern:
      {ASSET_FOLDER}/2021_{DOY}_anomaly
    e.g. projects/cygnss-dashboard/assets/CYGNSS/2021_182_anomaly
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
    return IC_INUND if kind == "inundation" else IC_ANOM

# ---------------------------------------------
# HELPERS – INUNDATION
# ---------------------------------------------
def mask_inund_band(img, band_index, thr_min, thr_max):
    """
    Select inundation band (band_index) and mask:
    - values outside [thr_min, thr_max]
    - pixels equal to 255 (no data)
    """
    band = img.select(band_index)
    mask = (
        band.gte(thr_min)
        .And(band.lte(thr_max))
        .And(band.lt(255))
    )
    return band.updateMask(mask)

def inund_valid_band(img, band_index):
    """
    Return band masked only by 255 (no threshold).
    """
    band = img.select(band_index)
    return band.updateMask(band.lt(255))

# ---------------------------------------------
# HELPERS – ANOMALIES
# ---------------------------------------------
def anomaly_band3_corrected(img):
    """
    Take anomaly band 3 (assumed single band: index 0),
    treat 255 as 'no data', subtract 100 from all other values.
    """
    raw = img.select(0)
    valid_mask = raw.neq(255)
    corrected = raw.subtract(100).updateMask(valid_mask)
    return corrected

def anomaly_band3_thresholded(img, thr_min, thr_max):
    """
    Correct anomalies (255->mask, -100 offset),
    then apply threshold [thr_min, thr_max].
    """
    corr = anomaly_band3_corrected(img)
    thr_mask = corr.gte(thr_min).And(corr.lte(thr_max))
    return corr.updateMask(thr_mask)

# ---------------------------------------------
# MEAN IMAGE (FOR MAP)
# ---------------------------------------------
def build_mean_image(selected_days, thr_min, thr_max, kind, band_index):
    """
    Compute pixel-wise mean of selected days for chosen data type:
      - inundation (band 1 or 3)
      - anomaly (band 3, corrected and thresholded)
    """
    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        ic_proc = ic_sel.map(
            lambda img: mask_inund_band(img, band_index, thr_min, thr_max)
        )
    else:  # anomaly
        ic_proc = ic_sel.map(
            lambda img: anomaly_band3_thresholded(img, thr_min, thr_max)
        )

    stacked = ic_proc.toBands()
    pixel_mean = stacked.reduce(ee.Reducer.mean())
    return pixel_mean

# ---------------------------------------------
# TIME SERIES FOR AREA (min/max/mean + counts) (CACHE)
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
    Time series of min/max/mean and pixel counts for a given area (bbox)
    for a chosen data type (inundation/anomaly).

    Returns list of dicts:
      {
        date,
        min, max, mean,
        count_total,    # all valid pixels (band != 255; after correction for anomaly)
        count_inrange   # pixels within [thr_min, thr_max]
      }
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
# SUMMARY STATS OF MEAN IMAGE FOR AREA (CACHE)
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
    Compute min/max/mean for the mean image (over selected days)
    within the given area (bbox), for a chosen data type.
    """
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    if kind == "inundation":
        ic_proc = ic_sel.map(
            lambda img: mask_inund_band(img, band_index, thr_min, thr_max)
        )
    else:
        ic_proc = ic_sel.map(
            lambda img: anomaly_band3_thresholded(img, thr_min, thr_max)
        )

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
# PIXEL COUNT IN AREA FOR MEAN IMAGE (IN-RANGE) (CACHE)
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
    Zwraca dwie wartości:
      - in_range_count: liczba pikseli w zakresie progów (mean image)
      - total_count:    liczba wszystkich ważnych pikseli (≠255) w prostokącie

    in_range_count liczymy na średnim obrazie po thresholdzie,
    total_count liczymy na pierwszym obrazie z kolekcji (piksele !=255).
    """
    selected_days = list(selected_days_tuple)
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    ic = get_collection(kind)
    ic_sel = ic.filter(ee.Filter.inList("day", selected_days))

    # --- obraz do liczenia pikseli w zakresie (mean po thresholdzie) ---
    if kind == "inundation":
        ic_inrange = ic_sel.map(
            lambda img: mask_inund_band(img, band_index, thr_min, thr_max)
        )
    else:  # anomaly
        ic_inrange = ic_sel.map(
            lambda img: anomaly_band3_thresholded(img, thr_min, thr_max)
        )

    stacked = ic_inrange.toBands()
    pixel_mean = stacked.reduce(ee.Reducer.mean())

    d_in = pixel_mean.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=3000,
        maxPixels=1e13,
    ).getInfo()
    if not d_in:
        in_range_count = 0
    else:
        val_in = list(d_in.values())[0]
        in_range_count = int(val_in) if val_in is not None else 0

    # --- obraz do liczenia wszystkich ważnych pikseli (≠255) ---
    first_img = ic_sel.first()
    if kind == "inundation":
        valid_img = inund_valid_band(first_img, band_index)
    else:
        valid_img = anomaly_band3_corrected(first_img)

    d_tot = valid_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=3000,
        maxPixels=1e13,
    ).getInfo()
    if not d_tot:
        total_count = 0
    else:
        val_tot = list(d_tot.values())[0]
        total_count = int(val_tot) if val_tot is not None else 0

    return in_range_count, total_count


# ---------------------------------------------
# FOLIUM MAP
# ---------------------------------------------
def build_map(image, thr_min, thr_max, kind, mode_label):
    """
    Visualize mean image with appropriate palette and legend.
    """
    band = image.select(0)

    if kind == "anomaly":
        palette = PALETTE_ANOM
    else:
        palette = PALETTE_INUND

    vis = {
        "min": thr_min,
        "max": thr_max,
        "palette": palette,
    }

    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="Esri.WorldImagery")

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
         background-color: white; padding: 10px; border:2px solid grey; z-index:9999;'>
     <b>{mode_label} ({thr_min}–{thr_max})</b><br>
     {legend_rows}
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m

# ---------------------------------------------
# ALTAIR PLOT – min/max/mean
# ---------------------------------------------
def plot_timeseries(df, title, kind, thr_max):
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
# ALTAIR PLOT – stacked pixel counts (all vs in-range)
# ---------------------------------------------
def plot_pixelcount_timeseries(df, title):
    """
    Stacked bar chart: in-range pixels + out-of-range pixels = total.
    Number on top of each bar shows total pixel count.
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
# STREAMLIT UI
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
# 0) DATA TYPE SELECTION
# ---------------------------------------------
mode_label = st.selectbox(
    "Data type:",
    list(DATA_MODES.keys()),
    index=0,
)
mode_cfg = DATA_MODES[mode_label]
kind = mode_cfg["kind"]          # 'inundation' or 'anomaly'
band_index = mode_cfg["band_index"]

# ---------------------------------------------
# 1) DATE RANGE SELECTION (CALENDAR)
# ---------------------------------------------
st.markdown("### Select date range")

date_range = st.date_input(
    "Date range (from–to):",
    value=(MIN_DATE, MIN_DATE),    # default: single day
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    format="YYYY-MM-DD",
)

# Handle both: range (tuple) and single date
if isinstance(date_range, tuple):
    if len(date_range) != 2 or date_range[0] is None or date_range[1] is None:
        st.stop()
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# If user clicked end date before start date -> silently sort
if start_date > end_date:
    start_date, end_date = end_date, start_date

# Clip to data availability
start_date = max(start_date, MIN_DATE)
end_date   = min(end_date,   MAX_DATE)

# Build list of all dates in the selected range
selected_dates = []
current_date = start_date
while current_date <= end_date:
    selected_dates.append(current_date)
    current_date += dt.timedelta(days=1)

if not selected_dates:
    st.stop()

# Convert selected dates to DOY and filter to our available DOY range
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
# 2) THRESHOLD SLIDER
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
# MEAN IMAGE FOR MAP
# ---------------------------------------------
mean_image = build_mean_image(sel_days, thr_min, thr_max, kind, band_index)

# ---------------------------------------------
# MAP WITH DRAWING TOOL
# ---------------------------------------------
m = build_map(mean_image, thr_min, thr_max, kind, mode_label)
map_state = st_folium(
    m,
    height=650,
    width=None,
    key=f"map_{kind}_{band_index}_{sel_days_tuple}_{thr_min}_{thr_max}",
)

st.markdown("---")

# ---------------------------------------------
# STATS & PIXEL COUNTS FOR DRAWN AREA
# ---------------------------------------------
st.subheader("Statistics and pixel counts for the drawn area")

user_min = user_max = user_mean = None
pixel_count_mean = None

feature = None
if map_state is not None:
    feature = map_state.get("last_active_drawing")
    if feature is None:
        drawings = map_state.get("all_drawings")
        if drawings:
            feature = drawings[-1]

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
            sel_days_tuple, thr_min, thr_max, xmin, ymin, xmax, ymax,
            kind, band_index,
        )

        pixel_count_inrange, pixel_count_total = compute_region_pixel_count(
            sel_days_tuple, thr_min, thr_max, xmin, ymin, xmax, ymax,
            kind, band_index,
        )

        region_ts = compute_region_ts_for_bbox(
            sel_days_tuple, thr_min, thr_max, xmin, ymin, xmax, ymax,
            kind, band_index,
        )

        if (
            any(v is None for v in (user_min, user_max, user_mean))
            or pixel_count_mean == 0
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
                c4.metric("In-range pixels (mean anomaly)", f"{pixel_count_inrange}")
                c5.metric("Total valid pixels", f"{pixel_count_total}")
            else:
                c1.metric("Min (area)", f"{user_min:.4f}")
                c2.metric("Max (area)", f"{user_max:.4f}")
                c3.metric("Mean (area)", f"{user_mean:.4f}")
                c4.metric("In-range pixels (mean)", f"{pixel_count_inrange}")
                c5.metric("Total valid pixels", f"{pixel_count_total}")


            if region_ts:
                df_r = pd.DataFrame(region_ts)
                df_r = df_r.sort_values("date").set_index("date")

                col_ts, col_cnt = st.columns(2)

                with col_ts:
                    title_ts = (
                        f"Min / Max / Mean anomaly time series (area, band 3)"
                        if kind == "anomaly"
                        else f"Min / Max / Mean time series (area, {mode_label})"
                    )
                    plot_timeseries(
                        df_r,
                        title_ts,
                        kind,
                        thr_max,
                    )

                with col_cnt:
                    title_cnt = (
                        "Pixel counts in area per day (anomaly band 3)"
                        if kind == "anomaly"
                        else f"Pixel counts in area per day ({mode_label})"
                    )
                    plot_pixelcount_timeseries(
                        df_r,
                        title_cnt,
                    )
            else:
                st.info("No data available to draw time series for the selected area (after masking).")
    else:
        st.info("Draw a rectangular area on the map using the drawing tool.")
else:
    st.info("Draw a rectangular area on the map using the drawing tool (rectangle icon in the top-left corner).")
