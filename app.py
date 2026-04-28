# app.py
# pip install streamlit folium earthengine-api streamlit-folium pandas altair google-auth

import datetime as dt
import io

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

BAND_OPTIONS = {
    1: "Daily observations",
    2: "3-daily interpolated data",
    3: "Interpolation flux",
    4: "1-day inundation anomalies",
    5: "Interpolated inundation anomalies",
}

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
NDVI_COLLECTION = "MODIS/061/MOD13Q1"
POP_COLLECTION = "CIESIN/GPWv411/GPW_Population_Count"


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
    selected_dates = []
    current_date = start_date
    while current_date <= end_date:
        selected_dates.append(current_date)
        current_date += dt.timedelta(days=1)

    year_start = dt.date(YEAR, 1, 1)
    doys = [
        (d - year_start).days + 1
        for d in selected_dates
        if START_DOY <= (d - year_start).days + 1 <= END_DOY
    ]
    return selected_dates, sorted(doys)


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


def add_optional_overlays(base_rgb, start_date, end_date, add_chirps, add_ndvi, add_population):
    start_str = start_date.strftime("%Y-%m-%d")
    end_exclusive = (end_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    out = base_rgb

    if add_chirps:
        chirps = (
            ee.ImageCollection(CHIRPS_COLLECTION)
            .filterDate(start_str, end_exclusive)
            .select("precipitation")
            .sum()
        )
        chirps_vis = chirps.visualize(
            min=0,
            max=20,
            palette=["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
            opacity=0.55,
        )
        out = out.blend(chirps_vis)

    if add_ndvi:
        ndvi = (
            ee.ImageCollection(NDVI_COLLECTION)
            .filterDate(start_str, end_exclusive)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
        )
        ndvi_vis = ndvi.visualize(
            min=0.0,
            max=0.8,
            palette=["#f7fcf5", "#a1d99b", "#31a354", "#006d2c"],
            opacity=0.55,
        )
        out = out.blend(ndvi_vis)

    if add_population:
        pop = (
            ee.ImageCollection(POP_COLLECTION)
            .sort("system:time_start", False)
            .first()
            .select("population_count")
        )
        pop_vis = pop.visualize(
            min=0,
            max=1000,
            palette=["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
            opacity=0.5,
        )
        out = out.blend(pop_vis)

    return out


def build_side_visual_image(
    selected_days,
    thr_min,
    thr_max,
    kind,
    band_index,
    start_date,
    end_date,
    add_chirps,
    add_ndvi,
    add_population,
):
    mean_image = build_mean_image(selected_days, thr_min, thr_max, kind, band_index)
    palette = PALETTE_ANOM if kind == "anomaly" else PALETTE_INUND

    base_rgb = mean_image.select(0).visualize(
        min=thr_min,
        max=thr_max,
        palette=palette,
    )

    return add_optional_overlays(
        base_rgb,
        start_date,
        end_date,
        add_chirps=add_chirps,
        add_ndvi=add_ndvi,
        add_population=add_population,
    )


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
    left_visual_image,
    left_label,
    left_thr_min,
    left_thr_max,
    left_kind,
    saved_feature=None,
    map_center=None,
    map_zoom=None,
    right_visual_image=None,
    right_label=None,
):
    try:
        if map_center is None:
            map_center = CENTER
        if map_zoom is None:
            map_zoom = ZOOM

        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="Esri.WorldImagery")

        left_map_id = left_visual_image.getMapId({})
        left_tile_url = left_map_id["tile_fetcher"].url_format

        left_layer = folium.TileLayer(
            tiles=left_tile_url,
            attr="Google Earth Engine",
            name=left_label,
            overlay=True,
            control=True,
        )
        left_layer.add_to(m)

        if right_visual_image is not None:
            right_map_id = right_visual_image.getMapId({})
            right_tile_url = right_map_id["tile_fetcher"].url_format

            if right_label is None:
                right_label = "Right layer"

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

        if left_kind == "anomaly":
            num_classes = len(PALETTE_ANOM)
            step = (left_thr_max - left_thr_min) / (num_classes - 1) if num_classes > 1 else 1
            ticks = [left_thr_min + i * step for i in range(num_classes)]
            colors = PALETTE_ANOM
            width = 260
        else:
            num_classes = len(PALETTE_INUND)
            step = (left_thr_max - left_thr_min) / (num_classes - 1) if num_classes > 1 else 1
            ticks = [left_thr_min + i * step for i in range(num_classes)]
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
         <b>{left_label} ({left_thr_min}–{left_thr_max})</b><br>
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
        f"LEFT: {view_info['left_label']}",
        f"RIGHT: {view_info['right_label']}",
        f"Map center: {view_info['map_center'][0]:.4f}, {view_info['map_center'][1]:.4f} | zoom: {view_info['map_zoom']}",
        f"Overlays LEFT: CHIRPS={view_info['left_chirps']}, NDVI={view_info['left_ndvi']}, POP={view_info['left_pop']}",
        f"Overlays RIGHT: CHIRPS={view_info['right_chirps']}, NDVI={view_info['right_ndvi']}, POP={view_info['right_pop']}",
    ]
    for line in header_lines:
        draw.text((left_margin, y), line, fill="black", font=font_body)
        y += line_h

    y += line_h
    draw.text((left_margin, y), "LEFT statistics", fill="black", font=font_title)
    y += line_h * 2

    rows = [
        ("Stats source", "LEFT asset layer only"),
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
# APP HEADER
# ---------------------------------------------
st.title("CYGNSS – Regional Viewer")
st.caption(
    "Compare left/right CYGNSS bands over independent date ranges and optional GEE overlays "
    "(CHIRPS precipitation, NDVI, population). "
    "Statistics are calculated only for the LEFT CYGNSS asset layer."
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

split_view = st.checkbox(
    "Enable split-view map comparison (left vs right)",
    value=True,
)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### LEFT panel")
    left_band_number = st.selectbox(
        "LEFT band:",
        list(BAND_OPTIONS.keys()),
        index=0,
        format_func=lambda b: f"Band {b} – {BAND_OPTIONS[b]}",
    )

    left_date_range = st.date_input(
        "LEFT date range (from–to):",
        value=(MIN_DATE, MIN_DATE),
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        format="YYYY-MM-DD",
        key="left_date_range",
    )

    left_add_chirps = st.checkbox("LEFT: add CHIRPS precipitation layer", value=False)
    left_add_ndvi = st.checkbox("LEFT: add NDVI layer", value=False)
    left_add_population = st.checkbox("LEFT: add population layer", value=False)

with right_col:
    st.markdown("### RIGHT panel")
    right_band_number = st.selectbox(
        "RIGHT band:",
        list(BAND_OPTIONS.keys()),
        index=0,
        format_func=lambda b: f"Band {b} – {BAND_OPTIONS[b]}",
        disabled=not split_view,
    )

    right_date_range = st.date_input(
        "RIGHT date range (from–to):",
        value=(MIN_DATE, MIN_DATE),
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        format="YYYY-MM-DD",
        key="right_date_range",
        disabled=not split_view,
    )

    right_add_chirps = st.checkbox(
        "RIGHT: add CHIRPS precipitation layer", value=False, disabled=not split_view
    )
    right_add_ndvi = st.checkbox("RIGHT: add NDVI layer", value=False, disabled=not split_view)
    right_add_population = st.checkbox(
        "RIGHT: add population layer", value=False, disabled=not split_view
    )

left_start_date, left_end_date = parse_date_range(left_date_range)
if left_start_date is None:
    st.warning("Invalid LEFT date range.")
    st.stop()

left_selected_dates, left_sel_days = dates_to_doys(left_start_date, left_end_date)
if not left_sel_days:
    st.warning("No valid LEFT dataset days found in selected range.")
    st.stop()

left_kind = band_kind(left_band_number)
left_band_index = left_band_number - 1
left_label = (
    f"LEFT | Band {left_band_number}: {BAND_OPTIONS[left_band_number]} | "
    f"{left_start_date.strftime('%Y-%m-%d')}→{left_end_date.strftime('%Y-%m-%d')}"
)

if left_kind == "anomaly":
    left_thr_min, left_thr_max = st.slider(
        "LEFT threshold range:",
        min_value=-100,
        max_value=100,
        value=(-20, 20),
        step=1,
        key="left_thr",
    )
else:
    left_thr_min, left_thr_max = st.slider(
        "LEFT threshold range:",
        min_value=0,
        max_value=100,
        value=(20, 100),
        step=1,
        key="left_thr",
    )

if left_thr_min >= left_thr_max:
    st.error("LEFT lower threshold must be smaller than upper threshold.")
    st.stop()

if split_view:
    right_start_date, right_end_date = parse_date_range(right_date_range)
    if right_start_date is None:
        st.warning("Invalid RIGHT date range.")
        st.stop()

    right_selected_dates, right_sel_days = dates_to_doys(right_start_date, right_end_date)
    if not right_sel_days:
        st.warning("No valid RIGHT dataset days found in selected range.")
        st.stop()

    right_kind = band_kind(right_band_number)
    right_band_index = right_band_number - 1
    right_label = (
        f"RIGHT | Band {right_band_number}: {BAND_OPTIONS[right_band_number]} | "
        f"{right_start_date.strftime('%Y-%m-%d')}→{right_end_date.strftime('%Y-%m-%d')}"
    )

    if right_kind == "anomaly":
        right_thr_min, right_thr_max = st.slider(
            "RIGHT threshold range:",
            min_value=-100,
            max_value=100,
            value=(-20, 20),
            step=1,
            key="right_thr",
        )
    else:
        right_thr_min, right_thr_max = st.slider(
            "RIGHT threshold range:",
            min_value=0,
            max_value=100,
            value=(20, 100),
            step=1,
            key="right_thr",
        )

    if right_thr_min >= right_thr_max:
        st.error("RIGHT lower threshold must be smaller than upper threshold.")
        st.stop()
else:
    right_sel_days = None
    right_start_date = None
    right_end_date = None
    right_kind = None
    right_band_index = None
    right_thr_min = None
    right_thr_max = None
    right_label = None

st.write("LEFT dates used:", ", ".join(d.strftime("%Y-%m-%d") for d in left_selected_dates))
if split_view and right_start_date is not None:
    st.write("RIGHT dates used:", ", ".join(d.strftime("%Y-%m-%d") for d in right_selected_dates))

# ---------------------------------------------
# BUILD IMAGES FOR MAP
# ---------------------------------------------
try:
    left_visual_image = build_side_visual_image(
        selected_days=left_sel_days,
        thr_min=left_thr_min,
        thr_max=left_thr_max,
        kind=left_kind,
        band_index=left_band_index,
        start_date=left_start_date,
        end_date=left_end_date,
        add_chirps=left_add_chirps,
        add_ndvi=left_add_ndvi,
        add_population=left_add_population,
    )
except Exception as e:
    st.error(f"Failed to build LEFT image: {e}")
    st.stop()

right_visual_image = None
if split_view and right_sel_days is not None:
    try:
        right_visual_image = build_side_visual_image(
            selected_days=right_sel_days,
            thr_min=right_thr_min,
            thr_max=right_thr_max,
            kind=right_kind,
            band_index=right_band_index,
            start_date=right_start_date,
            end_date=right_end_date,
            add_chirps=right_add_chirps,
            add_ndvi=right_add_ndvi,
            add_population=right_add_population,
        )
    except Exception as e:
        st.error(f"Failed to build RIGHT image: {e}")
        st.stop()

# ---------------------------------------------
# BUILD / DISPLAY MAP
# ---------------------------------------------
m = build_map(
    left_visual_image=left_visual_image,
    left_label=left_label,
    left_thr_min=left_thr_min,
    left_thr_max=left_thr_max,
    left_kind=left_kind,
    saved_feature=st.session_state.saved_feature,
    map_center=st.session_state.map_center,
    map_zoom=st.session_state.map_zoom,
    right_visual_image=right_visual_image,
    right_label=right_label,
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
# STATS & COUNTS FOR SELECTED REGION (LEFT ONLY)
# ---------------------------------------------
st.subheader("Statistics and pixel counts for the drawn area (LEFT asset layer only)")

user_min = user_max = user_mean = None
left_sel_days_tuple = tuple(left_sel_days)
pixel_count_inrange = None
pixel_count_total = None
region_drawn = "No"

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

        user_min, user_max, user_mean = compute_region_summary_for_bbox(
            left_sel_days_tuple,
            left_thr_min,
            left_thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            left_kind,
            left_band_index,
        )

        pixel_count_inrange, pixel_count_total = compute_region_pixel_count(
            left_sel_days_tuple,
            left_thr_min,
            left_thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            left_kind,
            left_band_index,
        )

        region_ts = compute_region_ts_for_bbox(
            left_sel_days_tuple,
            left_thr_min,
            left_thr_max,
            xmin,
            ymin,
            xmax,
            ymax,
            left_kind,
            left_band_index,
        )

        if any(v is None for v in (user_min, user_max, user_mean)) or pixel_count_total == 0:
            st.info(
                "There are no valid pixels in the selected area "
                "for the chosen LEFT thresholds/scale. Try a larger area or different thresholds."
            )
        else:
            c1, c2, c3, c4, c5 = st.columns(5)

            if left_kind == "anomaly":
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
                        f"Min / Max / Mean anomaly time series (LEFT area, band {left_band_number})"
                        if left_kind == "anomaly"
                        else f"Min / Max / Mean time series (LEFT area, band {left_band_number})"
                    )
                    plot_timeseries(df_r, title_ts, left_kind, left_thr_max)

                with col_cnt:
                    title_cnt = f"Daily pixel counts in LEFT area (band {left_band_number})"
                    plot_pixelcount_timeseries(df_r, title_cnt)
            else:
                st.info("No data available to draw LEFT time series for the selected area (after masking).")
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
    "left_chirps": left_add_chirps,
    "left_ndvi": left_add_ndvi,
    "left_pop": left_add_population,
    "right_chirps": right_add_chirps if split_view else False,
    "right_ndvi": right_add_ndvi if split_view else False,
    "right_pop": right_add_population if split_view else False,
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
