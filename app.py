# app.py
# pip install streamlit folium earthengine-api streamlit-folium pandas altair google-auth

import datetime as dt
import io
import re

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
    page_title="UCAR/CU CYGNSS v3.1 inundation data viewer",
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

PALETTE_COUNT = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c",
]
AGGREGATIONS = {"mean": "Mean", "min": "Minimum", "max": "Maximum", "count": "Valid-day count"}
TEMPORAL_AGGREGATIONS = ("mean", "min", "max")
TEMPORAL_EXTERNAL_LAYERS = ("chirps", "ndvi")
DAILY_CYGNSS_LAYERS = ("cygnss_1", "cygnss_4")

BAND_OPTIONS = {
    1: "1-day inundation observations",
    2: "3-day interpolated inundation",
    3: "Interpolation flags",
    4: "1-day inundation anomalies",
    5: "3-day interpolated inundation anomalies",
}

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
NDVI_COLLECTION = "MODIS/061/MOD13Q1"
POP_COLLECTION = "CIESIN/GPWv411/GPW_Population_Density"
ELEVATION_IMAGE = "USGS/SRTMGL1_003"

LAYER_OPTIONS = {
    "none": "None",
    "cygnss_1": f"CYGNSS - {BAND_OPTIONS[1]}",
    "cygnss_2": f"CYGNSS - {BAND_OPTIONS[2]}",
    "cygnss_4": f"CYGNSS - {BAND_OPTIONS[4]}",
    "cygnss_5": f"CYGNSS - {BAND_OPTIONS[5]}",
    "chirps": "Precipitation (CHIRPS)",
    "ndvi": "Vegetation (NDVI)",
    "population_density": "Population density",
    "elevation": "Elevation a.s.l.",
}

OVERLAY_LEGENDS = {
