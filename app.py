# === Streamlit Dashboard: Opady CHIRPS z Earth Engine (Włochy 2023) ===
import streamlit as st
import geemap.foliumap as geemap
import ee
import datetime

try:
    ee.Initialize()
except Exception as e:
    import streamlit as st
    st.warning("🔑 Potrzebna autoryzacja Google Earth Engine.")
    with st.expander("Kliknij tutaj, aby się zalogować"):
        try:
            ee.Authenticate()
            ee.Initialize()
            st.success("✅ Autoryzacja zakończona sukcesem!")
        except Exception as auth_error:
            st.error("❌ Błąd autoryzacji: " + str(auth_error))
            st.stop()

st.set_page_config(layout="wide")
st.title("📊 Dashboard opadów dziennych (CHIRPS, Włochy 2023)")

# Lista dat: wszystkie dni 2023
base_date = datetime.date(2023, 1, 1)
dates = [(base_date + datetime.timedelta(days=i)) for i in range(365)]
labels = [d.strftime("%Y-%m-%d") for d in dates]

# Wybór dat
selected_labels = []
st.subheader("Wybierz dni")
row_size = 12
for i in range(0, len(labels), row_size):
    cols = st.columns(row_size)
    for j, d in enumerate(labels[i:i+row_size]):
        if cols[j].checkbox(d, value=(i == 0 and j == 0)):
            selected_labels.append(d)

if not selected_labels:
    st.warning("Wybierz co najmniej jeden dzień.")
    st.stop()

# Ustaw region: Włochy
region = ee.Geometry.BBox(6, 36, 19, 48)

# Pobierz obrazy CHIRPS z GEE
collection = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterBounds(region)
              .filterDate(selected_labels[0], (datetime.datetime.strptime(selected_labels[-1], "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
              .select("precipitation"))

# Mapuj do właściwych dat (uwzględnia tylko wybrane)
def filter_and_label(date_str):
    return collection.filterDate(date_str, (datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")).first()

images = [filter_and_label(d) for d in selected_labels]
filtered = ee.ImageCollection(images)
sum_image = filtered.sum()

# Statystyki
mean_sum_val = sum_image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=region,
    scale=5000,
    maxPixels=1e13
).get("precipitation").getInfo()

max_sum_val = sum_image.reduceRegion(
    reducer=ee.Reducer.max(),
    geometry=region,
    scale=5000,
    maxPixels=1e13
).get("precipitation").getInfo()

col1, col2 = st.columns(2)
col1.metric("Średni sumaryczny opad [mm]", f"{mean_sum_val:.2f}")
col2.metric("Maks. sumaryczny opad [mm]", f"{max_sum_val:.2f}")

# Mapa
m = geemap.Map(center=[42, 12], zoom=5, basemap="HYBRID")
vis = {"min": 0.01, "max": 100, "palette": ["white", "blue"]}
masked_layer = sum_image.updateMask(sum_image.gt(0)).clip(region)
m.addLayer(masked_layer, vis, f"Suma opadów: {', '.join(selected_labels)}")
m.add_colorbar(vis, label="Opad [mm]", layer_name="Legenda opadów")
m.to_streamlit(height=600)
