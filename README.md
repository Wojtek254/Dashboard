# 🌧️ Dashboard opadów dziennych (CHIRPS, Włochy 2023)

Aplikacja Streamlit wyświetlająca sumaryczny dzienny opad z danych CHIRPS nad Włochami (2023), oparta na Google Earth Engine.

## 🔧 Jak uruchomić lokalnie?

1. **Sklonuj repozytorium**:
    ```bash
    git clone https://github.com/TWOJA_NAZWA_UZYTKOWNIKA/chirps-dashboard.git
    cd chirps-dashboard
    ```

2. **Zainstaluj zależności**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Zaloguj się do Google Earth Engine** (tylko przy pierwszym uruchomieniu):
    ```bash
    earthengine authenticate
    ```

4. **Uruchom aplikację Streamlit**:
    ```bash
    streamlit run app.py
    ```

## 📦 Wymagania

- Konto Google Earth Engine z aktywowanym dostępem: https://signup.earthengine.google.com/
- Python 3.8+ (rekomendowane 3.10)
- Zainstalowany `pip`

## 🗺️ Dane źródłowe

- CHIRPS Daily v2.0 ([GEE katalog](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY))
- Zakres przestrzenny: Włochy (6°E–19°E, 36°N–48°N)
- Zakres czasowy: dowolny wybór dnia z 2023 roku

## 🌐 Dostęp online

Możesz również opublikować ten dashboard na:
👉 [Streamlit Cloud](https://streamlit.io/cloud)

Wystarczy połączyć konto GitHub, wskazać `app.py`, i aplikacja będzie dostępna publicznie.

---

📬 W razie pytań, sugestii lub błędów – zapraszam do zgłaszania issues lub forka repozytorium.
