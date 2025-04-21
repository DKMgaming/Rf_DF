import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt
import folium
from streamlit_folium import st_folium

# --- Hàm phụ ---
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    azimuth = (degrees(atan2(x, y)) + 360) % 360
    return azimuth

def simulate_signal_strength(dist_km, h, freq_mhz):
    path_loss = 32.45 + 20 * np.log10(dist_km + 0.1) + 20 * np.log10(freq_mhz + 1)
    return -30 - path_loss + 10 * np.log10(h + 1)

def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    R = 6371.0  # Bán kính Trái Đất (km)
    brng = radians(azimuth_deg)
    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = np.arcsin(sin(lat1) * cos(distance_km / R) + cos(lat1) * sin(distance_km / R) * cos(brng))
    lon2 = lon1 + atan2(sin(brng) * sin(distance_km / R) * cos(lat1),
                        cos(distance_km / R) - sin(lat1) * sin(lat2))

    return degrees(lat2), degrees(lon2)

# --- Giao diện ---
st.set_page_config(layout="wide")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

tab1, tab2 = st.tabs(["1. Huấn luyện mô hình", "2. Dự đoán tọa độ"])

# --- Tab 1: Huấn luyện ---
with tab1:
    st.subheader("📡 Sinh dữ liệu mô phỏng và huấn luyện mô hình")

    if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
        np.random.seed(42)
        n_samples = 1000
        data = []
        for _ in range(n_samples):
            lat_tx = np.random.uniform(10.0, 21.0)
            lon_tx = np.random.uniform(105.0, 109.0)
            lat_rx = lat_tx + np.random.uniform(-0.05, 0.05)
            lon_rx = lon_tx + np.random.uniform(-0.05, 0.05)
            h_rx = np.random.uniform(5, 50)
            freq = np.random.uniform(400, 2600)

            azimuth = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
            distance = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
            signal = simulate_signal_strength(distance, h_rx, freq)

            data.append({
                "lat_receiver": lat_rx,
                "lon_receiver": lon_rx,
                "antenna_height": h_rx,
                "azimuth": azimuth,
                "frequency": freq,
                "signal_strength": signal,
                "distance_km": distance
            })

        df = pd.DataFrame(data)
        df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
        df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))

        X = df[['lat_receiver', 'lon_receiver', 'antenna_height', 'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
        y = df[['distance_km']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.success(f"Huấn luyện xong - MAE khoảng cách: {mae:.3f} km")

        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        st.download_button(
            label="📥 Tải mô hình huấn luyện (.joblib)",
            data=buffer,
            file_name="distance_model.joblib",
            mime="application/octet-stream"
        )

# --- Tab 2: Dự đoán ---
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")

    uploaded_model = st.file_uploader("📂 Tải mô hình đã huấn luyện (.joblib)", type=["joblib"])
    if uploaded_model:
        model = joblib.load(uploaded_model)

        with st.form("input_form"):
            lat_rx = st.number_input("Vĩ độ trạm thu", value=16.0)
            lon_rx = st.number_input("Kinh độ trạm thu", value=108.0)
            h_rx = st.number_input("Chiều cao anten (m)", value=30.0)
            signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80.0)
            freq = st.number_input("Tần số (MHz)", value=900.0)
            azimuth = st.number_input("Góc phương vị (độ)", value=45.0)
            submitted = st.form_submit_button("🔍 Dự đoán tọa độ nguồn phát")

        if submitted:
            az_sin = np.sin(np.radians(azimuth))
            az_cos = np.cos(np.radians(azimuth))
            X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])
            predicted_distance = model.predict(X_input)[0]

            lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, predicted_distance)

            st.success("🎯 Tọa độ nguồn phát xạ dự đoán:")
            st.markdown(f"- **Vĩ độ**: `{lat_pred:.6f}`")
            st.markdown(f"- **Kinh độ**: `{lon_pred:.6f}`")
            st.markdown(f"- **Khoảng cách dự đoán**: `{predicted_distance:.2f} km`")

            m = folium.Map(location=[lat_rx, lon_rx], zoom_start=10)
            folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
            folium.Marker([lat_pred, lon_pred], tooltip="Nguồn phát dự đoán", icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine(locations=[[lat_rx, lon_rx], [lat_pred, lon_pred]], color='green').add_to(m)

            with st.container():
                st_folium(m, width=700, height=500, returned_objects=[])
