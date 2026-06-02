import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
import warnings

# Mengabaikan peringatan perhitungan ARIMA agar tidak muncul di terminal
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Prediksi Angin Penerbangan", layout="wide")
st.title("✈️ Aviation Wind Predictor (History 24 Jam)")
st.write("Masukkan riwayat observasi cuaca **24 jam terakhir** pada tabel di bawah ini. Anda dapat mengetik langsung atau melakukan Copy-Paste dari file Excel.")

# ==========================================
# 2. LOAD MODEL & SCALER CNN-LSTM
# ==========================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler_angin') # Pastikan ekstensi .gz jika nama file Anda menggunakan itu
    
    # Arsitektur model harus persis seperti di Colab
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(24, 5)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9))
    
    model.load_weights('model.weights.h5')
    return model, scaler

model, scaler = load_artifacts()

# ==========================================
# 3. FUNGSI PERHITUNGAN ANGIN
# ==========================================
def calculate_wind_component(speed_kt, wind_dir, runway_designator):
    try:
        runway_heading = int(str(runway_designator)[:2]) * 10
    except:
        runway_heading = 0
    angle_diff_rad = math.radians(wind_dir - runway_heading)
    # Positif = Headwind, Negatif = Tailwind
    return speed_kt * math.cos(angle_diff_rad)

# ==========================================
# 4. SIDEBAR - PENGATURAN RUNWAY
# ==========================================
st.sidebar.header("⚙️ Pengaturan Landasan")
runway_aktif = st.sidebar.selectbox("Runway Aktif Saat Ini:", ["09", "27", "18", "36"])
tombol_prediksi = st.sidebar.button("🚀 Hitung Prediksi 9 Jam Kedepan")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Klik sel pada tabel di sebelah kanan untuk mengedit angka, atau tekan **Ctrl+V** untuk menempelkan data dari Excel.")

# ==========================================
# 5. TABEL INPUT DATA (EDITABLE DATAFRAME)
# ==========================================
# Kita buat data default (dummy) agar tabelnya tidak kosong
dummy_data = {
    "Jam ke-": [f"T-{24-i}" for i in range(24)], # T-24 (Kemarin) sampai T-1 (Saat ini)
    "Kecepatan Angin (Knot)": [10.0] * 24,
    "Arah Angin (Derajat)": [90] * 24,
    "Gusting (Knot)": [12.0] * 24,
    "Suhu (°C)": [28.0] * 24,
    "QNH (mb)": [1010.0] * 24
}
df_template = pd.DataFrame(dummy_data)

# Menampilkan tabel yang bisa diedit oleh User
edited_df = st.data_editor(df_template, use_container_width=True, hide_index=True)

# ==========================================
# 6. PROSES PREDIKSI & HASIL
# ==========================================
if tombol_prediksi:
    try:
        # Hitung wind component dari data tabel
        wind_components = []
        for idx, row in edited_df.iterrows():
            comp = calculate_wind_component(
                row["Kecepatan Angin (Knot)"], 
                row["Arah Angin (Derajat)"], 
                runway_aktif
            )
            wind_components.append(comp)

        # ----------------------------------------
        # A. PREDIKSI UTAMA (CNN-LSTM)
        # ----------------------------------------
        df_processed = pd.DataFrame({
            'windAlg.Spd_10m_kt': edited_df["Kecepatan Angin (Knot)"],
            'windAlg.GustAv_10m_kt': edited_df["Gusting (Knot)"],
            'zenoAlg.AT_5m_C': edited_df["Suhu (°C)"],
            'baroAlg.QNH_1m_mb': edited_df["QNH (mb)"],
            'wind_component': wind_components
        })
        
        scaled_input = scaler.transform(df_processed)
        X_pred = scaled_input.reshape(1, 24, 5)
        prediksi_scaled = model.predict(X_pred, verbose=0)
        
        dummy_array = np.zeros((9, 5))
        dummy_array[:, 4] = prediksi_scaled[0]
        prediksi_lstm = scaler.inverse_transform(dummy_array)[:, 4]

        # ----------------------------------------
        # B. PREDIKSI PEMBANDING (REGRESI LINIER)
        # ----------------------------------------
        # Menggunakan deret waktu 1-24 sebagai variabel independen untuk mencari tren
        time_train = np.arange(1, 25).reshape(-1, 1)
        wind_train = np.array(wind_components).reshape(-1, 1)
        
        lr_model = LinearRegression()
        lr_model.fit(time_train, wind_train)
        
        time_test = np.arange(25, 34).reshape(-1, 1)
        prediksi_lr = lr_model.predict(time_test).flatten()

        # ----------------------------------------
        # C. PREDIKSI PEMBANDING (ARIMA)
        # ----------------------------------------
        try:
            # Karena datanya hanya 24 baris, kita gunakan order yang lebih kecil (2,1,0) agar tidak error
            arima_model = ARIMA(wind_components, order=(2, 1, 0))
            arima_fit = arima_model.fit()
            prediksi_arima = arima_fit.forecast(steps=9)
        except Exception as e:
            # Fallback jika pola data dari user terlalu aneh untuk ARIMA
            st.warning(f"ARIMA gagal konvergen pada data ini: {e}")
            prediksi_arima = np.zeros(9)

        # ==========================================
        # 7. TAMPILKAN HASIL KOMPARASI
        # ==========================================
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        # Tabel Gabungan
        df_hasil = pd.DataFrame({
            "Jam ke-": [f"+{i} Jam" for i in range(1, 10)],
            "CNN-LSTM": np.round(prediksi_lstm, 2),
            "Regresi Linier": np.round(prediksi_lr, 2),
            "ARIMA": np.round(prediksi_arima, 2),
            "Status Utama (LSTM)": ["Tailwind 🔴" if x < 0 else "Headwind 🟢" for x in prediksi_lstm]
        })
        
        with col1:
            st.subheader(f"Tabel Komparasi Model (Runway {runway_aktif})")
            st.dataframe(df_hasil, hide_index=True, use_container_width=True)
            
            avg_wind = np.mean(prediksi_lstm)
            if avg_wind < -5:
                st.error("⚠️ **PERINGATAN:** Tailwind kuat diprediksi. Sangat disarankan **CHANGE RUNWAY**!")
            elif avg_wind < 0:
                st.warning("⚠️ **PERHATIAN:** Tailwind ringan terdeteksi. Pantau kondisi dengan saksama.")
            else:
                st.success("✅ **AMAN:** Kondisi Headwind dominan. Runway optimal.")
        
        with col2:
            st.subheader("Grafik Perbandingan Prediksi")
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot 3 Model
            ax.plot(range(1, 10), prediksi_lstm, marker='o', color='blue', linewidth=2, label='CNN-LSTM (Utama)')
            ax.plot(range(1, 10), prediksi_lr, marker='s', color='orange', linestyle='dotted', linewidth=1.5, label='Regresi Linier')
            ax.plot(range(1, 10), prediksi_arima, marker='^', color='green', linestyle='dashdot', linewidth=1.5, label='ARIMA')
            
            ax.axhline(0, color='red', linestyle='-', linewidth=1.5, label='Batas Aman (0)')
            
            # Area shading
            ax.fill_between(range(1, 10), 0, max(10, np.max(prediksi_lstm)+2), color='green', alpha=0.05)
            ax.fill_between(range(1, 10), min(-10, np.min(prediksi_lstm)-2), 0, color='red', alpha=0.05)
            
            ax.set_xlabel("Jam Kedepan")
            ax.set_ylabel("Knot (Positif=Headwind, Negatif=Tailwind)")
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")