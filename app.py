import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. KONFIGURASI HALAMAN & PATH
# ==========================================
st.set_page_config(
    page_title="Prediksi Perceraian Jabar (MLP vs RF)",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Nama File
DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
MODEL_MLP_FILE = MODELS_DIR / "model_mlp.h5"
MODEL_RF_FILE = MODELS_DIR / "model_rf.joblib"

# Nama Kolom (Konstanta)
TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Warna Visualisasi
COLOR_MLP = "#FF4B4B"  # Merah
COLOR_RF = "#1F77B4"   # Biru
COLOR_ACTUAL = "#2CA02C" # Hijau

# ==========================================
# 2. FUNGSI LOAD DATA & CLEANING
# ==========================================
@st.cache_data
def load_dataset():
    """Memuat dataset CSV dan membersihkan nama kolom."""
    if not DATA_FILE.exists():
        st.error(f"File data tidak ditemukan di {DATA_FILE}. Pastikan file CSV sudah diupload ke folder data.")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_FILE)
    
    # --- CLEANING NAMA KOLOM (SESUAI REQUEST) ---
    # Menghapus awalan panjang agar label jadi singkat (misal: "Faktor Ekonomi" -> "Ekonomi")
    new_columns = []
    for col in df.columns:
        clean_name = col.replace("Faktor Penyebab - ", "").replace("Faktor ", "")
        new_columns.append(clean_name)
    df.columns = new_columns
    
    return df

@st.cache_resource
def load_artifacts(df: pd.DataFrame):
    """
    Memuat Model MLP & RF serta membangun ulang Preprocessor.
    """
    try:
        # 1. Identifikasi Kolom (Otomatis dari dataframe yang sudah di-clean namanya)
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c != TARGET_COL]
        
        categorical_cols = [REGION_COL]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        # 2. Bangun & Fit Preprocessor (Penting: Fit ulang dengan nama kolom baru)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )
        preprocessor.fit(df[feature_cols])

        # 3. Load Model
        # Load MLP
        if MODEL_MLP_FILE.exists():
            mlp_model = load_model(MODEL_MLP_FILE, compile=False)
        else:
            mlp_model = None

        # Load RF
        if MODEL_RF_FILE.exists():
            rf_model = joblib.load(MODEL_RF_FILE)
        else:
            rf_model = None

        return preprocessor, mlp_model, rf_model, feature_cols, numeric_cols

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model/preprocessor: {e}")
        return None, None, None, [], []

# --- EKSEKUSI LOAD DATA ---
df = load_dataset()

# Pastikan data ada sebelum lanjut
if df.empty:
    st.stop()

# Load model & preprocessor
preprocessor, mlp_model, rf_model, feature_cols, numeric_cols = load_artifacts(df)

# Daftar Faktor Penyebab (Semua kolom numerik kecuali Tahun)
factor_cols = [c for c in numeric_cols if c != YEAR_COL]
years_list = sorted(df[YEAR_COL].unique())
regions_list = sorted(df[REGION_COL].unique())

# ==========================================
# 3. SIDEBAR (NAVIGASI & FILTER)
# ==========================================
with st.sidebar:
    # Logo dihapus sesuai request
    st.title("🎛️ Navigasi")
    
    page = st.radio("Pilih Halaman:", ["📊 Dashboard Data", "🔮 Prediksi & Perbandingan", "📈 Evaluasi Model"])
    
    st.markdown("---")
    
    # Filter Dashboard
    if page == "📊 Dashboard Data":
        st.header("🔍 Filter Dashboard")
        selected_region_sidebar = st.selectbox("Pilih Wilayah:", ["(Semua)"] + regions_list)
        selected_year_sidebar = st.selectbox("Pilih Tahun:", years_list, index=len(years_list)-1)
    
    # Footer Copyright (Posisi Bawah Sidebar)
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: grey; font-size: 12px;'>
        Copyright By<br>
        <b>Milda Nabilah Al-hamaz</b><br>
        202210715059
        </div>
        """, 
        unsafe_allow_html=True
    )

# ==========================================
# 4. HALAMAN UTAMA
# ==========================================

# --- HALAMAN 1: DASHBOARD DATA ---
if page == "📊 Dashboard Data":
    st.title("📊 Dashboard Data Perceraian Jawa Barat")
    st.markdown("Eksplorasi tren dan faktor penyebab perceraian.")

    # Filter Data
    df_filtered = df.copy()
    if selected_region_sidebar != "(Semua)":
        df_filtered = df_filtered[df_filtered[REGION_COL] == selected_region_sidebar]
    
    # Metrik Ringkasan
    total_cases = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][TARGET_COL].sum()
    avg_cases = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][TARGET_COL].mean()
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Kasus ({selected_year_sidebar})", f"{total_cases:,.0f}")
    c2.metric(f"Rata-rata Wilayah", f"{avg_cases:,.0f}")
    c3.metric("Wilayah Terdata", f"{df_filtered[REGION_COL].nunique()}")

    st.markdown("---")

    # Grafik
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Tren Perceraian")
        trend_df = df_filtered.groupby(YEAR_COL)[TARGET_COL].sum().reset_index()
        fig_trend = px.line(trend_df, x=YEAR_COL, y=TARGET_COL, markers=True, 
                            title=f"Tren Tahunan - {selected_region_sidebar}",
                            line_shape="spline", color_discrete_sequence=[COLOR_ACTUAL])
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader(f"⚠️ Faktor Dominan ({selected_year_sidebar})")
        # Mengambil data faktor (namanya sudah bersih/pendek dari load_dataset)
        df_factors = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][factor_cols].sum().reset_index()
        df_factors.columns = ["Faktor", "Jumlah"]
        df_factors = df_factors.sort_values("Jumlah", ascending=True).tail(10)
        
        fig_factors = px.bar(df_factors, x="Jumlah", y="Faktor", orientation='h',
                             title="Top 10 Penyebab",
                             text_auto='.2s',
                             color="Jumlah", color_continuous_scale="Reds")
        fig_factors.update_layout(yaxis_title=None) # Hilangkan label sumbu Y agar bersih
        st.plotly_chart(fig_factors, use_container_width=True)


# --- HALAMAN 2: PREDIKSI & PERBANDINGAN ---
elif page == "🔮 Prediksi & Perbandingan":
    st.title("🔮 Prediksi & Komparasi Model")
    st.markdown("Simulasi prediksi jumlah perceraian menggunakan **MLP vs Random Forest**.")

    # 1. INPUT UTAMA
    st.subheader("1. Pilih Lokasi & Waktu")
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        input_region = st.selectbox("Kabupaten/Kota:", regions_list)
    with col_main2:
        input_year = st.number_input("Tahun Prediksi:", min_value=2000, max_value=2030, value=2025)

    # Auto-Fill Logika
    region_data = df[df[REGION_COL] == input_region]
    if not region_data.empty:
        default_values = region_data[factor_cols].mean().fillna(0).astype(int).to_dict()
        st.success(f"✅ Input faktor otomatis diisi data rata-rata **{input_region}**.")
    else:
        default_values = {col: 0 for col in factor_cols}

    # 2. INPUT FAKTOR (DROPDOWN / EXPANDER)
    st.markdown("### 2. Input Faktor Penyebab")
    # Menggunakan Expander yang bertindak sebagai "Dropdown Menu" untuk input detail
    with st.expander("👇 Klik di sini untuk membuka/mengubah Angka Faktor", expanded=False):
        st.caption("Ubah angka di bawah untuk mensimulasikan kondisi (Misal: Ekonomi naik).")
        
        input_data = {}
        # Layout Grid agar rapi
        cols = st.columns(3) 
        for i, col_name in enumerate(factor_cols):
            with cols[i % 3]:
                # Label sudah bersih (pendek)
                val = st.number_input(
                    f"{col_name}", 
                    min_value=0, 
                    value=int(default_values.get(col_name, 0)),
                    key=f"input_{col_name}"
                )
                input_data[col_name] = val

    # Tombol Eksekusi
    if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
        if mlp_model is None or rf_model is None:
            st.error("Model tidak ditemukan di folder 'models/'.")
        else:
            # Siapkan Data Input
            input_row = {YEAR_COL: input_year, REGION_COL: input_region}
            input_row.update(input_data)
            
            df_input = pd.DataFrame([input_row])
            df_input = df_input[feature_cols] # Urutan kolom harus sama

            try:
                # Preprocessing
                X_input = preprocessor.transform(df_input)
                
                # Prediksi
                pred_mlp = mlp_model.predict(X_input)
                val_mlp = float(pred_mlp[0][0]) if pred_mlp.ndim > 1 else float(pred_mlp[0])
                
                pred_rf = rf_model.predict(X_input)
                val_rf = float(pred_rf[0])

                # Hasil
                st.divider()
                st.subheader("🎯 Hasil Prediksi")

                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("MLP (Neural Network)", f"{val_mlp:,.0f}", delta_color="off")
                c_res2.metric("Random Forest", f"{val_rf:,.0f}", delta_color="off")
                c_res3.metric("Selisih", f"{abs(val_mlp - val_rf):,.0f}")

                # Grafik Perbandingan
                comp_df = pd.DataFrame({
                    "Algoritma": ["MLP (Neural Net)", "Random Forest"],
                    "Prediksi": [val_mlp, val_rf]
                })

                fig_comp = px.bar(
                    comp_df, x="Algoritma", y="Prediksi", color="Algoritma",
                    text_auto='.2s', title="Perbandingan Kedua Algoritma",
                    color_discrete_map={"MLP (Neural Net)": COLOR_MLP, "Random Forest": COLOR_RF}
                )
                fig_comp.update_layout(showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.error(f"Error saat prediksi: {e}")


# --- HALAMAN 3: EVALUASI MODEL ---
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Performa Model")
    st.markdown("Evaluasi akurasi berdasarkan data tahun terakhir (Testing).")

    test_year = years_list[-1]
    df_test = df[df[YEAR_COL] == test_year].copy()
    
    if df_test.empty:
        st.warning("Data testing tidak cukup.")
    else:
        # Proses Evaluasi
        X_test = preprocessor.transform(df_test[feature_cols])
        y_true = df_test[TARGET_COL].values

        y_pred_mlp = mlp_model.predict(X_test).flatten()
        y_pred_rf = rf_model.predict(X_test)

        # Hitung Error
        metrics_data = {
            "Model": ["MLP", "Random Forest"],
            "MAE (Mean Absolute Error)": [
                mean_absolute_error(y_true, y_pred_mlp),
                mean_absolute_error(y_true, y_pred_rf)
            ],
            "RMSE (Root Mean Squared Error)": [
                np.sqrt(mean_squared_error(y_true, y_pred_mlp)),
                np.sqrt(mean_squared_error(y_true, y_pred_rf))
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data).set_index("Model")

        st.subheader(f"📊 Metrik Error (Data Tahun {test_year})")
        st.table(metrics_df.round(2)) # Round agar rapi dan tidak error

        # Scatter Plot
        st.subheader("🎯 Aktual vs Prediksi")
        
        fig_scatter = go.Figure()
        
        # Garis Perfect
        min_v, max_v = y_true.min(), y_true.max()
        fig_scatter.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode='lines', name='Perfect', line=dict(color='gray', dash='dash')
        ))

        # MLP
        fig_scatter.add_trace(go.Scatter(
            x=y_true, y=y_pred_mlp,
            mode='markers', name='MLP', marker=dict(color=COLOR_MLP, size=10, opacity=0.7)
        ))

        # RF
        fig_scatter.add_trace(go.Scatter(
            x=y_true, y=y_pred_rf,
            mode='markers', name='RF', marker=dict(color=COLOR_RF, size=10, symbol='x')
        ))

        fig_scatter.update_layout(
            xaxis_title="Jumlah Aktual",
            yaxis_title="Jumlah Prediksi",
            height=500,
            title="Semakin dekat ke garis putus-putus, semakin akurat."
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
