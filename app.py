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

# ==========================================
# 1. KONFIGURASI HALAMAN & PATH
# ==========================================
st.set_page_config(
    page_title="Dashboard Prediksi Perceraian Jabar (MLP vs RF)",
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

# Kolom Target & Fitur
TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Warna Visualisasi (Konsisten di seluruh aplikasi)
COLOR_MLP = "#FF4B4B"  # Merah Streamlit
COLOR_RF = "#1F77B4"   # Biru Plotly
COLOR_ACTUAL = "#2CA02C" # Hijau

# ==========================================
# 2. FUNGSI LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_dataset():
    """Memuat dataset CSV."""
    if not DATA_FILE.exists():
        st.error(f"File data tidak ditemukan di {DATA_FILE}. Pastikan sudah diupload.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    return df

@st.cache_resource
def load_artifacts(df: pd.DataFrame):
    """
    Memuat Model MLP & RF serta membangun ulang Preprocessor.
    Membangun ulang preprocessor lebih aman daripada meload .joblib 
    untuk menghindari error versi scikit-learn di cloud.
    """
    try:
        # 1. Identifikasi Kolom
        all_cols = df.columns.tolist()
        # Fitur adalah semua kolom kecuali Target ('Jumlah')
        feature_cols = [c for c in all_cols if c != TARGET_COL]
        
        categorical_cols = [REGION_COL]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        # 2. Bangun & Fit Preprocessor
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
            st.warning("Model MLP tidak ditemukan. Upload 'model_mlp.h5' ke folder models/.")
            mlp_model = None

        # Load RF
        if MODEL_RF_FILE.exists():
            rf_model = joblib.load(MODEL_RF_FILE)
        else:
            st.warning("Model RF tidak ditemukan. Upload 'model_rf.joblib' ke folder models/.")
            rf_model = None

        return preprocessor, mlp_model, rf_model, feature_cols, numeric_cols

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model/preprocessor: {e}")
        return None, None, None, [], []

# Load data awal
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
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.title("🎛️ Navigasi")
    
    page = st.radio("Pilih Halaman:", ["📊 Dashboard Data", "🔮 Prediksi & Perbandingan", "📈 Evaluasi Model"])
    
    st.markdown("---")
    st.header("🔍 Filter Global")
    selected_region_sidebar = st.selectbox("Pilih Wilayah (Untuk Dashboard):", ["(Semua)"] + regions_list)
    selected_year_sidebar = st.selectbox("Pilih Tahun:", years_list, index=len(years_list)-1)
    
    st.markdown("---")
    st.info(
        "**Tentang Proyek:**\n"
        "Aplikasi ini membandingkan performa **Multi-Layer Perceptron (MLP)** "
        "dan **Random Forest (RF)** dalam memprediksi jumlah perceraian di Jawa Barat "
        "berdasarkan berbagai faktor penyebab."
    )

# ==========================================
# 4. HALAMAN UTAMA
# ==========================================

# --- HALAMAN 1: DASHBOARD DATA ---
if page == "📊 Dashboard Data":
    st.title("📊 Dashboard Data Perceraian Jawa Barat")
    st.markdown("Eksplorasi tren dan faktor penyebab perceraian berdasarkan data historis.")

    # Filter Data untuk Visualisasi
    df_filtered = df.copy()
    if selected_region_sidebar != "(Semua)":
        df_filtered = df_filtered[df_filtered[REGION_COL] == selected_region_sidebar]
    
    # Layout Atas: Metrik Ringkasan
    total_cases = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][TARGET_COL].sum()
    avg_cases = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][TARGET_COL].mean()
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Perceraian ({selected_year_sidebar})", f"{total_cases:,.0f}")
    c2.metric(f"Rata-rata per Wilayah ({selected_year_sidebar})", f"{avg_cases:,.0f}")
    c3.metric("Jumlah Wilayah Data", f"{df_filtered[REGION_COL].nunique()}")

    st.markdown("---")

    # Baris 1 Grafik: Tren Tahunan & Distribusi Faktor
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Tren Perceraian dari Waktu ke Waktu")
        # Group by tahun untuk melihat total
        trend_df = df_filtered.groupby(YEAR_COL)[TARGET_COL].sum().reset_index()
        fig_trend = px.line(trend_df, x=YEAR_COL, y=TARGET_COL, markers=True, 
                            title=f"Tren Jumlah Perceraian - {selected_region_sidebar}",
                            line_shape="spline", color_discrete_sequence=[COLOR_ACTUAL])
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader(f"causes Faktor Penyebab Dominan ({selected_year_sidebar})")
        # Ambil data tahun terpilih, sum semua faktor
        df_factors = df_filtered[df_filtered[YEAR_COL] == selected_year_sidebar][factor_cols].sum().reset_index()
        df_factors.columns = ["Faktor", "Jumlah"]
        df_factors = df_factors.sort_values("Jumlah", ascending=True).tail(10) # Top 10
        
        fig_factors = px.bar(df_factors, x="Jumlah", y="Faktor", orientation='h',
                             title="Top 10 Faktor Penyebab Perceraian",
                             color="Jumlah", color_continuous_scale="Reds")
        st.plotly_chart(fig_factors, use_container_width=True)

# --- HALAMAN 2: PREDIKSI & PERBANDINGAN ---
elif page == "🔮 Prediksi & Perbandingan":
    st.title("🔮 Prediksi & Komparasi Model")
    st.markdown(
        "Masukkan parameter di bawah ini. Sistem akan memprediksi jumlah perceraian menggunakan **dua algoritma sekaligus** "
        "sehingga Anda bisa membandingkan hasilnya."
    )

    with st.form("prediction_form"):
        st.subheader("1. Atur Parameter Input")
        
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            input_region = st.selectbox("Kabupaten/Kota", regions_list)
        with c_in2:
            input_year = st.number_input("Tahun Prediksi", min_value=2000, max_value=2030, value=2025)

        st.markdown("#### 2. Masukkan Jumlah Kasus per Faktor Penyebab")
        st.caption("Jika tidak ada data, biarkan 0.")
        
        # Buat input fields secara dinamis dalam grid
        input_data = {}
        cols = st.columns(3) # 3 kolom input
        for i, col_name in enumerate(factor_cols):
            with cols[i % 3]:
                # Ambil nilai default/mean dari dataset agar user tidak mulai dari 0 semua
                default_val = int(df[col_name].mean())
                val = st.number_input(f"{col_name}", min_value=0, value=0)
                input_data[col_name] = val
        
        submitted = st.form_submit_button("🚀 Jalankan Prediksi", type="primary")

    # LOGIKA PREDIKSI
    if submitted:
        if mlp_model is None or rf_model is None:
            st.error("Model belum dimuat dengan benar. Periksa folder models.")
        else:
            # 1. Siapkan DataFrame Input (Sesuai urutan feature_cols saat training)
            input_row = {YEAR_COL: input_year, REGION_COL: input_region}
            input_row.update(input_data)
            
            df_input = pd.DataFrame([input_row])
            
            # Pastikan urutan kolom sama dengan feature_cols
            df_input = df_input[feature_cols]

            # 2. Preprocessing
            try:
                X_input = preprocessor.transform(df_input)
                
                # 3. Prediksi
                # MLP (hasilnya array 2D, perlu flatten)
                pred_mlp = mlp_model.predict(X_input)
                val_mlp = float(pred_mlp[0][0])
                
                # RF (hasilnya array 1D)
                pred_rf = rf_model.predict(X_input)
                val_rf = float(pred_rf[0])

                # Hitung total manual dari input (sebagai referensi logika)
                total_factors = sum(input_data.values())

                st.markdown("---")
                st.subheader("🎯 Hasil Prediksi & Perbandingan")

                # Tampilan Metrik Berdampingan
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"### 🧠 MLP Model")
                    st.metric("Prediksi Total", f"{val_mlp:,.0f}", delta="Neural Network")
                
                with col_res2:
                    st.markdown(f"### 🌲 Random Forest")
                    st.metric("Prediksi Total", f"{val_rf:,.0f}", delta="Ensemble")
                
                with col_res3:
                    st.markdown("### 🔢 Sum of Factors")
                    st.caption("Penjumlahan manual input faktor")
                    st.metric("Total Input", f"{total_factors:,.0f}")

                # Visualisasi Perbandingan (Bar Chart)
                comparison_data = pd.DataFrame({
                    "Model": ["MLP (Neural Net)", "Random Forest"],
                    "Prediksi": [val_mlp, val_rf],
                    "Color": [COLOR_MLP, COLOR_RF]
                })

                fig_comp = px.bar(
                    comparison_data, x="Model", y="Prediksi", color="Model",
                    title="Perbandingan Hasil Prediksi Kedua Algoritma",
                    text_auto='.2s',
                    color_discrete_map={"MLP (Neural Net)": COLOR_MLP, "Random Forest": COLOR_RF}
                )
                fig_comp.update_layout(showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True)

                # Analisis Selisih
                diff = abs(val_mlp - val_rf)
                st.info(
                    f"💡 **Insight:** Selisih prediksi antara kedua model adalah **{diff:,.0f}** kasus. "
                    "Jika selisih kecil, kedua model memiliki konsensus yang kuat. "
                    "Random Forest biasanya lebih stabil pada data tabular, sedangkan MLP bisa menangkap pola non-linear yang lebih kompleks."
                )

            except Exception as e:
                st.error(f"Gagal melakukan prediksi. Error: {e}")

# --- HALAMAN 3: EVALUASI MODEL ---
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Performa Model")
    st.markdown("Halaman ini menunjukkan seberapa akurat kedua model berdasarkan data pengujian (tahun terakhir/testing).")

    # Kita lakukan simulasi evaluasi sederhana menggunakan data yang ada di CSV
    # Anggaplah data tahun terakhir (2024 atau max year) adalah data testing
    test_year = years_list[-1]
    df_test = df[df[YEAR_COL] == test_year].copy()
    
    if df_test.empty:
        st.warning("Tidak cukup data untuk evaluasi.")
    else:
        # Preprocessing Data Test
        X_test = preprocessor.transform(df_test[feature_cols])
        y_true = df_test[TARGET_COL].values

        # Prediksi Batch
        y_pred_mlp = mlp_model.predict(X_test).flatten()
        y_pred_rf = rf_model.predict(X_test)

        # Hitung Error (MAE & RMSE)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae_mlp = mean_absolute_error(y_true, y_pred_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_true, y_pred_mlp))
        
        mae_rf = mean_absolute_error(y_true, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, y_pred_rf))

        # Tampilkan Tabel Metrik
        st.subheader(f"📊 Metrik Error (Data Uji Tahun {test_year})")
        metrics_df = pd.DataFrame({
            "Model": ["MLP", "Random Forest"],
            "MAE (Mean Absolute Error)": [mae_mlp, mae_rf],
            "RMSE (Root Mean Squared Error)": [rmse_mlp, rmse_rf]
        })
        st.table(metrics_df.style.format("{:.2f}"))

        # Grafik Scatter: Actual vs Predicted
        st.subheader("🎯 Plot Sebaran: Nilai Aktual vs Prediksi")
        
        # Gabungkan hasil untuk plotting
        res_df = pd.DataFrame({
            "Region": df_test[REGION_COL],
            "Actual": y_true,
            "MLP Prediction": y_pred_mlp,
            "RF Prediction": y_pred_rf
        })

        # Plot Interaktif menggunakan Graph Objects agar bisa dual layer
        fig_scatter = go.Figure()
        
        # Garis referensi perfect prediction
        fig_scatter.add_trace(go.Scatter(
            x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
            mode='lines', name='Perfect Prediction', line=dict(color='gray', dash='dash')
        ))

        # Scatter MLP
        fig_scatter.add_trace(go.Scatter(
            x=res_df["Actual"], y=res_df["MLP Prediction"],
            mode='markers', name='MLP', marker=dict(color=COLOR_MLP, size=10, symbol='circle')
        ))

        # Scatter RF
        fig_scatter.add_trace(go.Scatter(
            x=res_df["Actual"], y=res_df["RF Prediction"],
            mode='markers', name='Random Forest', marker=dict(color=COLOR_RF, size=10, symbol='x')
        ))

        fig_scatter.update_layout(
            title="Aktual vs Prediksi (Semakin dekat ke garis putus-putus, semakin akurat)",
            xaxis_title="Jumlah Aktual",
            yaxis_title="Jumlah Prediksi",
            height=600
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("🔍 Detail Data Prediksi")
        st.dataframe(res_df)
