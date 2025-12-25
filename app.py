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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS PRO
# ==========================================
st.set_page_config(
    page_title="Sistem Prediksi Perceraian Jabar",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom: Menyembunyikan elemen bawaan & Styling Copyright
st.markdown("""
    <style>
        /* Sembunyikan Menu Hamburger & Footer Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Styling Sidebar agar lebih rapi */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Styling Copyright di Bawah Sidebar (Sticky) */
        .sidebar-copyright {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 244px; /* Lebar default sidebar */
            padding: 15px;
            text-align: center;
            background-color: #ffffff;
            font-size: 12px;
            color: #444;
            border-top: 1px solid #e0e0e0;
            z-index: 1000;
            font-family: sans-serif;
        }
        
        /* Styling Judul Halaman */
        h1 {
            color: #2c3e50;
            font-family: 'Helvetica', sans-serif;
        }
        
        /* Kartu Metrik Custom */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Styling Insight Box */
        .insight-box {
            background-color: #e8f4f8;
            border-left: 5px solid #457B9D;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETUP PATH & KONSTANTA
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
MODEL_MLP_FILE = MODELS_DIR / "model_mlp.h5"
MODEL_RF_FILE = MODELS_DIR / "model_rf.joblib"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Palet Warna Konsisten
COLOR_MLP = "#E63946"     # Merah Cerah
COLOR_RF = "#457B9D"      # Biru Kalem
COLOR_ACTUAL = "#2A9D8F"  # Hijau Tosca
COLOR_WARN = "#F4A261"    # Oranye

# ==========================================
# 3. FUNGSI UTAMA (LOAD & CLEAN)
# ==========================================
@st.cache_data
def load_and_clean_data():
    """Memuat data dan membersihkan nama kolom yang panjang."""
    if not DATA_FILE.exists():
        st.error(f"❌ File data tidak ditemukan di: {DATA_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_FILE)
    
    # --- LOGIKA CLEANING NAMA KOLOM ---
    new_cols = []
    for col in df.columns:
        if col in [TARGET_COL, YEAR_COL, REGION_COL]:
            new_cols.append(col)
        else:
            # Hapus kata-kata yang tidak perlu
            clean = col.replace("Faktor Penyebab - ", "") \
                       .replace("Faktor Perceraian - ", "") \
                       .replace("Faktor Penyebab ", "") \
                       .replace("Faktor ", "") \
                       .replace("Penyebab ", "") \
                       .strip()
            new_cols.append(clean)
    
    df.columns = new_cols
    return df

@st.cache_resource
def load_system_artifacts(df: pd.DataFrame):
    """
    Membangun ulang preprocessor dari data bersih & memuat model.
    """
    try:
        # 1. Identifikasi Kolom (Dari data yang sudah bersih namanya)
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c != TARGET_COL]
        
        categorical_cols = [REGION_COL]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        # 2. Preprocessor (Scaler & Encoder)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )
        preprocessor.fit(df[feature_cols])

        # 3. Load Model
        mlp_model = load_model(MODEL_MLP_FILE, compile=False) if MODEL_MLP_FILE.exists() else None
        rf_model = joblib.load(MODEL_RF_FILE) if MODEL_RF_FILE.exists() else None
        
        if not mlp_model or not rf_model:
            st.warning("⚠️ Salah satu model (MLP/RF) tidak ditemukan di folder models/.")

        return preprocessor, mlp_model, rf_model, feature_cols, numeric_cols

    except Exception as e:
        st.error(f"❌ Gagal memuat sistem AI: {e}")
        return None, None, None, [], []

# --- EKSEKUSI DATA LOADING ---
df = load_and_clean_data()

if df.empty:
    st.stop() # Hentikan jika data kosong

preprocessor, mlp_model, rf_model, feature_cols, numeric_cols = load_system_artifacts(df)

# List Helper
factor_cols = [c for c in numeric_cols if c != YEAR_COL]
years_list = sorted(df[YEAR_COL].unique())
regions_list = sorted(df[REGION_COL].unique())

# ==========================================
# 4. SIDEBAR (NAVIGASI & INFO)
# ==========================================
with st.sidebar:
    st.title("🎛️ Navigasi Sistem")
    
    # Menu Navigasi
    page = st.radio("Pilih Menu:", 
                    ["📊 Dashboard Data", "🔮 Prediksi & Perbandingan", "📈 Evaluasi Model"])
    
    st.markdown("---")
    
    # Filter Global (Hanya aktif di dashboard)
    if page == "📊 Dashboard Data":
        st.subheader("🔍 Filter Data")
        selected_region = st.selectbox("Wilayah:", ["(Semua)"] + regions_list)
        selected_year = st.selectbox("Tahun:", years_list, index=len(years_list)-1)
    
    st.markdown("---")
    
    # Tombol Reset Cache
    if st.button("🔄 Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()
        
    st.info(
        "**Info Model:**\n"
        "1. **MLP (Neural Network)**: Deep Learning.\n"
        "2. **Random Forest**: Ensemble Learning."
    )
    
    # COPYRIGHT FOOTER (HTML)
    st.sidebar.markdown(
        """
        <div class='sidebar-copyright'>
            <b>Copyright © 2025</b><br>
            Developed By:<br>
            <b>Milda Nabilah Al-hamaz</b><br>
            NPM: 202210715059
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 5. KONTEN HALAMAN
# ==========================================

# --- PAGE 1: DASHBOARD DATA ---
if page == "📊 Dashboard Data":
    st.title("📊 Dashboard Analisis Perceraian")
    st.markdown("Monitoring data historis perceraian di Jawa Barat.")
    
    # Filter Dataframe
    df_view = df.copy()
    if selected_region != "(Semua)":
        df_view = df_view[df_view[REGION_COL] == selected_region]
    
    # Statistik Utama
    total_kasus = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].sum()
    avg_kasus = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].mean()
    max_kasus = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].max()
    
    # Baris Metrik
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Total Kasus ({selected_year})", f"{total_kasus:,.0f} Kasus")
    m2.metric(f"Rata-rata Wilayah", f"{avg_kasus:,.0f} Kasus")
    m3.metric(f"Rekor Tertinggi", f"{max_kasus:,.0f} Kasus")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["📈 Visualisasi Grafik", "📄 Data Mentah"])
    
    with tab1:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Tren Kasus Perceraian")
            # Agregasi per tahun
            trend = df_view.groupby(YEAR_COL)[TARGET_COL].sum().reset_index()
            fig_trend = px.line(trend, x=YEAR_COL, y=TARGET_COL, markers=True,
                                title=f"Tren Tahunan ({selected_region})",
                                color_discrete_sequence=[COLOR_ACTUAL], template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_g2:
            st.subheader(f"Top Penyebab ({selected_year})")
            # Hitung total per faktor
            factors_sum = df_view[df_view[YEAR_COL] == selected_year][factor_cols].sum()
            df_factors = factors_sum.reset_index()
            df_factors.columns = ["Faktor", "Jumlah"]
            df_factors = df_factors.sort_values("Jumlah", ascending=True).tail(10)
            
            # Ambil faktor dominan untuk insight
            top_factor_name = df_factors.iloc[-1]['Faktor'] if not df_factors.empty else "-"
            top_factor_val = df_factors.iloc[-1]['Jumlah'] if not df_factors.empty else 0
            
            fig_bar = px.bar(df_factors, x="Jumlah", y="Faktor", orientation='h',
                             text_auto='.2s', color="Jumlah", 
                             color_continuous_scale="Reds", template="plotly_white")
            fig_bar.update_layout(yaxis_title=None)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with tab2:
        st.subheader("Dataset Lengkap")
        st.dataframe(df_view, use_container_width=True)

    # --- INSIGHT BOX (DASHBOARD) ---
    st.markdown(f"""
    <div class="insight-box">
        <h4>💡 Insight Dashboard</h4>
        <ul>
            <li>Pada tahun <b>{selected_year}</b>, tercatat total <b>{total_kasus:,.0f}</b> kasus perceraian di wilayah terpilih.</li>
            <li>Faktor penyebab paling dominan adalah <b>"{top_factor_name}"</b> dengan total <b>{top_factor_val:,.0f}</b> kasus.</li>
            <li>Analisis ini membantu pemerintah untuk memprioritaskan penanganan pada faktor penyebab utama.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# --- PAGE 2: PREDIKSI & PERBANDINGAN ---
elif page == "🔮 Prediksi & Perbandingan":
    st.title("🔮 Prediksi Interaktif & Komparasi")
    st.markdown("Prediksi jumlah perceraian menggunakan **MLP vs Random Forest**.")

    # 1. Konfigurasi Wilayah
    st.subheader("1. Konfigurasi Wilayah & Waktu")
    c1, c2 = st.columns(2)
    with c1:
        inp_region = st.selectbox("Pilih Kabupaten/Kota:", regions_list)
    with c2:
        inp_year = st.number_input("Tahun Prediksi:", 2000, 2030, 2025)

    # --- LOGIKA SESSION STATE UNTUK INPUT FAKTOR ---
    # Ini penting agar nilai tidak reset saat ganti dropdown
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {col: 0 for col in factor_cols}
    if 'last_region' not in st.session_state:
        st.session_state['last_region'] = None

    # Jika ganti wilayah, auto-fill ulang (sekali saja)
    if st.session_state['last_region'] != inp_region:
        hist_data = df[df[REGION_COL] == inp_region]
        if not hist_data.empty:
            defaults = hist_data[factor_cols].mean().fillna(0).astype(int).to_dict()
            st.session_state['input_data'] = defaults
            st.toast(f"Data otomatis diisi rata-rata {inp_region}", icon="✅")
        else:
            st.session_state['input_data'] = {col: 0 for col in factor_cols}
        st.session_state['last_region'] = inp_region

    # 2. Input Faktor (DROPDOWN MODE) 
    st.markdown("### 2. Parameter Faktor Penyebab")
    st.caption("Pilih faktor dari dropdown di bawah, lalu ubah angkanya.")
    
    with st.container(border=True):
        # Dropdown untuk memilih nama faktor
        selected_factor = st.selectbox("👇 Pilih Faktor yang ingin diubah:", factor_cols)
        
        # Input angka untuk faktor yang dipilih
        current_val = st.session_state['input_data'][selected_factor]
        new_val = st.number_input(f"Masukkan Jumlah Kasus Akibat '{selected_factor}':", 
                                  min_value=0, value=int(current_val))
        
        # Simpan perubahan ke session state
        st.session_state['input_data'][selected_factor] = new_val
        
        # Tampilkan ringkasan data yang akan diprediksi
        with st.expander("📄 Lihat Ringkasan Semua Data Input"):
            st.json(st.session_state['input_data'])

    # Tombol Eksekusi
    if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
        if not mlp_model or not rf_model:
            st.error("❌ Model AI belum dimuat. Cek folder 'models/'.")
        else:
            # Siapkan data input dari Session State
            row_data = {YEAR_COL: inp_year, REGION_COL: inp_region}
            row_data.update(st.session_state['input_data'])
            
            df_pred = pd.DataFrame([row_data])
            df_pred = df_pred[feature_cols]

            try:
                # Preprocessing
                X_pred = preprocessor.transform(df_pred)
                
                # Prediksi
                p_mlp = mlp_model.predict(X_pred)
                val_mlp = float(p_mlp[0][0]) if p_mlp.ndim > 1 else float(p_mlp[0])
                
                p_rf = rf_model.predict(X_pred)
                val_rf = float(p_rf[0])
                
                st.divider()
                st.subheader("🎯 Hasil Prediksi AI")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("MLP (Neural Network)", f"{val_mlp:,.0f}", delta="Deep Learning")
                k2.metric("Random Forest", f"{val_rf:,.0f}", delta="Ensemble")
                diff = abs(val_mlp - val_rf)
                k3.metric("Selisih Prediksi", f"{diff:,.0f}", delta_color="inverse")
                
                # Chart
                comp_data = pd.DataFrame({
                    "Model AI": ["MLP (Neural Net)", "Random Forest"],
                    "Prediksi Jumlah": [val_mlp, val_rf],
                    "Warna": [COLOR_MLP, COLOR_RF]
                })
                
                fig_comp = px.bar(comp_data, x="Model AI", y="Prediksi Jumlah", color="Model AI",
                                  title="Komparasi Hasil Prediksi", text_auto='.2s',
                                  color_discrete_map={"MLP (Neural Net)": COLOR_MLP, "Random Forest": COLOR_RF})
                st.plotly_chart(fig_comp, use_container_width=True)

                # --- INSIGHT BOX (PREDIKSI) ---
                higher_model = "MLP" if val_mlp > val_rf else "Random Forest"
                st.markdown(f"""
                <div class="insight-box">
                    <h4>💡 Kesimpulan & Insight Prediksi</h4>
                    <ul>
                        <li><b>Perbandingan Model:</b> Terdapat selisih sebesar <b>{diff:,.0f}</b> kasus antara kedua algoritma.</li>
                        <li><b>Kecenderungan:</b> Model <b>{higher_model}</b> memberikan prediksi yang lebih tinggi.</li>
                        <li><b>Rekomendasi:</b> Jika selisih kecil (<10%), hasil prediksi sangat meyakinkan (konsensus kuat). Jika besar, pertimbangkan menggunakan nilai rata-rata kedua model sebagai acuan konservatif.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                 

[Image of Random Forest algorithm diagram]


            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")


# --- PAGE 3: EVALUASI MODEL ---
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Kinerja Model")
    st.markdown("Evaluasi akurasi menggunakan data tahun terakhir (Testing Data).")
    
    test_yr = years_list[-1]
    df_test = df[df[YEAR_COL] == test_yr].copy()
    
    if df_test.empty:
        st.warning("Data tidak cukup untuk evaluasi.")
    else:
        # Proses Evaluasi
        X_test = preprocessor.transform(df_test[feature_cols])
        y_true = df_test[TARGET_COL].values
        
        y_mlp = mlp_model.predict(X_test).flatten()
        y_rf = rf_model.predict(X_test)
        
        # Hitung Error
        mae_mlp = mean_absolute_error(y_true, y_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_true, y_mlp))
        
        mae_rf = mean_absolute_error(y_true, y_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, y_rf))
        
        st.subheader(f"📊 Tabel Error (Data Tahun {test_yr})")
        
        metrics = pd.DataFrame({
            "Model": ["MLP (Neural Network)", "Random Forest"],
            "MAE (Mean Absolute Error)": [mae_mlp, mae_rf],
            "RMSE (Root Mean Squared Error)": [rmse_mlp, rmse_rf]
        })
        st.table(metrics.set_index("Model").round(2))
        
        # Plot Scatter
        st.subheader("🎯 Validasi Visual: Aktual vs Prediksi")
        fig_sc = go.Figure()
        min_v, max_v = y_true.min(), y_true.max()
        fig_sc.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], 
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(color='gray', dash='dash')))
        fig_sc.add_trace(go.Scatter(x=y_true, y=y_mlp, mode='markers', name='MLP',
                                    marker=dict(color=COLOR_MLP, size=10, opacity=0.7)))
        fig_sc.add_trace(go.Scatter(x=y_true, y=y_rf, mode='markers', name='Random Forest',
                                    marker=dict(color=COLOR_RF, size=10, symbol='x')))
        fig_sc.update_layout(xaxis_title="Jumlah Aktual", yaxis_title="Prediksi", height=500)
        st.plotly_chart(fig_sc, use_container_width=True)
        
        # --- INSIGHT BOX (EVALUASI) ---
        best_model_mae = "MLP" if mae_mlp < mae_rf else "Random Forest"
        best_val_mae = min(mae_mlp, mae_rf)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 Kesimpulan Evaluasi</h4>
            <ul>
                <li><b>Model Terbaik (MAE):</b> Berdasarkan Mean Absolute Error, model <b>{best_model_mae}</b> lebih akurat dengan rata-rata kesalahan prediksi hanya <b>{best_val_mae:.2f}</b> kasus.</li>
                <li><b>Interpretasi Grafik:</b> Titik-titik yang semakin dekat dengan garis putus-putus menunjukkan prediksi yang sangat akurat.</li>
                <li><b>Saran:</b> Gunakan model <b>{best_model_mae}</b> sebagai acuan utama dalam pengambilan keputusan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
