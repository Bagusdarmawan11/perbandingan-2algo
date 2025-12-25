import os
import json
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
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Sistem Prediksi Perceraian Jabar",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        .sidebar-copyright {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 244px;
            padding: 15px;
            text-align: center;
            background-color: #ffffff;
            font-size: 12px;
            color: #444;
            border-top: 1px solid #e0e0e0;
            z-index: 1000;
            font-family: sans-serif;
        }
        
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .insight-box {
            background-color: #e8f4f8;
            border-left: 5px solid #457B9D;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 20px;
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
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson" # File Peta (Opsional)

MODEL_MLP_FILE = MODELS_DIR / "model_mlp.h5"
MODEL_RF_FILE = MODELS_DIR / "model_rf.joblib"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Palet Warna
COLOR_MLP = "#E63946"
COLOR_RF = "#457B9D"
COLOR_ACTUAL = "#2A9D8F"

# Koordinat Manual (Backup jika GeoJSON tidak ada)
JABAR_COORDS = {
    "KABUPATEN BOGOR": [-6.594, 106.789], "KABUPATEN SUKABUMI": [-6.921, 106.927],
    "KABUPATEN CIANJUR": [-6.817, 107.131], "KABUPATEN BANDUNG": [-7.025, 107.519],
    "KABUPATEN GARUT": [-7.202, 107.886], "KABUPATEN TASIKMALAYA": [-7.358, 108.106],
    "KABUPATEN CIAMIS": [-7.327, 108.354], "KABUPATEN KUNINGAN": [-6.976, 108.483],
    "KABUPATEN CIREBON": [-6.737, 108.549], "KABUPATEN MAJALENGKA": [-6.836, 108.227],
    "KABUPATEN SUMEDANG": [-6.858, 107.920], "KABUPATEN INDRAMAYU": [-6.327, 108.322],
    "KABUPATEN SUBANG": [-6.571, 107.760], "KABUPATEN PURWAKARTA": [-6.556, 107.444],
    "KABUPATEN KARAWANG": [-6.322, 107.306], "KABUPATEN BEKASI": [-6.241, 107.123],
    "KABUPATEN BANDUNG BARAT": [-6.843, 107.502], "KABUPATEN PANGANDARAN": [-7.696, 108.654],
    "KOTA BOGOR": [-6.597, 106.799], "KOTA SUKABUMI": [-6.927, 106.929],
    "KOTA BANDUNG": [-6.917, 107.619], "KOTA CIREBON": [-6.732, 108.552],
    "KOTA BEKASI": [-6.238, 106.975], "KOTA DEPOK": [-6.402, 106.794],
    "KOTA CIMAHI": [-6.873, 107.542], "KOTA TASIKMALAYA": [-7.327, 108.220],
    "KOTA BANJAR": [-7.374, 108.532]
}

# ==========================================
# 3. FUNGSI UTAMA (LOAD & CLEAN)
# ==========================================
@st.cache_data
def load_and_clean_data():
    """Memuat data dan membersihkan nama kolom."""
    if not DATA_FILE.exists():
        st.error(f"❌ File data tidak ditemukan di: {DATA_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_FILE)
    
    # --- LOGIKA CLEANING LABEL ---
    new_cols = []
    for col in df.columns:
        if col in [TARGET_COL, YEAR_COL, REGION_COL]:
            new_cols.append(col)
        else:
            # Hapus variasi kata panjang
            clean = col.replace("Faktor Penyebab Perceraian - ", "") \
                       .replace("Faktor Penyebab Perceraian ", "") \
                       .replace("Faktor Perceraian - ", "") \
                       .replace("Faktor Perceraian ", "") \
                       .replace("Faktor Penyebab - ", "") \
                       .replace("Faktor Penyebab ", "") \
                       .replace("Penyebab Perceraian ", "") \
                       .replace("Faktor ", "") \
                       .replace("Penyebab ", "") \
                       .strip()
            
            if " - " in clean:
                clean = clean.split(" - ")[-1]
            
            new_cols.append(clean.strip("- "))
    
    df.columns = new_cols
    # Upper case region agar konsisten
    df[REGION_COL] = df[REGION_COL].str.upper()
    return df

@st.cache_data
def load_geojson_data():
    """Memuat file GeoJSON jika ada."""
    if GEOJSON_FILE.exists():
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_system_artifacts(df: pd.DataFrame):
    try:
        # Identifikasi ulang kolom dari DF yang SUDAH DIBERSIHKAN
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c != TARGET_COL]
        
        categorical_cols = [REGION_COL]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )
        preprocessor.fit(df[feature_cols])

        mlp_model = load_model(MODEL_MLP_FILE, compile=False) if MODEL_MLP_FILE.exists() else None
        rf_model = joblib.load(MODEL_RF_FILE) if MODEL_RF_FILE.exists() else None
        
        # Kembalikan juga daftar kolom faktor yang valid
        valid_factor_cols = [c for c in numeric_cols if c != YEAR_COL]
        
        return preprocessor, mlp_model, rf_model, valid_factor_cols

    except Exception as e:
        st.error(f"❌ Gagal memuat sistem AI: {e}")
        return None, None, None, []

# --- EKSEKUSI ---
df = load_and_clean_data()

if df.empty:
    st.stop()

# Load Artifacts & Dapatkan kolom faktor yang SUDAH BERSIH
preprocessor, mlp_model, rf_model, factor_cols = load_system_artifacts(df)

years_list = sorted(df[YEAR_COL].unique())
regions_list = sorted(df[REGION_COL].unique())

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🎛️ Navigasi")
    
    page = st.radio("Menu:", [
        "📊 Dashboard Data", 
        "📈 Eksplorasi Daerah & Faktor", 
        "🗺️ Peta Jawa Barat", 
        "🔮 Prediksi & Perbandingan", 
        "📈 Evaluasi Model"
    ])
    
    st.markdown("---")
    
    if page == "📊 Dashboard Data":
        selected_region = st.selectbox("Wilayah:", ["(Semua)"] + regions_list)
        selected_year = st.selectbox("Tahun:", years_list, index=len(years_list)-1)
    
    elif page == "📈 Eksplorasi Daerah & Faktor":
        exp_year = st.selectbox("Pilih Tahun:", years_list, index=len(years_list)-1)

    elif page == "🗺️ Peta Jawa Barat":
        map_year = st.selectbox("Pilih Tahun Peta:", years_list, index=len(years_list)-1)

    st.markdown("---")
    if st.button("🔄 Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()
        
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
# 5. HALAMAN UTAMA
# ==========================================

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard Data":
    st.title("📊 Dashboard Analisis")
    
    df_view = df.copy()
    if selected_region != "(Semua)":
        df_view = df_view[df_view[REGION_COL] == selected_region]
    
    total = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].sum()
    avg = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].mean()
    mx = df_view[df_view[YEAR_COL] == selected_year][TARGET_COL].max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Kasus", f"{total:,.0f}")
    c2.metric("Rata-rata", f"{avg:,.0f}")
    c3.metric("Tertinggi", f"{mx:,.0f}")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📈 Grafik", "📄 Data"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tren Tahunan")
            trend = df_view.groupby(YEAR_COL)[TARGET_COL].sum().reset_index()
            fig = px.line(trend, x=YEAR_COL, y=TARGET_COL, markers=True, color_discrete_sequence=[COLOR_ACTUAL])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Penyebab Utama")
            # Pastikan hanya menjumlahkan kolom faktor yang valid
            f_sum = df_view[df_view[YEAR_COL] == selected_year][factor_cols].sum().reset_index()
            f_sum.columns = ["Faktor", "Jumlah"]
            f_sum = f_sum.sort_values("Jumlah", ascending=True).tail(10)
            
            top_f = f_sum.iloc[-1]['Faktor'] if not f_sum.empty else "-"
            top_v = f_sum.iloc[-1]['Jumlah'] if not f_sum.empty else 0
            
            fig2 = px.bar(f_sum, x="Jumlah", y="Faktor", orientation='h', color="Jumlah", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)
            
    with tab2:
        st.dataframe(df_view, use_container_width=True)
        
    st.markdown(f"<div class='insight-box'><h4>💡 Insight</h4>Faktor tertinggi tahun {selected_year} adalah <b>{top_f}</b> ({top_v:,.0f} kasus).</div>", unsafe_allow_html=True)


# --- PAGE 2: EKSPLORASI DAERAH (FIXED MELT) ---
elif page == "📈 Eksplorasi Daerah & Faktor":
    st.title("📈 Eksplorasi Daerah")
    
    df_exp = df[df[YEAR_COL] == exp_year].copy()
    
    # 1. Bar Chart Warna-warni
    st.subheader(f"Total Kasus per Wilayah ({exp_year})")
    df_sorted = df_exp.sort_values(TARGET_COL, ascending=False)
    
    fig_reg = px.bar(df_sorted, x=REGION_COL, y=TARGET_COL, color=REGION_COL, 
                     title="Perbandingan Total Kasus", text_auto='.2s')
    fig_reg.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig_reg, use_container_width=True)

    # 2. Treemap (FIXED)
    st.subheader("Komposisi Faktor")
    
    # Kunci perbaikan: Gunakan 'factor_cols' yang sudah diverifikasi ada di dataframe
    valid_cols = [c for c in factor_cols if c in df_exp.columns]
    
    if valid_cols:
        try:
            melted = df_exp.melt(id_vars=[REGION_COL], value_vars=valid_cols, 
                                 var_name="Faktor", value_name="Jumlah")
            melted = melted[melted["Jumlah"] > 0]
            
            fig_tree = px.treemap(melted, path=[REGION_COL, "Faktor"], values="Jumlah", color=REGION_COL)
            st.plotly_chart(fig_tree, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal visualisasi faktor: {e}")
    else:
        st.warning("Tidak ada kolom faktor yang valid untuk divisualisasikan.")


# --- PAGE 3: PETA JAWA BARAT (AUTO DETECT GEOJSON) ---
elif page == "🗺️ Peta Jawa Barat":
    st.title("🗺️ Peta Sebaran")
    st.markdown(f"Visualisasi Geografis Data Tahun **{map_year}**")
    
    df_map = df[df[YEAR_COL] == map_year].copy()
    geojson_data = load_geojson_data()

    # PILIHAN 1: Jika file GeoJSON ada (Tampilan Peta Wilayah/Choropleth)
    if geojson_data:
        try:
            # Gunakan Choropleth sesuai referensi kode lama kamu
            fig_map = px.choropleth(
                df_map,
                geojson=geojson_data,
                locations=REGION_COL,
                # Pastikan 'properties.NAME_2' sesuai dengan file GeoJSON kamu
                # Jika error, coba ganti 'properties.KAB_KOTA' atau 'properties.NAMOBJ'
                featureidkey="properties.NAME_2", 
                color=TARGET_COL,
                color_continuous_scale="Reds",
                hover_name=REGION_COL,
                title=f"Peta Sebaran Wilayah {map_year}"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            st.success("✅ Menggunakan Peta GeoJSON.")
        except Exception as e:
            st.error(f"Gagal memuat GeoJSON: {e}. Beralih ke mode koordinat.")
            geojson_data = None # Paksa fallback ke scatter

    # PILIHAN 2: Jika GeoJSON tidak ada atau Gagal (Tampilan Bubble Map)
    if not geojson_data:
        lats, lons = [], []
        for reg in df_map[REGION_COL]:
            coords = JABAR_COORDS.get(reg.strip().upper(), [-6.9, 107.6])
            lats.append(coords[0])
            lons.append(coords[1])
            
        df_map['lat'] = lats
        df_map['lon'] = lons
        
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", size=TARGET_COL, color=TARGET_COL,
            hover_name=REGION_COL, color_continuous_scale="Reds", size_max=40, zoom=7.5,
            mapbox_style="carto-positron", title=f"Peta Panas Perceraian {map_year}"
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.info("ℹ️ Menggunakan Peta Koordinat (File GeoJSON tidak ditemukan). Upload file .geojson ke folder data/ jika ingin tampilan wilayah.")

    st.markdown("<div class='insight-box'><h4>💡 Info Peta</h4>Warna merah pekat menandakan daerah dengan kasus tertinggi.</div>", unsafe_allow_html=True)


# --- PAGE 4: PREDIKSI (MULTI-SELECT STYLE) ---
elif page == "🔮 Prediksi & Perbandingan":
    st.title("🔮 Prediksi & Komparasi")
    st.markdown("Simulasikan prediksi dengan mengubah faktor penyebab tertentu.")

    # 1. FORM UTAMA
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Wilayah & Waktu")
            inp_reg = st.selectbox("Pilih Wilayah:", regions_list)
            inp_yr = st.number_input("Tahun Prediksi:", 2000, 2030, 2025)
        
        with c2:
            st.markdown("#### 2. Ubah Faktor Penyebab")
            st.caption("Pilih faktor yang ingin diubah. Faktor yang **TIDAK** dipilih akan bernilai 0.")
            
            # MULTISELECT (Sesuai Referensi)
            selected_factors = st.multiselect("Pilih Faktor:", factor_cols)
            
            # Input Dinamis hanya untuk faktor yang dipilih
            inputs = {}
            if selected_factors:
                for f in selected_factors:
                    # Ambil nilai rata-rata dari data asli sebagai saran (optional)
                    inputs[f] = st.number_input(f"Nilai {f}:", min_value=0, value=0)
            
        st.markdown("---")
        submit = st.form_submit_button("🚀 HITUNG PREDIKSI")

    # 2. HASIL PREDIKSI
    if submit:
        if not mlp_model or not rf_model:
            st.error("Model belum dimuat.")
        else:
            # Build Data Row
            row = {YEAR_COL: inp_yr, REGION_COL: inp_reg}
            
            # Logika Pengisian Nilai Faktor:
            # - Jika dipilih di multiselect -> Pakai nilai input user
            # - Jika TIDAK dipilih -> 0
            for f in factor_cols:
                if f in inputs:
                    row[f] = inputs[f]
                else:
                    row[f] = 0 # Default 0
                
            df_in = pd.DataFrame([row])
            # Pastikan urutan kolom sesuai feature_cols saat training
            # feature_cols sudah didefinisikan di load_system_artifacts (selain target)
            final_features = [c for c in df.columns if c != TARGET_COL]
            df_in = df_in[final_features]
            
            try:
                # Preprocessing
                X_in = preprocessor.transform(df_in)
                
                # Prediksi MLP (Flatten karena outputnya 2D array)
                p_mlp = float(mlp_model.predict(X_in).flatten()[0])
                
                # Prediksi RF
                p_rf = float(rf_model.predict(X_in)[0])
                
                # TAMPILAN HASIL
                st.subheader("🎯 Hasil Prediksi")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("MLP (Neural Net)", f"{p_mlp:,.0f}", delta="Deep Learning")
                k2.metric("Random Forest", f"{p_rf:,.0f}", delta="Ensemble")
                diff = abs(p_mlp - p_rf)
                k3.metric("Selisih", f"{diff:,.0f}", delta_color="inverse")
                
                # Visualisasi Bar Chart
                res = pd.DataFrame({
                    "Model": ["MLP", "Random Forest"], 
                    "Prediksi": [p_mlp, p_rf],
                    "Color": [COLOR_MLP, COLOR_RF]
                })
                fig = px.bar(res, x="Model", y="Prediksi", color="Model", text_auto='.2s',
                             color_discrete_map={"MLP": COLOR_MLP, "Random Forest": COLOR_RF})
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                high = "MLP" if p_mlp > p_rf else "Random Forest"
                st.markdown(f"""
                <div class="insight-box">
                    <h4>💡 Kesimpulan Prediksi</h4>
                    Model <b>{high}</b> memprediksi angka lebih tinggi. Selisih antara kedua model adalah {diff:,.0f}.
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")


# --- PAGE 5: EVALUASI ---
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi")
    
    test_yr = years_list[-1]
    df_test = df[df[YEAR_COL] == test_yr].copy()
    
    if df_test.empty:
        st.warning("Data kosong.")
    else:
        # Gunakan kolom fitur yang valid
        final_features = [c for c in df.columns if c != TARGET_COL]
        X_t = preprocessor.transform(df_test[final_features])
        y_t = df_test[TARGET_COL].values
        
        p_mlp = mlp_model.predict(X_t).flatten()
        p_rf = rf_model.predict(X_t)
        
        mae_mlp = mean_absolute_error(y_t, p_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_t, p_mlp))
        mae_rf = mean_absolute_error(y_t, p_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_t, p_rf))
        
        st.subheader(f"Metrik Error ({test_yr})")
        met = pd.DataFrame({
            "Model": ["MLP", "RF"],
            "MAE": [mae_mlp, mae_rf],
            "RMSE": [rmse_mlp, rmse_rf]
        })
        st.table(met.set_index("Model").round(2))
        
        st.subheader("Aktual vs Prediksi")
        fig = go.Figure()
        lim = [y_t.min(), y_t.max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', name='Perfect', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=y_t, y=p_mlp, mode='markers', name='MLP', marker=dict(color=COLOR_MLP, opacity=0.6)))
        fig.add_trace(go.Scatter(x=y_t, y=p_rf, mode='markers', name='RF', marker=dict(color=COLOR_RF, symbol='x')))
        st.plotly_chart(fig, use_container_width=True)
        
        best = "MLP" if mae_mlp < mae_rf else "RF"
        st.markdown(f"<div class='insight-box'><h4>💡 Kesimpulan</h4>Model <b>{best}</b> lebih akurat (MAE Terkecil: {min(mae_mlp, mae_rf):.2f}).</div>", unsafe_allow_html=True)
