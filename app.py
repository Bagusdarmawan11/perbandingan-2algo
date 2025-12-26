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
        
        /* Styling Tabs agar lebih terlihat */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px; 
            margin-top: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px; 
            white-space: pre-wrap; 
            background-color: #f8f9fa; 
            border-radius: 5px 5px 0 0;
            border: 1px solid #ddd;
            border-bottom: none;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #ffffff; 
            border-top: 3px solid #E63946;
            font-weight: bold;
            color: #E63946;
        }
        
        /* Metric Box Styling */
        div[data-testid="stMetric"] {
            background-color: #ffffff; 
            border: 1px solid #e0e0e0; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        /* Insight Box Fixed */
        .insight-box {
            background-color: #f0f7fb;
            border-left: 5px solid #457B9D;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #1d3557;
            font-family: 'Source Sans Pro', sans-serif;
        }
        .insight-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: #1d3557;
            display: block;
        }
        .insight-content {
            font-size: 1rem;
            line-height: 1.6;
        }
        
        /* Sidebar Copyright */
        .sidebar-copyright {
            position: fixed; bottom: 0; left: 0; width: 244px;
            padding: 15px; text-align: center; background-color: #ffffff;
            font-size: 12px; color: #666; border-top: 1px solid #e0e0e0;
            z-index: 1000;
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
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"

MODEL_MLP_FILE = MODELS_DIR / "model_mlp.h5"
MODEL_RF_FILE = MODELS_DIR / "model_rf.joblib"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Palet Warna
COLOR_MLP = "#E63946"     # Merah
COLOR_RF = "#457B9D"      # Biru
COLOR_ACTUAL = "#2A9D8F"  # Hijau

# Koordinat Manual (Backup)
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
        st.stop()
    
    df = pd.read_csv(DATA_FILE)
    
    # --- CLEANING LABEL ---
    new_cols = []
    for col in df.columns:
        if col in [TARGET_COL, YEAR_COL, REGION_COL]:
            new_cols.append(col)
        else:
            # Hapus variasi kata panjang & Typo
            clean = col.replace("Faktor Penyebab Perceraian", "") \
                       .replace("Faktor Perceraian", "") \
                       .replace("Fakor Perceraian", "") \
                       .replace("Faktor Penyebab", "") \
                       .replace("Penyebab Perceraian", "") \
                       .replace("Penyebab", "") \
                       .replace("Faktor", "") \
                       .replace("Fakor", "") \
                       .replace("Nilai", "") \
                       .replace("-", "") \
                       .strip()
            
            # Jika masih ada sisa " - "
            if " - " in clean:
                clean = clean.split(" - ")[-1]
            
            new_cols.append(clean.strip())
            
    # Deduplikasi nama kolom (Fix ValueError: Melt)
    final_cols = []
    seen = {}
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 0
            final_cols.append(c)
            
    df.columns = final_cols
    # Tidak di-upper case agar match dengan GeoJSON (Title Case)
    df[REGION_COL] = df[REGION_COL].str.strip()
    return df

@st.cache_data
def load_geojson():
    """Memuat file GeoJSON."""
    if not GEOJSON_FILE.exists():
        return None
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return geojson

@st.cache_resource
def load_artifacts(df: pd.DataFrame):
    """Memuat Preprocessor dan Model."""
    try:
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

        factor_cols = [c for c in numeric_cols if c != YEAR_COL]

        return preprocessor, mlp_model, rf_model, feature_cols, factor_cols

    except Exception as e:
        st.error(f"❌ Gagal memuat sistem AI: {e}")
        st.stop()

# --- EKSEKUSI DATA ---
df = load_and_clean_data()
preprocessor, mlp_model, rf_model, feature_cols, factor_cols = load_artifacts(df)

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# ==========================================
# 4. HEADER & SIDEBAR (FILTER ONLY)
# ==========================================
st.title("📊 Prediksi Perceraian Provinsi Jawa Barat")
st.caption("Platform analisis tren dan prediksi menggunakan **Multi-Layer Perceptron (MLP)** dan **Random Forest**.")

# SIDEBAR: HANYA FILTER & COPYRIGHT (MENU DIHAPUS)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=50)
    st.header("🎛️ Filter Global")
    
    # Filter Global Tahun (Mempengaruhi Dashboard, Eksplorasi, Peta)
    selected_year = st.selectbox(
        "Pilih Tahun Analisis:",
        options=years,
        index=len(years) - 1
    )
    
    # Filter Global Wilayah (Mempengaruhi Dashboard & Tabel)
    selected_region = st.selectbox(
        "Pilih Wilayah (Opsional):",
        options=["(Semua)"] + regions
    )
    
    st.markdown("---")
    if st.button("🔄 Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown(
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
# 5. TABS NAVIGASI UTAMA (MENU DI ATAS)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 Eksplorasi", 
    "🗺️ Peta Sebaran", 
    "🔮 Prediksi AI", 
    "📉 Evaluasi Model"
])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
with tab1:
    st.subheader(f"Ringkasan Data Tahun {selected_year}")
    
    df_dash = df.copy()
    if selected_region != "(Semua)":
        df_dash = df_dash[df_dash[REGION_COL] == selected_region]
    
    # Metrik
    total = df_dash[df_dash[YEAR_COL] == selected_year][TARGET_COL].sum()
    avg = df_dash[df_dash[YEAR_COL] == selected_year][TARGET_COL].mean()
    mx = df_dash[df_dash[YEAR_COL] == selected_year][TARGET_COL].max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Kasus", f"{total:,.0f}")
    c2.metric("Rata-rata Kasus", f"{avg:,.0f}")
    c3.metric("Kasus Tertinggi (Wilayah)", f"{mx:,.0f}")
    
    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("##### 📈 Tren Tahunan")
        trend = df_dash.groupby(YEAR_COL)[TARGET_COL].sum().reset_index()
        fig_trend = px.line(trend, x=YEAR_COL, y=TARGET_COL, markers=True, color_discrete_sequence=[COLOR_ACTUAL])
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_g2:
        st.markdown(f"##### 🧩 Top 5 Faktor Penyebab ({selected_year})")
        # Hitung faktor
        f_sum = df_dash[df_dash[YEAR_COL] == selected_year][factor_cols].sum().reset_index()
        f_sum.columns = ["Faktor", "Jumlah"]
        f_sum = f_sum.sort_values("Jumlah", ascending=True).tail(10)
        
        fig_bar = px.bar(f_sum, x="Jumlah", y="Faktor", orientation='h', color="Jumlah", color_continuous_scale="Reds")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: EKSPLORASI (HORIZONTAL BAR)
# ==========================================
with tab2:
    st.subheader(f"Analisis Detail Tahun {selected_year}")
    df_year = df[df[YEAR_COL] == selected_year].copy()

    # 1. Grafik Daerah (Horizontal)
    st.markdown("#### 🔥 Peringkat Wilayah (Kasus Tertinggi)")
    df_year_sorted = df_year.sort_values(TARGET_COL, ascending=True)

    fig_region = px.bar(
        df_year_sorted,
        x=TARGET_COL, y=REGION_COL, orientation="h",
        text_auto='.2s', template="plotly_white",
        labels={REGION_COL: "", TARGET_COL: "Total Kasus"},
        color=REGION_COL
    )
    fig_region.update_layout(yaxis=dict(categoryorder="total ascending"), height=700, showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

    # 2. Grafik Faktor (Horizontal)
    st.markdown("#### 🧩 Kontribusi Faktor Penyebab")
    
    valid_factors = [c for c in factor_cols if c in df_year.columns]
    
    if valid_factors:
        factor_sum = df_year[valid_factors].sum().sort_values(ascending=True)
        factor_df = factor_sum.reset_index()
        factor_df.columns = ["Faktor", "Nilai"]

        fig_factor = px.bar(
            factor_df,
            x="Nilai", y="Faktor", orientation="h",
            text_auto='.2s', template="plotly_white", 
            color="Nilai", color_continuous_scale="Blues",
            labels={"Nilai": "Jumlah Kasus", "Faktor": ""}
        )
        fig_factor.update_layout(yaxis=dict(categoryorder="total ascending"), height=600)
        st.plotly_chart(fig_factor, use_container_width=True)
    
    # Insight
    top_reg = df_year_sorted.iloc[-1][REGION_COL]
    top_val = df_year_sorted.iloc[-1][TARGET_COL]
    st.info(f"💡 **Insight:** Wilayah dengan kasus tertinggi pada tahun {selected_year} adalah **{top_reg}** ({top_val:,.0f} kasus).")


# ==========================================
# TAB 3: PETA JAWA BARAT
# ==========================================
with tab3:
    st.subheader(f"Peta Sebaran Kasus ({selected_year})")
    df_map = df[df[YEAR_COL] == selected_year].copy()
    geojson = load_geojson()

    if geojson:
        try:
            fig_map = px.choropleth(
                df_map,
                geojson=geojson,
                locations=REGION_COL,
                featureidkey="properties.NAME_2", # Sesuaikan dengan GeoJSON
                color=TARGET_COL,
                color_continuous_scale="Reds",
                hover_name=REGION_COL,
                title=f"Intensitas Kasus Perceraian di Jawa Barat"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.warning("Gagal memuat peta wilayah. Menampilkan peta titik.")
            geojson = None

    if not geojson:
        lats, lons = [], []
        for reg in df_map[REGION_COL]:
            coords = JABAR_COORDS.get(reg.upper(), [-6.9, 107.6])
            lats.append(coords[0])
            lons.append(coords[1])
            
        df_map['lat'] = lats
        df_map['lon'] = lons
        
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", size=TARGET_COL, color=TARGET_COL,
            hover_name=REGION_COL, color_continuous_scale="Reds", size_max=45, zoom=7.5,
            mapbox_style="carto-positron"
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)


# ==========================================
# TAB 4: PREDIKSI (MULTISELECT)
# ==========================================
with tab4:
    st.subheader("🔮 Simulasi Prediksi")
    st.info("Pilih wilayah dan faktor penyebab. Faktor yang dipilih akan diisi nilai **Median**, sisanya **0**.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Wilayah & Waktu**")
            regions_input = st.multiselect("Pilih Wilayah:", options=regions, default=[regions[0]])
            years_input = st.multiselect("Pilih Tahun:", options=list(range(2020, 2031)), default=[2025])
        
        with c2:
            st.markdown("**2. Faktor Penyebab (Aktif)**")
            selected_factor_labels = st.multiselect("Pilih Faktor:", options=factor_cols)
            
            # Input Dinamis (Optional jika ingin custom nilai, tapi default Median cukup)
            inputs = {}
            if selected_factor_labels:
                st.caption("Ubah nilai jika perlu (Default: Median):")
                for f in selected_factor_labels:
                    default_val = float(df[f].median())
                    inputs[f] = st.number_input(f"{f}:", min_value=0.0, value=default_val)

        st.markdown("---")
        submit_pred = st.form_submit_button("🚀 Hitung Prediksi")

    if submit_pred:
        if not regions_input or not years_input:
            st.warning("Pilih minimal satu Wilayah dan Tahun.")
        elif not mlp_model or not rf_model:
            st.error("Model AI belum siap.")
        else:
            rows = []
            for r in regions_input:
                for y in years_input:
                    row = {REGION_COL: r, YEAR_COL: y}
                    for f in factor_cols:
                        # Jika dipilih, pakai input user (atau median). Jika tidak, 0.
                        row[f] = inputs.get(f, 0.0) if f in selected_factor_labels else 0.0
                    rows.append(row)

            input_df = pd.DataFrame(rows)
            # Pastikan urutan kolom
            X_pred = preprocessor.transform(input_df[feature_cols])
            
            y_mlp = mlp_model.predict(X_pred).flatten()
            y_rf = rf_model.predict(X_pred)
            
            # Result Table
            res_df = input_df[[REGION_COL, YEAR_COL]].copy()
            res_df["Prediksi MLP"] = y_mlp
            res_df["Prediksi RF"] = y_rf
            res_df["Selisih"] = abs(y_mlp - y_rf)
            
            for c in ["Prediksi MLP", "Prediksi RF", "Selisih"]:
                res_df[c] = res_df[c].apply(lambda x: f"{x:,.0f}")
            
            st.success("Prediksi Selesai!")
            st.dataframe(res_df, use_container_width=True)
            
            # Visualisasi
            melted = res_df.melt(id_vars=[REGION_COL, YEAR_COL], 
                                 value_vars=["Prediksi MLP", "Prediksi RF"],
                                 var_name="Model", value_name="Total")
            melted["Total"] = melted["Total"].str.replace(",", "").astype(float)
            
            fig = px.bar(melted, x="Total", y=REGION_COL, color="Model", barmode="group",
                         title="Perbandingan Prediksi", orientation='h',
                         color_discrete_map={"Prediksi MLP": COLOR_MLP, "Prediksi RF": COLOR_RF})
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 5: EVALUASI (FIXED HTML RENDER)
# ==========================================
with tab5:
    st.subheader("📉 Evaluasi Performa Model")
    test_yr = years[-1]
    
    df_test = df[df[YEAR_COL] == test_yr].copy()
    if df_test.empty:
        st.warning("Data testing tidak tersedia.")
    else:
        # Hitung Metrik
        X_t = preprocessor.transform(df_test[feature_cols])
        y_true = df_test[TARGET_COL].values
        
        p_mlp = mlp_model.predict(X_t).flatten()
        p_rf = rf_model.predict(X_t)
        
        mae_mlp = mean_absolute_error(y_true, p_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_true, p_mlp))
        r2_mlp = r2_score(y_true, p_mlp)
        
        mae_rf = mean_absolute_error(y_true, p_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, p_rf))
        r2_rf = r2_score(y_true, p_rf)

        # Kartu Utama
        best_model = "MLP" if mae_mlp < mae_rf else "Random Forest"
        best_mae = min(mae_mlp, mae_rf)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Terbaik", best_model)
        c2.metric("Rata-rata Error (MAE)", f"{best_mae:.0f}")
        c3.metric("Akurasi (R²)", f"{max(r2_mlp, r2_rf):.1%}")
        
        st.markdown("---")
        
        # Tabel
        met_df = pd.DataFrame({
            "Model": ["MLP", "Random Forest"],
            "MAE": [mae_mlp, mae_rf],
            "RMSE": [rmse_mlp, rmse_rf],
            "R2": [r2_mlp, r2_rf]
        })
        st.table(met_df.set_index("Model").style.format("{:.2f}"))
        
        # Scatter
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_true, y=p_mlp, mode='markers', name='MLP', marker=dict(color=COLOR_MLP)))
        fig_sc.add_trace(go.Scatter(x=y_true, y=p_rf, mode='markers', name='RF', marker=dict(color=COLOR_RF, symbol='x')))
        fig_sc.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], 
                                    mode='lines', name='Ideal', line=dict(color='gray', dash='dash')))
        fig_sc.update_layout(title="Aktual vs Prediksi", xaxis_title="Aktual", yaxis_title="Prediksi")
        st.plotly_chart(fig_sc, use_container_width=True)
        
        # Insight Kaya (HTML Fix)
        # String dibuat rapat ke kiri (no indentation) agar Markdown merender HTML dengan benar
        insight_html = f"""
<div class='insight-box'>
<div class='insight-title'>💡 Kesimpulan Evaluasi</div>
<div class='insight-content'>
<p>Berdasarkan pengujian data tahun <b>{test_yr}</b>, model <b>{best_model}</b> menunjukkan akurasi yang lebih tinggi.</p>
<ul>
<li><b>Akurasi:</b> Memiliki error terendah (MAE: {best_mae:.2f}).</li>
<li><b>Konsistensi:</b> Prediksi lebih mendekati garis ideal pada grafik scatter.</li>
<li><b>Rekomendasi:</b> Gunakan model ini untuk acuan utama.</li>
</ul>
</div>
</div>
"""
        st.markdown(insight_html, unsafe_allow_html=True)
