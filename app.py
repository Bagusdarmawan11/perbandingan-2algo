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
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .insight-box {
            background-color: #f0f7fb;
            border-left: 5px solid #457B9D;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #1d3557;
        }
        .insight-title {
            font-weight: bold;
            font-size: 1.1em;
            display: block;
            margin-bottom: 8px;
            color: #1d3557;
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
                       .replace("- ", "") \
                       .strip()
            
            # Jika masih ada sisa " - "
            if " - " in clean:
                clean = clean.split(" - ")[-1]
            
            new_cols.append(clean.strip())
            
    # Deduplikasi nama kolom
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

# --- EKSEKUSI ---
df = load_and_clean_data()
preprocessor, mlp_model, rf_model, feature_cols, factor_cols = load_artifacts(df)

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# ==========================================
# 4. SIDEBAR FILTER
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
        selected_region = st.selectbox("Wilayah:", ["(Semua)"] + regions)
        selected_year = st.selectbox("Tahun:", years, index=len(years)-1)
    
    elif page == "📈 Eksplorasi Daerah & Faktor":
        exp_year = st.selectbox("Pilih Tahun:", years, index=len(years)-1)

    elif page == "🗺️ Peta Jawa Barat":
        map_year = st.selectbox("Pilih Tahun Peta:", years, index=len(years)-1)

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
# 5. TABS UTAMA
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Eksplorasi Daerah & Faktor",
    "🗺️ Peta Jawa Barat",
    "🔮 Prediksi (MLP vs RF)",
    "📑 Tabel Data",
    "📉 Evaluasi Model"
])

# ====== TAB 1: EKSPLORASI ======
with tab1:
    st.subheader(f"📈 Analisis Tahun {selected_year}")
    df_year = df[df[YEAR_COL] == selected_year].copy()

    # 1. Grafik Daerah (Horizontal)
    st.markdown("#### 🔥 Daerah dengan Angka Perceraian Tertinggi")
    df_year_sorted = df_year.sort_values(TARGET_COL, ascending=True)

    fig_region = px.bar(
        df_year_sorted,
        x=TARGET_COL, y=REGION_COL, orientation="h",
        labels={REGION_COL: "Wilayah", TARGET_COL: "Total Kasus"},
        text_auto='.2s', template="plotly_white",
        color=REGION_COL
    )
    fig_region.update_layout(yaxis=dict(categoryorder="total ascending"), height=600, showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

    # 2. Grafik Faktor
    st.markdown("---")
    st.markdown("#### 🧩 Faktor-faktor Penyebab Utama")
    
    valid_factors = [c for c in factor_cols if c in df_year.columns]
    
    if valid_factors:
        factor_sum = df_year[valid_factors].sum().sort_values(ascending=True)
        factor_df = factor_sum.reset_index()
        factor_df.columns = ["Faktor", "Nilai"]

        fig_factor = px.bar(
            factor_df,
            x="Nilai", y="Faktor", orientation="h",
            title=f"Total Kontribusi per Faktor ({selected_year})",
            text_auto='.2s', template="plotly_white", 
            color="Nilai", color_continuous_scale="Reds"
        )
        fig_factor.update_layout(yaxis=dict(categoryorder="total ascending"), height=600)
        st.plotly_chart(fig_factor, use_container_width=True)
    else:
        st.warning("Data faktor tidak ditemukan.")
    
    st.markdown(f"""
    <div class='insight-box'>
        <span class='insight-title'>💡 Insight Eksplorasi</span>
        Grafik batang di atas memudahkan Anda melihat peringkat wilayah dari kasus terendah hingga tertinggi. 
        Pada tahun <b>{selected_year}</b>, faktor penyebab yang paling mendominasi dapat dilihat pada grafik kedua, 
        di mana batang terpanjang menunjukkan penyebab paling sering terjadi.
    </div>""", unsafe_allow_html=True)


# ====== TAB 2: PETA ======
with tab2:
    st.subheader(f"🗺️ Peta Persebaran Jawa Barat ({selected_year})")
    df_year = df[df[YEAR_COL] == selected_year].copy()
    geojson = load_geojson()

    if geojson:
        try:
            fig_map = px.choropleth(
                df_year,
                geojson=geojson,
                locations=REGION_COL,
                featureidkey="properties.NAME_2", # Sesuaikan dengan GeoJSON
                color=TARGET_COL,
                color_continuous_scale="Reds",
                hover_name=REGION_COL,
                title=f"Peta Wilayah Perceraian ({selected_year})"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal render peta wilayah. Menggunakan peta titik.")
            geojson = None

    if not geojson:
        lats, lons = [], []
        for reg in df_year[REGION_COL]:
            coords = JABAR_COORDS.get(reg.strip().upper(), [-6.9, 107.6])
            lats.append(coords[0])
            lons.append(coords[1])
            
        df_year['lat'] = lats
        df_year['lon'] = lons
        
        fig_map = px.scatter_mapbox(
            df_year, lat="lat", lon="lon", size=TARGET_COL, color=TARGET_COL,
            hover_name=REGION_COL, color_continuous_scale="Reds", size_max=45, zoom=7.2,
            mapbox_style="carto-positron", title=f"Peta Titik Panas ({selected_year})"
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.info("ℹ️ Menampilkan Peta Koordinat.")

    st.markdown("""
    <div class='insight-box'>
        <span class='insight-title'>💡 Analisis Spasial</span>
        Peta ini memvisualisasikan konsentrasi kasus perceraian. Wilayah dengan warna lebih gelap (merah tua) 
        menunjukkan tingkat kerawanan sosial yang lebih tinggi terkait perceraian. Ini dapat menjadi acuan 
        untuk prioritas penyuluhan keluarga.
    </div>""", unsafe_allow_html=True)


# ====== TAB 3: PREDIKSI (CLEAN INPUT) ======
with tab3:
    st.subheader("🔮 Simulasi & Prediksi (MLP vs RF)")
    st.markdown("Pilih wilayah dan faktor penyebab. Faktor yang dipilih akan diisi nilai **MEDIAN**, sisanya **0**.")

    with st.form("prediction_form"):
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("##### 1. Parameter Prediksi")
            regions_input = st.multiselect(
                "Pilih Kabupaten/Kota:",
                options=regions,
                default=[regions[0]] if len(regions) > 0 else [],
            )

            years_input = st.multiselect(
                "Pilih Tahun Prediksi:",
                options=list(range(2020, 2031)),
                default=[2025],
            )

            st.markdown("##### 2. Faktor Penyebab Aktif")
            st.caption("Pilih faktor yang diasumsikan terjadi:")
            # Faktor otomatis bersih namanya karena load_data() sudah membersihkannya
            selected_factor_labels = st.multiselect(
                "Pilih Faktor:",
                options=factor_cols,
            )

        with col_right:
            st.info("Klik tombol di bawah untuk memproses.")
            submit = st.form_submit_button("🚀 Hitung Prediksi")

    if submit:
        if not regions_input or not years_input:
            st.warning("Pilih minimal satu Wilayah dan satu Tahun.")
        elif not mlp_model or not rf_model:
            st.error("Model AI tidak ditemukan.")
        else:
            rows = []
            for region in regions_input:
                for year in years_input:
                    row = {REGION_COL: region, YEAR_COL: year}
                    for col in factor_cols:
                        if col in selected_factor_labels:
                            row[col] = df[col].median()
                        else:
                            row[col] = 0.0
                    rows.append(row)

            input_df = pd.DataFrame(rows)
            input_df_final = input_df[feature_cols]

            try:
                X_p = preprocessor.transform(input_df_final)
                y_mlp = mlp_model.predict(X_p).flatten()
                y_rf = rf_model.predict(X_p)

                result_df = input_df[[REGION_COL, YEAR_COL]].copy()
                result_df["Prediksi MLP"] = y_mlp
                result_df["Prediksi RF"] = y_rf
                result_df["Selisih"] = abs(y_mlp - y_rf)
                
                # Format
                for c in ["Prediksi MLP", "Prediksi RF", "Selisih"]:
                    result_df[c] = result_df[c].apply(lambda x: f"{x:,.0f}")

                st.success("✅ Prediksi Selesai!")
                st.dataframe(result_df, use_container_width=True)
                
                # Visualisasi
                if len(result_df) > 0:
                    melted_res = result_df.melt(id_vars=[REGION_COL, YEAR_COL], 
                                              value_vars=["Prediksi MLP", "Prediksi RF"],
                                              var_name="Model", value_name="Nilai")
                    melted_res["Nilai"] = melted_res["Nilai"].str.replace(",", "").astype(float)
                    
                    fig_comp = px.bar(melted_res, x="Nilai", y=REGION_COL, color="Model", barmode="group",
                                      title="Perbandingan Prediksi per Wilayah", orientation='h',
                                      color_discrete_map={"Prediksi MLP": COLOR_MLP, "Prediksi RF": COLOR_RF})
                    st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.error(f"Error prediksi: {e}")
                
    st.markdown(f"""
    <div class='insight-box'>
        <span class='insight-title'>💡 Analisis Prediksi</span>
        Hasil di atas menunjukkan estimasi jumlah perceraian berdasarkan faktor yang Anda pilih. 
        Perbedaan angka antara MLP (Neural Network) dan Random Forest menunjukkan ketidakpastian model.
        Angka prediksi ini dapat digunakan sebagai dasar perencanaan (misal: penyediaan layanan konseling) di tahun mendatang.
    </div>""", unsafe_allow_html=True)


# ====== TAB 4: TABEL DATA ======
with tab4:
    st.subheader("📑 Tabel Data Mentah")
    c1, c2 = st.columns(2)
    with c1:
        reg_filter = st.selectbox("Filter Wilayah:", ["(Semua)"] + regions)
    with c2:
        yr_filter = st.selectbox("Filter Tahun:", ["(Semua)"] + [str(y) for y in years])
    
    df_filtered = df.copy()
    if reg_filter != "(Semua)":
        df_filtered = df_filtered[df_filtered[REGION_COL] == reg_filter]
    if yr_filter != "(Semua)":
        df_filtered = df_filtered[df_filtered[YEAR_COL] == int(yr_filter)]
        
    st.dataframe(df_filtered, use_container_width=True)


# ====== TAB 5: EVALUASI (IMPROVED INSIGHT) ======
with tab5:
    st.subheader("📉 Evaluasi Performa Model")
    test_yr = years[-1]
    
    df_test = df[df[YEAR_COL] == test_yr].copy()
    if df_test.empty:
        st.warning("Data uji kosong.")
    else:
        X_t = preprocessor.transform(df_test[feature_cols])
        y_t = df_test[TARGET_COL].values

        p_mlp = mlp_model.predict(X_t).flatten()
        p_rf = rf_model.predict(X_t)

        # Hitung Metrik
        mae_mlp = mean_absolute_error(y_t, p_mlp)
        mae_rf = mean_absolute_error(y_t, p_rf)
        rmse_mlp = np.sqrt(mean_squared_error(y_t, p_mlp))
        rmse_rf = np.sqrt(mean_squared_error(y_t, p_rf))
        
        # Hitung R2 Score (Akurasi kecocokan)
        r2_mlp = r2_score(y_t, p_mlp)
        r2_rf = r2_score(y_t, p_rf)

        # 1. KARTU METRIK UTAMA
        best_model_name = "MLP (Neural Network)" if mae_mlp < mae_rf else "Random Forest"
        best_mae_val = min(mae_mlp, mae_rf)
        best_r2_val = max(r2_mlp, r2_rf)

        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Model Terbaik", best_model_name, delta="Rekomendasi Utama")
        c2.metric("Rata-rata Error (MAE)", f"{best_mae_val:.0f} Kasus", delta="Semakin kecil semakin baik", delta_color="inverse")
        c3.metric("Akurasi Pola (R²)", f"{best_r2_val:.1%}", help="Seberapa baik model mengikuti pola data aktual")

        st.markdown("---")

        # 2. TABEL PERBANDINGAN
        st.markdown("##### 📋 Detail Metrik Error")
        metrics = pd.DataFrame({
            "Model": ["MLP (Neural Network)", "Random Forest"],
            "MAE (Mean Absolute Error)": [mae_mlp, mae_rf],
            "RMSE (Root Mean Sq. Error)": [rmse_mlp, rmse_rf],
            "R² Score (Goodness of Fit)": [r2_mlp, r2_rf]
        })
        st.table(metrics.set_index("Model").style.format("{:.2f}"))

        # 3. SCATTER PLOT
        st.markdown("##### 🎯 Validasi Visual: Aktual vs Prediksi")
        fig_sc = go.Figure()
        # Garis Ideal
        fig_sc.add_trace(go.Scatter(x=[y_t.min(), y_t.max()], y=[y_t.min(), y_t.max()], 
                                    mode='lines', name='Garis Ideal (Akurasi 100%)', 
                                    line=dict(color='gray', dash='dash', width=2)))
        # Scatter Points
        fig_sc.add_trace(go.Scatter(x=y_t, y=p_mlp, mode='markers', name='MLP Predictions', 
                                    marker=dict(color=COLOR_MLP, size=10, opacity=0.7, symbol='circle')))
        fig_sc.add_trace(go.Scatter(x=y_t, y=p_rf, mode='markers', name='RF Predictions', 
                                    marker=dict(color=COLOR_RF, size=10, opacity=0.7, symbol='x')))
        
        fig_sc.update_layout(
            xaxis_title="Jumlah Aktual (Real)", 
            yaxis_title="Jumlah Prediksi Model",
            height=500,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        
        # 4. INSIGHT MENDALAM
        improvement = abs(mae_mlp - mae_rf)
        
        st.markdown(f"""
        <div class='insight-box'>
            <span class='insight-title'>💡 Kesimpulan Evaluasi Komprehensif</span>
            <p>Evaluasi dilakukan menggunakan data tahun terakhir (<b>{test_yr}</b>) sebagai data uji yang tidak pernah dilihat model selama proses pelatihan (training).</p>
            
            <p><b>1. Performa Model Terbaik:</b><br>
            Model <b>{best_model_name}</b> terbukti lebih unggul dengan rata-rata kesalahan (MAE) sebesar <b>{best_mae_val:.0f}</b> kasus per wilayah. 
            Ini berarti, rata-rata prediksi model hanya meleset sekitar {best_mae_val:.0f} orang dari kenyataan.</p>
            
            <p><b>2. Perbandingan Stabilitas:</b><br>
            Selisih performa antara kedua model adalah <b>{improvement:.2f}</b> poin. 
            Nilai <b>R² sebesar {best_r2_val:.1%}</b> menunjukkan bahwa model ini sangat baik dalam menangkap pola kenaikan/penurunan kasus antar wilayah.</p>
            
            <p><b>3. Interpretasi Grafik Scatter:</b><br>
            Titik-titik pada grafik di atas menunjukkan sebaran prediksi. Semakin dekat titik-titik tersebut ke <b>Garis Ideal (Putus-putus)</b>, semakin akurat modelnya. 
            Terlihat bahwa model {best_model_name} memiliki sebaran yang lebih rapat ke garis diagonal dibandingkan kompetitornya.</p>
        </div>""", unsafe_allow_html=True)
