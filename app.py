# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.neighbors import BallTree
import pydeck as pdk
import tensorflow as tf
from sklearn.compose import ColumnTransformer

# ==== Warna per-cluster ====
CLUSTER_PALETTE = [
    [255, 99, 132],   # pink/red
    [54, 162, 235],   # blue
    [255, 206, 86],   # yellow
    [75, 192, 192],   # teal
    [153, 102, 255],  # purple
    [255, 159, 64],   # orange
    [0, 200, 83],     # green
    [233, 30, 99],    # magenta-ish
    [0, 188, 212],    # cyan
    [205, 220, 57],   # lime
]

def build_cluster_color_map(series_clusters):
    """Return dict: cluster_value -> [R,G,B]. Deterministic by sorted unique."""
    uniq = sorted(pd.unique(series_clusters))
    cmap = {}
    for i, c in enumerate(uniq):
        cmap[c] = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
    return cmap


# =========================
# Konstanta & util
# =========================
EARTH_RADIUS_KM = 6371.0088

# Kolom yang DIHARAPKAN oleh 'preprocess' saat training
# (berdasarkan error kamu sebelumnya)
EXPECTED_COLS = ['long', 'lat', 'id_provinsi', 'id_kabupaten', 'id_makanan', 'id_reting']

# Alias nama kolom yang mungkin dipakai di UI/mentah -> target kolom training
ALIASES = {
    'longitude': 'long',
    'latitude': 'lat',
    'provinsi_id': 'id_provinsi',
    'district_id': 'id_kabupaten',
    'kabupaten_id': 'id_kabupaten',
    'food_id': 'id_makanan',
    'id_makanan': 'id_makanan',
    'id_provinsi': 'id_provinsi',
    'id_kabupaten': 'id_kabupaten',
    'id_rating': 'id_reting',
    'rating_id': 'id_reting'
}

def normalize_schema(df_row: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama & urutan kolom agar cocok dengan preprocess training."""
    df = df_row.copy()

    # 1) Map alias -> target (tanpa menghapus kolom asal)
    for src, tgt in ALIASES.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]

    # 2) Tambah kolom yang hilang
    for col in EXPECTED_COLS:
        if col not in df.columns:
            if col in ['long', 'lat']:
                df[col] = np.nan
            else:
                df[col] = "UNK"

    # 3) Urutkan sesuai expected
    df = df[EXPECTED_COLS]

    # 4) Validasi & tipe
    if df['long'].isna().any() or df['lat'].isna().any():
        raise ValueError("Longitude/Latitude wajib diisi.")

    # Kategori sebagai string (aman untuk OneHotEncoder handle_unknown='ignore')
    for c in ['id_provinsi', 'id_kabupaten', 'id_makanan', 'id_reting']:
        df[c] = df[c].astype(str).replace("", "UNK").fillna("UNK")

    # Numerik untuk koordinat
    df['long'] = pd.to_numeric(df['long'])
    df['lat'] = pd.to_numeric(df['lat'])

    return df

def build_tooltip_fields(df: pd.DataFrame):
    """Siapkan teks tooltip untuk pydeck berdasarkan kolom yang tersedia."""
    # Gunakan kolom yang ada saja agar tidak error
    fields = []
    if 'place_name' in df.columns:
        fields.append("{place_name}")
    if 'address' in df.columns:
        fields.append("{address}")
    if 'cluster' in df.columns:
        fields.append("Cluster {cluster}")
    if 'distance_km' in df.columns:
        fields.append("{distance_km} km")
    if not fields:
        fields = ["Restoran"]
    return "\\n".join(fields)

# =========================
# Load artefak (cache)
# =========================
@st.cache_resource(show_spinner=False)
def _load_artifacts():
    scaler_geo = load("scaler_geo.joblib")
    kmeans = load("kmeans.joblib")
    preprocess = load("preprocess.joblib")
    tree = load("balltree_haversine.joblib")
    restaurants_ref = pd.read_csv("restaurants_ref.csv")
    model = tf.keras.models.load_model("nn_cluster_classifier.keras")
    return scaler_geo, kmeans, preprocess, tree, restaurants_ref, model

try:
    scaler_geo, kmeans, preprocess, tree, restaurants_ref, model = _load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak: {e}")
    st.stop()

# Pastikan kolom minimal ada di referensi restoran
for need in ['latitude', 'longitude']:
    if need not in restaurants_ref.columns:
        st.error(f"Kolom '{need}' wajib ada di restaurants_ref.csv")
        st.stop()
# Cluster dipakai untuk filter opsional; kalau tidak ada, kita lanjut tanpa filter
HAS_CLUSTER_REF = 'cluster' in restaurants_ref.columns

# =========================
# UI
# =========================
st.title("Clustering & Klasifikasi Restoran + Nearest Finder")
st.caption("Prediksi cluster (KMeans & NN) dan cari restoran terdekat (haversine).")

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=-6.200000, format="%.6f")
with col2:
    longitude = st.number_input("Longitude", value=106.816666, format="%.6f")

province_id = st.text_input("province_id (opsional)", value="")
district_id = st.text_input("district_id (opsional)", value="")
food_id = st.text_input("food_id (opsional)", value="")
rating_id = st.text_input("rating_id (opsional)", value="")

n_neighbors = st.slider("Jumlah restoran terdekat", 1, 50, 5)
only_same_cluster = st.checkbox("Hanya tampilkan restoran dari cluster yang sama (berdasarkan NN)", value=False and HAS_CLUSTER_REF)
if only_same_cluster and not HAS_CLUSTER_REF:
    st.info("restaurants_ref.csv tidak punya kolom 'cluster', filter cluster dimatikan.")

if st.button("Prediksi & Cari Terdekat"):
    try:
        # =========================
        # Prediksi cluster via KMeans (geo-only)
        # =========================
        # scaler_geo dilatih pada urutan [longitude, latitude]
        X_geo_new = scaler_geo.transform([[longitude, latitude]])
        cluster_kmeans = int(kmeans.predict(X_geo_new)[0])

        # =========================
        # Prediksi cluster via NN (fitur lengkap)
        # =========================
        raw_row = pd.DataFrame([{
            'longitude': longitude,
            'latitude': latitude,
            'id_provinsi': province_id,   # mapping ke nama training dilakukan di normalize_schema
            'id_kabupaten': district_id,
            'id_makanan': food_id,
            'id_reting': rating_id        # konsisten dg ejaan saat training
        }])

        data_row = normalize_schema(raw_row)
        X_trans = preprocess.transform(data_row)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        pred_proba = model.predict(X_trans, verbose=0)[0]
        cluster_nn = int(np.argmax(pred_proba))

        st.subheader("Hasil Prediksi Cluster")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="KMeans (geo-only)", value=f"Cluster {cluster_kmeans}")
        with c2:
            st.metric(label="Neural Net (fitur lengkap)", value=f"Cluster {cluster_nn}")

        st.write("Probabilitas NN per cluster:")
        st.json({f"c{idx}": float(p) for idx, p in enumerate(pred_proba)})

        # =========================
        # Nearest restaurants (haversine)
        # =========================
        if len(restaurants_ref) == 0:
            st.warning("Tidak ada data restoran di referensi.")
            st.stop()

        # BallTree dilatih pada koordinat radian [lat, lon]
        query_rad = np.deg2rad([[latitude, longitude]])
        k_query = min(max(n_neighbors * 5, n_neighbors), len(restaurants_ref))
        dist_rad, idxs = tree.query(query_rad, k=k_query)
        dist_km = dist_rad[0] * EARTH_RADIUS_KM
        idxs = idxs[0]

        candidates = restaurants_ref.iloc[idxs].copy()
        candidates['distance_km'] = np.round(dist_km, 3)

        if only_same_cluster and HAS_CLUSTER_REF:
            before_n = len(candidates)
            candidates = candidates[candidates['cluster'] == cluster_nn]
            if candidates.empty:
                st.warning("Tidak ada restoran dalam cluster yang sama. Menampilkan terdekat tanpa filter cluster.")
                candidates = restaurants_ref.iloc[idxs].copy()
                candidates['distance_km'] = np.round(dist_km, 3)

        result = candidates.nsmallest(n_neighbors, 'distance_km')

        def mk_tooltip_row(row):
            parts = []
            if 'place_name' in row.index and pd.notna(row['place_name']):
                parts.append(str(row['place_name']))
            if 'address' in row.index and pd.notna(row['address']):
                parts.append(str(row['address']))
            if 'cluster' in row.index:
                parts.append(f"Cluster {row['cluster']}")
            if 'distance_km' in row.index:
                parts.append(f"{row['distance_km']} km")
            parts.append(f"({row['latitude']:.6f}, {row['longitude']:.6f})")
            return "\n".join(parts)

        result = result.copy()
        result['tooltip'] = result.apply(mk_tooltip_row, axis=1)

        st.subheader("Restoran Terdekat")

        # daftar kolom yang ingin ditampilkan (urutannya opsional)
        preferred_cols = ['restaurant_id','place_name','address','latitude','longitude','cluster','distance_km']

        # ambil hanya kolom yang benar-benar ada
        cols_show = [c for c in preferred_cols if c in result.columns]
        st.dataframe(result[cols_show], use_container_width=True)

        # Tombol download CSV (pakai kolom yang ada juga)
        csv_bytes = result[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV titik terdekat", data=csv_bytes, file_name="nearest_restaurants.csv", mime="text/csv")

        # =========================
        # Peta interaktif (pydeck)
        # =========================
        # --- Peta interaktif dengan pydeck: titik user + titik restoran + garis penghubung ---
        # --- Peta interaktif (pydeck) dengan warna per-cluster ---
        if not result.empty:
            midpoint = [float(result['latitude'].mean()), float(result['longitude'].mean())]

            user_df = pd.DataFrame([{"latitude": latitude, "longitude": longitude}])

            if HAS_CLUSTER_REF and ('cluster' in result.columns):
                # siapkan warna per cluster
                color_map = build_cluster_color_map(result['cluster'])
                # titik restoran berwarna sesuai cluster (alpha 220)
                result = result.copy()
                result['color'] = result['cluster'].map(color_map).apply(lambda rgb: [rgb[0], rgb[1], rgb[2], 220])

                # arcs: dari user -> restoran, target_color = warna cluster
                arcs = result[['latitude','longitude','color']].copy()
                arcs['src_lat'] = latitude
                arcs['src_lon'] = longitude

                layer_points = pdk.Layer(
                    "ScatterplotLayer",
                    data=result,
                    get_position='[longitude, latitude]',
                    get_fill_color='color',
                    get_radius=60,
                    pickable=True
                )
                layer_user = pdk.Layer(
                    "ScatterplotLayer",
                    data=user_df,
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 255, 255, 255],  # user = putih terang
                    get_radius=120,
                    pickable=False
                )
                layer_arcs = pdk.Layer(
                    "ArcLayer",
                    data=arcs,
                    get_source_position='[src_lon, src_lat]',
                    get_target_position='[longitude, latitude]',
                    get_source_color=[255, 255, 255, 120],  # asal (user) = putih semi-transparan
                    get_target_color='color',               # target = warna cluster
                    get_width=3,
                    pickable=False
                )
            else:
                # fallback: jika tidak ada kolom cluster
                layer_points = pdk.Layer(
                    "ScatterplotLayer",
                    data=result,
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 255, 0, 220],   # kuning terang
                    get_radius=60,
                    pickable=True
                )
                layer_user = pdk.Layer(
                    "ScatterplotLayer",
                    data=user_df,
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 255, 255, 255], # putih terang
                    get_radius=120,
                    pickable=False
                )
                layer_arcs = pdk.Layer(
                    "ArcLayer",
                    data=pd.DataFrame({
                        "src_lat": [latitude]*len(result),
                        "src_lon": [longitude]*len(result),
                        "latitude": result["latitude"].values,
                        "longitude": result["longitude"].values
                    }),
                    get_source_position='[src_lon, src_lat]',
                    get_target_position='[longitude, latitude]',
                    get_source_color=[255, 255, 255, 120],
                    get_target_color=[255, 255, 0, 220],  # kuning terang
                    get_width=3,
                    pickable=False
                )

            tooltip_text = "{place_name}\n{address}\nCluster {cluster}\n{distance_km} km\n({latitude}, {longitude})"
            r = pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=12),
                layers=[layer_points, layer_user, layer_arcs],
                tooltip={"text": "{tooltip}"}
            )
            st.pydeck_chart(r)
        else:
            st.info("Tidak ada hasil untuk ditampilkan di peta.")


    except ValueError as ve:
        # Error umum: kolom/tulisan salah, lat/long kosong, dsb
        st.error(f"Input tidak valid: {ve}")
    except Exception as e:
        st.error(f"Terjadi error: {e}")
