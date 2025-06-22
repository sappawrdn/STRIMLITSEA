# 🔧 STREAMLIT DASHBOARD FINAL (MODIFIED)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pydeck as pdk


# 🚀 Setup halaman
st.set_page_config(page_title="Dashboard Pendidikan & Kemiskinan", layout="wide")
st.title("📊 Dashboard Data Pendidikan dan Kemiskinan")

# 📂 Load data
@st.cache_data
def load_data():
    df = pd.read_excel("alldata.xlsx")
    df.columns = df.columns.str.strip()
    df['Jumlah Penduduk'] = df['Jumlah Penduduk'].round(0).astype(int)
    df['Jumlah Penduduk Miskin'] = df['Jumlah Penduduk Miskin'].round(0).astype(int)
    df['Sudah Verifikasi'] = df[['k_2', 'k_3', 'k_7', 'k_12', 'k_13']].sum(axis=1)

    df_koordinat = pd.read_csv("koordinat_daerah.csv")
    df_koordinat.columns = df_koordinat.columns.str.strip()
    df = df.merge(df_koordinat, on="Kabupaten/Kota", how="left")

    return df

data = load_data()

fitur_klaster = [
    'Jumlah Penduduk Miskin',
    'k_2', 'k_3', 'k_7', 'k_12', 'k_13', 'Sudah Verifikasi'
]

# --- 🧭 Sidebar Filters ---
st.sidebar.markdown("## 📍 Filter Daerah")
provinsi_list = sorted(data['Provinsi'].unique())
provinsi_list_with_all = ["-- Pilih Semua --"] + provinsi_list
selected_provinsi = st.sidebar.multiselect("Pilih Provinsi", provinsi_list_with_all)

if "-- Pilih Semua --" in selected_provinsi or not selected_provinsi:
    selected_provinsi = provinsi_list

kota_list = sorted(data[data['Provinsi'].isin(selected_provinsi)]['Kabupaten/Kota'].unique())
kota_list_with_all = ["-- Pilih Semua --"] + kota_list
selected_kota = st.sidebar.multiselect("Pilih Kabupaten/Kota", kota_list_with_all)

if "-- Pilih Semua --" in selected_kota or not selected_kota:
    selected_kota = kota_list

filtered_data = data[
    (data['Provinsi'].isin(selected_provinsi)) &
    (data['Kabupaten/Kota'].isin(selected_kota))
].copy()

# 👉 Pilih Algoritma
st.subheader("🔍 Pilih Algoritma Clustering")
selected_algo = st.selectbox("Pilih salah satu algoritma:", ["KMeans", "Agglomerative Clustering", "DBSCAN"])

# 👉 Clustering Process
if st.button("🔄 Proses Ulang Clustering"):
    data_cluster = data[fitur_klaster].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cluster)

    if selected_algo == "KMeans":
        model = KMeans(n_clusters=3, random_state=42)
    elif selected_algo == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=3)
    elif selected_algo == "DBSCAN":
        model = DBSCAN(eps=0.5, min_samples=5)

    labels = model.fit_predict(data_scaled)
    data['Cluster'] = -1
    data.loc[data_cluster.index, 'Cluster'] = labels

    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(data_scaled, labels)
        st.success(f"✅ Silhouette Score: **{score:.4f}**")
    else:
        st.warning("⚠️ Silhouette Score tidak bisa dihitung.")

# --- 🧩 Tampilkan Cluster Result Berdasarkan Filter
if 'Cluster' in data.columns and not data['Cluster'].isna().all():
    filtered_data = data[
        (data['Provinsi'].isin(selected_provinsi)) &
        (data['Kabupaten/Kota'].isin(selected_kota))
    ].copy()

    st.subheader("🧩 Hasil Clustering (Sesuai Filter)")
    st.dataframe(
        filtered_data[['Provinsi', 'Kabupaten/Kota', 'Cluster']]
        .sort_values(by='Cluster'),
        use_container_width=True
    )

    # Optional: Download button
    csv = filtered_data[['Provinsi', 'Kabupaten/Kota', 'Cluster']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Hasil Clustering (CSV)",
        data=csv,
        file_name='hasil_clustering.csv',
        mime='text/csv'
    )

# --- 📋 Tampilkan Data Terfilter ---
if not filtered_data.empty:
    st.subheader("📋 Data Terfilter")
    st.dataframe(filtered_data, use_container_width=True)

    if 'Cluster' in filtered_data.columns and 'Latitude' in filtered_data.columns and 'Longitude' in filtered_data.columns:
        st.subheader("🗺️ Visualisasi Peta Clustering")

        cluster_colors = {
            0: [0, 128, 255],
            1: [255, 0, 0],
            2: [255, 165, 0],
        }

        filtered_data['color'] = filtered_data['Cluster'].apply(lambda x: cluster_colors.get(x, [150, 150, 150]))

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=filtered_data,
            get_position='[Longitude, Latitude]',
            get_fill_color='color',
            get_radius=5000,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=-2.5,
            longitude=117,
            zoom=4.2,
            pitch=0
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={"text": "{Kabupaten/Kota}\nCluster: {Cluster}"}
        ))

        st.markdown("### 🎨 Keterangan Warna Cluster")

        legend_html = ""
        for k, rgb in cluster_colors.items():
            color_hex = '#%02x%02x%02x' % tuple(rgb)  # RGB ke HEX
            legend_html += f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
            legend_html += f"<div style='width:20px; height:20px; background-color:{color_hex}; border-radius:3px; margin-right:8px;'></div>"
            legend_html += f"<span>Cluster {k}</span></div>"

        st.markdown(legend_html, unsafe_allow_html=True)


    st.subheader("📌 Statistik Ringkas")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Jumlah Daerah", len(filtered_data))
    col1.metric("📈 Rata-rata % Miskin", f"{filtered_data['Persentase'].mean():.2f}%")
    col2.metric("👥 Total Penduduk", f"{filtered_data['Jumlah Penduduk'].sum():,}")
    col2.metric("🧍‍♂️ Total Penduduk Miskin", f"{filtered_data['Jumlah Penduduk Miskin'].sum():,}")
    col3.metric("✅ Sudah Diverifikasi", f"{int(filtered_data['Sudah Verifikasi'].sum()):,}")

    if 'Cluster' in filtered_data.columns:
        st.subheader("📊 Rata-rata Nilai per Cluster")
        cluster_stats = filtered_data[['Cluster'] + fitur_klaster].groupby('Cluster').mean().reset_index()
        st.dataframe(cluster_stats.style.format("{:.2f}"), use_container_width=True)

    st.subheader("🚨 5 Daerah dengan Prioritas Tertinggi")
    scaler_priority = MinMaxScaler()
    filtered_data[['Norm_Miskin', 'Norm_Verifikasi']] = scaler_priority.fit_transform(
        filtered_data[['Jumlah Penduduk Miskin', 'Sudah Verifikasi']]
    )
    filtered_data['Skor Prioritas'] = (
        0.5 * filtered_data['Norm_Miskin'] +
        0.5 * filtered_data['Norm_Verifikasi']
    )
    top5 = filtered_data.sort_values(by='Skor Prioritas', ascending=False).head(5)
    st.dataframe(
        top5[['Provinsi', 'Kabupaten/Kota', 'Jumlah Penduduk Miskin', 'Sudah Verifikasi', 'Skor Prioritas']].style.format({
            'Jumlah Penduduk Miskin': '{:,.0f}',
            'Sudah Verifikasi': '{:,.0f}',
            'Skor Prioritas': '{:,.2f}',
        }),
        use_container_width=True
    )

    with st.expander("ℹ️ Cara Menentukan Prioritas"):
        st.markdown("""
        Skor dihitung dari kombinasi dua indikator utama:
        - **Jumlah Penduduk Miskin** (50% bobot)
        - **Jumlah Kendala Pendidikan yang Diverifikasi** (50% bobot)
        Semakin tinggi skor, semakin besar urgensi intervensi pada daerah tersebut.
        """)

# 📦 KESIMPULAN PER ALGORITMA
st.subheader("📌 Kesimpulan Clustering")

with st.expander("📦 Kesimpulan: KMeans Clustering"):
    if selected_algo == "KMeans":
        st.markdown("""
        - **Cluster 0** → Daerah dengan jumlah miskin & kendala pendidikan paling rendah.
        - **Cluster 1** → Jumlah miskin tinggi dan kendala sedang.
        - **Cluster 2** → Daerah paling terdampak: kemiskinan & hambatan pendidikan sangat tinggi.
        """)

with st.expander("🌳 Kesimpulan: Agglomerative Clustering"):
    if selected_algo == "Agglomerative Clustering":
        st.markdown("""
        - **Cluster 0** → Kondisi aman: miskin & hambatan rendah.
        - **Cluster 1** → Miskin besar, kendala sedang.
        - **Cluster 2** → Banyak anak bekerja, disabilitas, dan masalah akses.
        """)

with st.expander("🧩 Catatan: DBSCAN Clustering"):
    if selected_algo == "DBSCAN":
        st.markdown("""
        - DBSCAN mendeteksi outlier.
        - Banyak `Cluster -1`? Sesuaikan parameter `eps` & `min_samples`.
        """)
