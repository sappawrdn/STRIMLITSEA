import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("📊 Clustering Dashboard - Pendidikan dan Kemiskinan")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel kamu", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    # Pilih kolom yang mau di-cluster
    columns_to_cluster = ['Jumlah Penduduk Miskin', 'k_2', 'k_3', 'k_7', 'k_12', 'k_13', 'Sudah Verifikasi']
    
    # Preprocessing
    data_clean = data[columns_to_cluster].dropna()
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_clean)
    
    # Sidebar pilihan algoritma
    algo = st.sidebar.selectbox("Pilih Algoritma Clustering", ["KMeans", "DBSCAN", "Agglomerative"])
    
    # Visualisasi clustering
    if algo == "KMeans":
        k = st.sidebar.slider("Jumlah Klaster (k)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data_normalized)
    elif algo == "DBSCAN":
        eps = st.sidebar.slider("Eps", 0.1, 2.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data_normalized)
    else:
        k = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(data_normalized)

    # Tambahkan label ke data
    data_clean['Cluster'] = labels

    # Visualisasi scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        data_normalized[:, 0], data_normalized[:, -1], c=labels, cmap='viridis'
    )
    ax.set_xlabel("Jumlah Penduduk Miskin (Normalized)")
    ax.set_ylabel("Sudah Verifikasi (Normalized)")
    ax.set_title(f"Hasil Clustering dengan {algo}")
    st.pyplot(fig)

    # Hitung silhouette score (kalau bisa)
    if len(set(labels)) > 1 and (len(set(labels)) != 1 or -1 not in labels):
        sil_score = silhouette_score(data_normalized[:, [0, -1]], labels)
        st.success(f"Silhouette Score: {sil_score:.4f}")
    else:
        st.warning("Tidak bisa hitung Silhouette Score (klaster < 2)")

    # Tampilkan data hasil klasterisasi
    st.subheader("📋 Data dengan Label Klaster")
    st.dataframe(data_clean.reset_index(drop=True))