# File: streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

# Set style untuk plot
plt.style.use("seaborn")

# 1. Header Aplikasi
st.title("Clustering Deployment with Fuzzy C-Means")
st.write("Aplikasi ini melakukan reduksi dimensi dengan SVD dan clustering menggunakan Fuzzy C-Means.")

# 2. Upload Dataset
uploaded_file = st.file_uploader("Upload dataset dalam format CSV", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset berhasil dimuat:")
    st.write(df.head())

    # Pastikan kolom teks tersedia
    if 'stopword_removal' not in df.columns:
        st.error("Dataset harus memiliki kolom 'stopword_removal' untuk analisis.")
    else:
        # Load model SVD yang telah disimpan
        svd_model = joblib.load('svd_model.pkl')  # Ganti path dengan file model Anda
        
        # Load data TF-IDF yang sudah diproses
        st.write("Melakukan reduksi dimensi menggunakan model SVD...")
        tfidf_array = np.load('tfidf_array.npy')  # Ganti path dengan file Anda
        reduced_data = svd_model.transform(tfidf_array)

        # Normalisasi data
        scaler = StandardScaler()
        reduced_data = scaler.fit_transform(reduced_data)

        # 3. Fuzzy C-Means Clustering
        st.write("Melakukan clustering menggunakan Fuzzy C-Means...")
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            reduced_data.T, 2, 2, error=0.005, maxiter=1000, init=None
        )
        labels = np.argmax(u, axis=0)
        st.success("Clustering selesai.")

        # 4. Visualisasi Scatter Plot
        st.write("Hasil Visualisasi Clustering:")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.7)
        ax.set_title("Scatter Plot Clustering")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)

# 5. Tentang Aplikasi
st.sidebar.title("Tentang")
st.sidebar.info("Aplikasi ini dibuat menggunakan Streamlit untuk deployment model clustering.")
