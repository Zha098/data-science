import streamlit as st
import numpy as np
from joblib import load

# Load model SVM yang disimpan
model = load('Deccision_tree_classifer.joblib')

def predict_pass(G1, G2, G3):
    input_features = np.array([[G1, G2, G3]])  # Hanya 3 fitur
    prediction_continuous = model.predict_proba(input_features)[0][1]
    
    st.write(f"Probabilitas Lulus (Output Model): {prediction_continuous:.4f}")
    threshold = 0.5
    prediction = 1 if prediction_continuous >= threshold else 0
    return prediction, prediction_continuous


# Streamlit UI
st.title("Prediksi Kelulusan Siswa")
st.write("Masukkan fitur-fitur untuk memprediksi apakah siswa akan lulus atau tidak.")

# Input dari pengguna
G1 = st.number_input("Nilai G1:", min_value=0, max_value=20, value=10)
G2 = st.number_input("Nilai G2:", min_value=0, max_value=20, value=10)
G3 = st.number_input("Nilai G3:", min_value=0, max_value=20, value=10)

# Tombol Prediksi
if st.button("Prediksi"):
    prediction, prediction_continuous = predict_pass( G1, G2, G3)
    if prediction == 1:
        st.success(f"Prediksi: **Lulus** 🎉 (Output Model: {prediction_continuous:.2f})")
    else:
        st.error(f"Prediksi: **Tidak Lulus** 😔 (Output Model: {prediction_continuous:.2f})")
