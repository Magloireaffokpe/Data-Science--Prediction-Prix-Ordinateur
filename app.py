import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle, les encodeurs et le scaler
model = joblib.load("laptop_price_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💻 Estimation du prix des ordinateurs portables")

# ---------------- Champs utilisateurs ----------------
company = st.selectbox("Entreprise", encoders['Company'].classes_)
typename = st.selectbox("Type d'appareil", encoders['TypeName'].classes_)
inches = st.slider("Taille écran (pouces)", 10.0, 18.0, step=0.1)
resolution = st.selectbox("Résolution écran", encoders['ScreenResolution'].classes_)
cpu = st.selectbox("Processeur", encoders['Cpu'].classes_)
ram = st.slider("RAM (en Go)", 2, 64, step=2)
memory = st.selectbox("Mémoire", encoders['Memory'].classes_)
gpu = st.selectbox("Carte graphique", encoders['Gpu'].classes_)
opsys = st.selectbox("Système d'exploitation", encoders['OpSys'].classes_)
weight = st.number_input("Poids (kg)", min_value=0.5, max_value=5.0, step=0.1)
brand = st.selectbox("Marque", encoders['Brand'].classes_)
cpu_brand = st.selectbox("Marque du processeur", encoders['Cpu Brand'].classes_)
gpu_brand = st.selectbox("Marque GPU", encoders['Gpu Brand'].classes_)
memory_type = st.selectbox("Type de mémoire", encoders['Memory_Type'].classes_)
touchscreen = st.radio("Écran tactile ?", ["Oui", "Non"])
ips = st.radio("Technologie IPS ?", ["Oui", "Non"])
x_res = st.number_input("Résolution horizontale (X)", min_value=800)
y_res = st.number_input("Résolution verticale (Y)", min_value=600)

# ---------------- Prediction ----------------
if st.button("💡 Estimer le prix"):

    # Encodage des variables catégorielles
    data = {
        'Company': encoders['Company'].transform([company])[0],
        'TypeName': encoders['TypeName'].transform([typename])[0],
        'Inches': inches,
        'ScreenResolution': encoders['ScreenResolution'].transform([resolution])[0],
        'Cpu': encoders['Cpu'].transform([cpu])[0],
        'Ram': ram,
        'Memory': encoders['Memory'].transform([memory])[0],
        'Gpu': encoders['Gpu'].transform([gpu])[0],
        'OpSys': encoders['OpSys'].transform([opsys])[0],
        'Weight': weight,
        'Brand': encoders['Brand'].transform([brand])[0],
        'Cpu Brand': encoders['Cpu Brand'].transform([cpu_brand])[0],
        'Gpu Brand': encoders['Gpu Brand'].transform([gpu_brand])[0],
        'Memory_Type': encoders['Memory_Type'].transform([memory_type])[0],
        'Touchscreen': 1 if touchscreen == "Oui" else 0,
        'IPS': 1 if ips == "Oui" else 0,
        'X_res': x_res,
        'Y_res': y_res,
    }

    try:
        data['ppi'] = ((x_res ** 2 + y_res ** 2) ** 0.5) / inches
    except ZeroDivisionError:
        st.error("Erreur : la taille de l’écran (inches) ne peut pas être zéro.")
        st.stop()

    # Transformation en DataFrame
    features_df = pd.DataFrame([data])

    # Vérification (optionnelle mais utile)
    if features_df.shape[1] != scaler.n_features_in_:
        st.error(f"⚠️ Le modèle attend {scaler.n_features_in_} features mais a reçu {features_df.shape[1]}")
        st.stop()

    # Standardisation
    features_scaled = scaler.transform(features_df)

    # Prédiction
    prediction = model.predict(features_scaled)[0]

    st.success(f"💰 Prix estimé : {round(prediction, 2)} FCFA")
