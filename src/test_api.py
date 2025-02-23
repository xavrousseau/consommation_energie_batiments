import pandas as pd
import requests
import json
import bentoml

# 📂 Charger le CSV nettoyé (adapter le chemin si nécessaire)
CSV_PATH = r"F:/1.Boulot/03_Github/consommation_energie_batiments/data/processed/dataset_cleaned.csv"
data = pd.read_csv(CSV_PATH)

# ✅ Récupérer les features attendues depuis les modèles BentoML
try:
    model_energy_ref = bentoml.sklearn.get("site_energy_use_model:latest")
    model_co2_ref = bentoml.sklearn.get("ghg_emissions_model:latest")

    features_energy = model_energy_ref.custom_objects["features"]
    features_co2 = model_co2_ref.custom_objects["features"]

    # 🔄 Vérification de la correspondance des features
    if not set(features_energy).issubset(data.columns):
        raise ValueError("❌ Les features pour le modèle de consommation d'énergie ne correspondent pas aux colonnes du CSV.")

    if not set(features_co2).issubset(data.columns):
        raise ValueError("❌ Les features pour le modèle des émissions de CO₂ ne correspondent pas aux colonnes du CSV.")

    # 🔄 Sélectionner la première ligne du dataset en fonction des features attendues
    sample_row_energy = data[features_energy].iloc[0].tolist()
    sample_row_co2 = data[features_co2].iloc[0].tolist()

    # 📦 Préparer les données pour les endpoints
    energy_payload = json.dumps({"features": sample_row_energy})
    co2_payload = json.dumps({"features": sample_row_co2})

    # 🌐 Définir les endpoints de l'API
    energy_url = "http://127.0.0.1:3000/predict_energy"
    co2_url = "http://127.0.0.1:3000/predict_co2"

    headers = {"Content-Type": "application/json"}

    # 🚀 Tester l'endpoint consommation d'énergie
    response_energy = requests.post(energy_url, headers=headers, data=energy_payload)
    print("⚡ Résultat prédiction consommation d'énergie :", response_energy.json())

    # 🚀 Tester l'endpoint émissions de CO₂
    response_co2 = requests.post(co2_url, headers=headers, data=co2_payload)
    print("🌿 Résultat prédiction émissions de CO₂ :", response_co2.json())

    print("✅ Test de l'API terminé. Les prédictions sont affichées ci-dessus.")

except Exception as e:
    print(f"❌ Une erreur s'est produite : {str(e)}")
