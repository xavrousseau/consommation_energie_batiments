# ============================================================
# 🧪 Script de test (pytest) : test_api.py
#     - Teste les endpoints de l'API BentoML en local
#     - Utilise les chemins depuis config.py pour cohérence
#     - Utilise pytest pour une exécution simplifiée et structurée
#     - Logs détaillés et colorés avec loguru pour un suivi clair
# ============================================================

import sys
from pathlib import Path
import pandas as pd
import requests
import bentoml
from loguru import logger
import pytest
import os

# ============================================================
# 📂 Configuration dynamique pour accéder au dossier src/
# ------------------------------------------------------------
# Cette section permet au script de trouver le module config.py
# qui se trouve dans le dossier src/.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # 📍 Remonte à la racine du projet
sys.path.append(str(BASE_DIR / "src"))  # 🔗 Ajoute src/ au chemin Python

# 🔎 Debug temporaire pour valider les chemins
print(f"📁 Chemin actuel : {os.getcwd()}")
print(f"📁 Chemin attendu pour config.py : {BASE_DIR / 'src' / 'config.py'}")
print(f"🧐 Fichier config.py existe ? {os.path.exists(BASE_DIR / 'src' / 'config.py')}")

from config import PROCESSED_ENERGY_PATH, PROCESSED_CO2_PATH, ENERGY_MODEL_PATH, CO2_MODEL_PATH

# ✅ Vérification de l'import du fichier config.py
try:
    print(f"✅ Chemin vers PROCESSED_ENERGY_PATH : {PROCESSED_ENERGY_PATH}")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")

# ============================================================
# 📂 Fixture pour charger les datasets traités
# ------------------------------------------------------------
# ⚡ Cette fonction charge une seule fois les datasets pour tous les tests
# afin d'améliorer les performances (scope="session").
# ============================================================
@pytest.fixture(scope="session")
def load_data():
    """📂 Charge les datasets traités une seule fois par session de test."""
    print(f"🔍 Tentative de chargement : {PROCESSED_ENERGY_PATH}")
    print(f"🔍 Tentative de chargement : {PROCESSED_CO2_PATH}")
    try:
        data_energy = pd.read_csv(PROCESSED_ENERGY_PATH)
        data_co2 = pd.read_csv(PROCESSED_CO2_PATH)
        logger.success("✅ Données chargées avec succès.")
        return data_energy, data_co2
    except FileNotFoundError as e:
        logger.error(f"❌ Fichier introuvable : {e}")
        pytest.exit("❌ Arrêt des tests : données manquantes.", returncode=1)


# ============================================================
# 🔗 Fixture pour charger les modèles et récupérer les features
# ------------------------------------------------------------
# ⚡ Charge une seule fois les modèles depuis BentoML pour tous les tests.
# Retourne la liste des features nécessaires à chaque modèle.
# ============================================================
@pytest.fixture(scope="session")
def load_models():
    """🔗 Charge les modèles BentoML et récupère les features attendues."""
    try:
        logger.info("🔄 Chargement des modèles BentoML...")
        model_energy_ref = bentoml.sklearn.load_model(ENERGY_MODEL_PATH)
        model_co2_ref = bentoml.sklearn.load_model(CO2_MODEL_PATH)

        features_energy = model_energy_ref.custom_objects["features"]
        features_co2 = model_co2_ref.custom_objects["features"]

        logger.info(f"⚡ Features énergie : {features_energy}")
        logger.info(f"🌿 Features CO₂ : {features_co2}")

        return features_energy, features_co2
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")
        pytest.exit("❌ Arrêt des tests : modèles manquants.", returncode=1)

# ============================================================
# 🌐 Configuration des endpoints API
# ------------------------------------------------------------
# Définition des endpoints de prédiction (énergie et CO₂).
# ============================================================
energy_url = "http://127.0.0.1:3000/predict_energy"
co2_url = "http://127.0.0.1:3000/predict_co2"
headers = {"Content-Type": "application/json"}

# ============================================================
# 🚀 Fonction utilitaire renommée pour éviter les conflits avec pytest
# ------------------------------------------------------------
# 💡 Cette fonction envoie une requête POST et vérifie la réponse.
# ============================================================
def run_endpoint_test(url, payload, label):
    """🚀 Exécute un test sur un endpoint donné et log les résultats."""
    try:
        logger.info(f"🚀 Test de l'endpoint : {label}")
        response = requests.post(url, headers=headers, json=payload)
        assert response.status_code == 200, f"Erreur HTTP: {response.status_code}, {response.text}"
        logger.success(f"✅ Résultat {label} : {response.json()}")
        return response.json()
    except Exception as e:
        logger.error(f"❌ Exception pendant le test {label} : {str(e)}")
        pytest.fail(f"❌ Échec lors du test {label}")

# ============================================================
# 🎯 Tests parametrés avec pytest sur plusieurs échantillons
# ------------------------------------------------------------
# 🏃 Chaque test est exécuté sur les 3 premières lignes du dataset.
# Le nom a été ajusté pour respecter les conventions pytest.
# ============================================================
@pytest.mark.parametrize("index", [0, 1, 2])
def test_energy_prediction(load_data, load_models, index):
    """🔋 Teste l'endpoint de prédiction de la consommation d'énergie."""
    data_energy, _ = load_data
    features_energy, _ = load_models

    sample = data_energy[features_energy].iloc[index].tolist()
    response = run_endpoint_test(energy_url, {"features": sample}, "🔋 Consommation d'énergie")
    assert "site_energy_use" in response, "❌ Clé 'site_energy_use' manquante dans la réponse."

@pytest.mark.parametrize("index", [0, 1, 2])
def test_co2_prediction(load_data, load_models, index):
    """🌿 Teste l'endpoint de prédiction des émissions de CO₂."""
    _, data_co2 = load_data
    _, features_co2 = load_models

    sample = data_co2[features_co2].iloc[index].tolist()
    response = run_endpoint_test(co2_url, {"features": sample}, "🌿 Émissions de CO₂")
    assert "ghg_emissions_total" in response, "❌ Clé 'ghg_emissions_total' manquante dans la réponse."

# ============================================================
# 🎉 Instructions pour exécuter les tests :
#     ➔ pytest tests/test_api.py
#     ➔ pytest --maxfail=1 --disable-warnings -v
# ------------------------------------------------------------
# 📝 Ce script s'assure que l'API retourne les prédictions attendues
#     et que les endpoints sont accessibles et fonctionnels.
# ============================================================
