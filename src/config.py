# ============================================================
# ⚙️ Configuration centralisée pour le projet BentoML
# ============================================================

from pathlib import Path
import os

# 📦 Répertoire de base du projet (racine)
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 📂 Chemins des données
# ============================================================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "2016_Building_Energy_Benchmarking.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_ENERGY_PATH = PROCESSED_DIR / "dataset_processed_site_energy_use.csv"
PROCESSED_CO2_PATH = PROCESSED_DIR / "dataset_processed_ghg_emissions_total.csv"

# ============================================================
# 📂 Chemins des modèles et features
# ============================================================
MODELS_DIR = BASE_DIR / "models"
ENERGY_MODEL_PATH = MODELS_DIR / "site_energy_use_model" / "final_model_site_energy_use.pkl"
ENERGY_FEATURES_PATH = MODELS_DIR / "site_energy_use_model" / "final_features_site_energy_use.pkl"

CO2_MODEL_PATH = MODELS_DIR / "ghg_emissions_model" / "final_model_ghg_emissions_total.pkl"
CO2_FEATURES_PATH = MODELS_DIR / "ghg_emissions_model" / "final_features_ghg_emissions_total.pkl"

# ============================================================
# ⚡ Paramètres globaux et BentoML (2 ports pour services séparés)
# ============================================================
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
BENTOML_SERVICE_NAME_ENERGY = os.getenv("BENTOML_SERVICE_NAME_ENERGY", "energy_prediction_service")
BENTOML_SERVICE_NAME_CO2 = os.getenv("BENTOML_SERVICE_NAME_CO2", "co2_prediction_service")

# ============================================================
# 🌐 Configuration des logs
# ============================================================
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "project.log"

# ============================================================
# 📝 Informations complémentaires
# ============================================================
# - Tous les chemins sont relatifs pour garantir la portabilité (ex: Docker, GCP)
# - Les variables peuvent être surchargées via des variables d'environnement
# - Gestion des logs centralisée pour un suivi facile
# - Ce fichier doit être importé dans tous les scripts du projet
# - ⚡ Service énergie : http://127.0.0.1:3000
# - 🌿 Service CO₂    : http://127.0.0.1:3001
