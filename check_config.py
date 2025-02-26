# ============================================================
# 📂 Script rapide de vérification pour config.py
# ============================================================

import sys
from pathlib import Path

# 🔗 Ajout du chemin src/ pour que Python trouve config.py
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from config import (
    BASE_DIR, RAW_DATA_PATH, PROCESSED_ENERGY_PATH, PROCESSED_CO2_PATH,
    ENERGY_MODEL_PATH, ENERGY_FEATURES_PATH, CO2_MODEL_PATH, CO2_FEATURES_PATH, LOGS_DIR
)

def check_path(path, description):
    if path.exists():
        print(f"✅ {description} : {path}")
    else:
        print(f"❌ {description} INVALIDE : {path} 🚨")

# 🔄 Vérification des chemins
print(f"\n📁 Répertoire de base : {BASE_DIR}")
check_path(RAW_DATA_PATH, "Fichier de données brutes (RAW)")
check_path(PROCESSED_ENERGY_PATH, "Fichier énergie (PROCESSED)")
check_path(PROCESSED_CO2_PATH, "Fichier CO₂ (PROCESSED)")
check_path(ENERGY_MODEL_PATH, "Modèle énergie (PKL)")
check_path(ENERGY_FEATURES_PATH, "Features énergie (PKL)")
check_path(CO2_MODEL_PATH, "Modèle CO₂ (PKL)")
check_path(CO2_FEATURES_PATH, "Features CO₂ (PKL)")
check_path(LOGS_DIR, "Répertoire des logs")
