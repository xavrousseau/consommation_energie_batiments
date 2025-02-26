# ============================================================
# 🚀 Script pour charger et sauvegarder les modèles existants avec BentoML
#     - Consommation d'énergie (site_energy_use)
#     - Émissions de CO₂ (ghg_emissions_total)
# ============================================================

import bentoml
import joblib
from loguru import logger
from config import (
    ENERGY_MODEL_PATH, 
    ENERGY_FEATURES_PATH, 
    CO2_MODEL_PATH, 
    CO2_FEATURES_PATH
)

# ============================================================
# 📂 Chargement des modèles et des features existants
# ============================================================
logger.info("📂 Chargement des modèles et features existants depuis les chemins spécifiés dans config.py...")

try:
    # ✅ Chargement des modèles
    energy_model = joblib.load(ENERGY_MODEL_PATH)
    energy_features = joblib.load(ENERGY_FEATURES_PATH)
    logger.info(f"⚡ Modèle énergie chargé depuis : {ENERGY_MODEL_PATH}")
    logger.info(f"📝 Features énergie chargées depuis : {ENERGY_FEATURES_PATH}")

    co2_model = joblib.load(CO2_MODEL_PATH)
    co2_features = joblib.load(CO2_FEATURES_PATH)
    logger.info(f"🌿 Modèle CO₂ chargé depuis : {CO2_MODEL_PATH}")
    logger.info(f"📝 Features CO₂ chargées depuis : {CO2_FEATURES_PATH}")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles ou des features : {e}")
    raise e

# ============================================================
# 💾 Sauvegarde dans le Model Store BentoML avec custom_objects
# ============================================================
try:
    logger.info("💾 Sauvegarde des modèles dans le Model Store BentoML avec custom_objects...")

    bentoml.sklearn.save_model(
        "site_energy_use_model",
        energy_model,
        custom_objects={"features": energy_features}
    )
    logger.success("✅ Modèle de consommation d'énergie sauvegardé dans BentoML.")

    bentoml.sklearn.save_model(
        "ghg_emissions_model",
        co2_model,
        custom_objects={"features": co2_features}
    )
    logger.success("✅ Modèle des émissions de CO₂ sauvegardé dans BentoML.")

except Exception as e:
    logger.error(f"❌ Échec lors de la sauvegarde des modèles BentoML : {e}")
    raise e

# ============================================================
# 🎉 Résultat final
# ============================================================
logger.success("🚀 Les modèles et leurs features ont été intégrés avec succès dans le Model Store BentoML.")
logger.info("⚡ Ils sont prêts à être utilisés avec le service BentoML (service.py). ✅")
