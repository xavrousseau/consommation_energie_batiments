# ============================================================
# 🚀 API BentoML : Services indépendants avec ports dédiés (BentoML >=1.4.1)
#     - EnergyPredictionService : Consommation d'énergie (port 3000)
#     - CO2PredictionService    : Émissions de CO₂ (port 3001)
# ============================================================

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field, validator
import numpy as np
from loguru import logger
from src.config import ENERGY_MODEL_PATH, CO2_MODEL_PATH

# ============================================================
# 🔗 Chargement des modèles depuis BentoML
# ============================================================
logger.info("🔄 Chargement du modèle énergie depuis BentoML (port 3000)...")
model_energy_ref = bentoml.sklearn.get("site_energy_use_model:latest")
model_energy_runner = model_energy_ref.to_runner()
features_energy = model_energy_ref.custom_objects.get("features", [])
logger.info(f"⚡ Modèle énergie chargé. Features : {features_energy}")

logger.info("🔄 Chargement du modèle CO₂ depuis BentoML (port 3001)...")
model_co2_ref = bentoml.sklearn.get("ghg_emissions_model:latest")
model_co2_runner = model_co2_ref.to_runner()
features_co2 = model_co2_ref.custom_objects.get("features", [])
logger.info(f"🌿 Modèle CO₂ chargé. Features : {features_co2}")

# ============================================================
# 📜 Validation d'entrée avec Pydantic
# ============================================================
class EnergyInputData(BaseModel):
    """🔋 Données pour la prédiction de la consommation d'énergie."""
    features: list[float] = Field(..., description=f"{len(features_energy)} valeurs attendues.")

    @validator('features')
    def check_length(cls, v):
        if len(v) != len(features_energy):
            raise ValueError(f"❌ {len(features_energy)} attendues, {len(v)} reçues.")
        return v

class CO2InputData(BaseModel):
    """🌿 Données pour la prédiction des émissions de CO₂."""
    features: list[float] = Field(..., description=f"{len(features_co2)} valeurs attendues.")

    @validator('features')
    def check_length(cls, v):
        if len(v) != len(features_co2):
            raise ValueError(f"❌ {len(features_co2)} attendues, {len(v)} reçues.")
        return v

# ============================================================
# 🌐 Service 1 : EnergyPredictionService (port 3000)
# ============================================================
@bentoml.service(
    name="energy_prediction_service",
    runners=[model_energy_runner],
    traffic={"port": 3000}  # 🌐 Port explicitement défini
)
class EnergyPredictionService:
    @bentoml.api(input=JSON(pydantic_model=EnergyInputData), output=JSON())
    async def predict_energy(self, data: EnergyInputData):
        """🔋 Endpoint pour prédiction consommation d'énergie."""
        try:
            logger.info("🔍 Prédiction énergie en cours...")
            input_features = np.array(data.features).reshape(1, -1)
            energy_pred = await model_energy_runner.predict.async_run(input_features)
            logger.info(f"⚡ Résultat énergie : {energy_pred[0]:.2f} kBtu")
            return {"site_energy_use": float(energy_pred[0])}
        except Exception as e:
            logger.error(f"❌ Erreur prédiction énergie : {str(e)}")
            return {"error": str(e)}

# ============================================================
# 🌐 Service 2 : CO2PredictionService (port 3001)
# ============================================================
@bentoml.service(
    name="co2_prediction_service",
    runners=[model_co2_runner],
    traffic={"port": 3001}  # 🌐 Port explicitement défini
)
class CO2PredictionService:
    @bentoml.api(input=JSON(pydantic_model=CO2InputData), output=JSON())
    async def predict_co2(self, data: CO2InputData):
        """🌿 Endpoint pour prédiction des émissions de CO₂."""
        try:
            logger.info("🔍 Prédiction CO₂ en cours...")
            input_features = np.array(data.features).reshape(1, -1)
            co2_pred = await model_co2_runner.predict.async_run(input_features)
            logger.info(f"🌿 Résultat CO₂ : {co2_pred[0]:.2f} tonnes de CO₂")
            return {"ghg_emissions_total": float(co2_pred[0])}
        except Exception as e:
            logger.error(f"❌ Erreur prédiction CO₂ : {str(e)}")
            return {"error": str(e)}

# ============================================================
# 🏃 Exécution locale :
#    ➔ bentoml serve src.service:EnergyPredictionService --reload
#    ➔ bentoml serve src.service:CO2PredictionService    --reload
#
# 🌐 Swagger UI :
#    ➔ Énergie : http://127.0.0.1:3000
#    ➔ CO₂     : http://127.0.0.1:3001
# ============================================================
logger.info("🚀 Services API prêts :")
logger.info("🔋 EnergyPredictionService : http://127.0.0.1:3000")
logger.info("🌿 CO2PredictionService    : http://127.0.0.1:3001")
