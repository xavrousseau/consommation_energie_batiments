# ============================================================
# 🌿 **API BentoML : Service Émissions de CO₂**
# 📚 **Entièrement commenté, structuré et pédagogique**
# ============================================================

# ============================================================
# 🔧 **Service : Émissions de CO₂ (src/co2_service.py)**
# ------------------------------------------------------------
# 🎯 **Objectif :** Prédire les émissions de CO₂ (tonnes)
# 📌 **Rôle :** Estimer les émissions en temps réel à partir des
#     caractéristiques du bâtiment.
# ============================================================

# 📦 **Imports nécessaires et leur rôle**
import bentoml  # Framework pour le déploiement rapide de modèles ML
from bentoml.io import JSON  # Gestion des entrées/sorties au format JSON
from pydantic import BaseModel, Field, validator  # Validation robuste des données
import numpy as np  # Manipulation numérique efficace
from loguru import logger  # Gestion avancée et lisible des logs
from src.config import CO2_SERVICE_PORT  # Configuration du port d'écoute

# 🔗 **Chargement du modèle CO₂ depuis BentoML**
logger.info(f"🔄 Chargement du modèle CO₂ sur le port {CO2_SERVICE_PORT}...")
model_co2_ref = bentoml.sklearn.get("ghg_emissions_model:latest")  # Dernier modèle CO₂ enregistré
features_co2 = model_co2_ref.custom_objects.get("features", [])  # Features utilisées lors de l'entraînement
logger.info(f"📋 Features CO₂ : {features_co2}")
co2_runner = model_co2_ref.to_runner()  # Création du runner pour exécution asynchrone
logger.info("🌿 Runner CO₂ configuré.")

# 🌐 **Définition du service BentoML (sans argument 'port' pour compatibilité)**
co2_prediction_service = bentoml.Service(
    name="co2_prediction_service",
    runners=[co2_runner]
)

# 📜 **Validation des données entrantes pour le CO₂**
class CO2InputData(BaseModel):
    """
    📄 **Description :**
    - Reçoit une liste de valeurs numériques.
    - Valide la longueur de la liste pour correspondre aux features attendues.
    """
    features: list[float] = Field(..., description=f"{len(features_co2)} valeurs attendues.")

    @validator('features')
    def check_length(cls, v):
        if len(v) != len(features_co2):
            raise ValueError(f"❌ {len(features_co2)} attendues, {len(v)} reçues.")
        return v

# ✨ **Endpoint principal pour la prédiction des émissions de CO₂**
@co2_prediction_service.api(input=JSON(pydantic_model=CO2InputData), output=JSON())
async def predict_co2(data: CO2InputData):
    """
    🌿 **Endpoint :** `/predict_co2`
    - 🔍 Transforme les données entrantes en tableau numpy.
    - 🏃 Exécute la prédiction via le runner.
    - 🌟 Retourne la prédiction sous forme JSON.
    - 💡 **Exemple JSON attendu :** { "features": [0.2, 0.4, ...] }
    """
    try:
        logger.info("🔍 Prédiction CO₂ en cours...")
        input_features = np.array(data.features).reshape(1, -1)
        co2_pred = await co2_runner.predict.async_run(input_features)
        logger.info(f"🌿 Résultat CO₂ : {co2_pred[0]:.2f} tonnes.")
        return {"ghg_emissions_total": float(co2_pred[0])}
    except Exception as e:
        logger.error(f"❌ Erreur CO₂ : {str(e)}")
        return {"error": str(e)}

# ============================================================
# 🏃 **Commande d'exécution locale**
# ------------------------------------------------------------
# 🌿 CO₂ :
#    ➔ bentoml serve src.co2_service:co2_prediction_service --reload
# 🌐 Swagger UI :
#    ➔ CO₂ : http://127.0.0.1:3001
# ============================================================

if __name__ == "__main__":
    logger.info("🚀 ✅ Service API CO₂ prêt et opérationnel sur le port dédié.")
