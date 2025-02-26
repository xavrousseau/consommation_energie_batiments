# ============================================================
# 🚀 **API BentoML : Service Énergie**
# 📚 **Entièrement commenté, structuré et pédagogique**
# ============================================================

# ============================================================
# 🔧 **Service : Consommation d'Énergie (src/energy_service.py)**
# ------------------------------------------------------------
# 🎯 **Objectif :** Prédire la consommation énergétique (kBtu)
# ✅ **Port dédié :** 3000
# 📌 **Rôle :** Permet à un utilisateur d'obtenir une estimation
#     en temps réel selon les caractéristiques de son bâtiment.
# ============================================================

# 📦 **Imports nécessaires et leur rôle**
import bentoml  # Framework pour le déploiement rapide de modèles ML
from bentoml.io import JSON  # Gestion des entrées/sorties au format JSON
from pydantic import BaseModel, Field, validator  # Validation robuste des données
import numpy as np  # Manipulation numérique efficace
from loguru import logger  # Gestion avancée et lisible des logs
from src.config import ENERGY_SERVICE_PORT  # Configuration du port d'écoute

# 🔗 **Chargement du modèle énergie depuis BentoML**
logger.info(f"🔄 Chargement du modèle énergie sur le port {ENERGY_SERVICE_PORT}...")
model_energy_ref = bentoml.sklearn.get("site_energy_use_model:latest")  # Dernier modèle enregistré
features_energy = model_energy_ref.custom_objects.get("features", [])  # Features utilisées lors de l'entraînement
logger.info(f"📋 Features énergie : {features_energy}")
energy_runner = model_energy_ref.to_runner()  # Création du runner pour exécution asynchrone
logger.info("⚡ Runner énergie configuré.")

# 🌐 **Définition du service BentoML (port 3000)**
energy_prediction_service = bentoml.Service(
    name="energy_prediction_service",
    runners=[energy_runner],
    port=ENERGY_SERVICE_PORT
)

# 📜 **Validation des données entrantes pour l'énergie**
class EnergyInputData(BaseModel):
    """
    📄 **Description :**
    - Reçoit une liste de valeurs numériques.
    - Valide la longueur de la liste pour correspondre aux features attendues.
    """
    features: list[float] = Field(..., description=f"{len(features_energy)} valeurs attendues.")

    @validator('features')
    def check_length(cls, v):
        if len(v) != len(features_energy):
            raise ValueError(f"❌ {len(features_energy)} attendues, {len(v)} reçues.")
        return v

# ✨ **Endpoint principal pour la prédiction énergétique**
@energy_prediction_service.api(input=JSON(pydantic_model=EnergyInputData), output=JSON())
async def predict_energy(data: EnergyInputData):
    """
    ⚡ **Endpoint :** `/predict_energy`
    - 🔍 Transforme les données entrantes en tableau numpy.
    - 🏃 Exécute la prédiction via le runner.
    - 🌟 Retourne la prédiction sous forme JSON.
    - 💡 **Exemple JSON attendu :** { "features": [0.1, 0.5, ...] }
    """
    try:
        logger.info("🔍 Prédiction énergie en cours...")
        input_features = np.array(data.features).reshape(1, -1)
        energy_pred = await energy_runner.predict.async_run(input_features)
        logger.info(f"⚡ Résultat : {energy_pred[0]:.2f} kBtu")
        return {"site_energy_use": float(energy_pred[0])}
    except Exception as e:
        logger.error(f"❌ Erreur énergie : {str(e)}")
        return {"error": str(e)}

# ============================================================
# 🏃 **Commande d'exécution locale**
# ------------------------------------------------------------
# 🔌 Énergie :
#    ➔ bentoml serve src.energy_service:energy_prediction_service --reload
# 🌐 Swagger UI :
#    ➔ Énergie : http://127.0.0.1:3000
# ============================================================

if __name__ == "__main__":
    logger.info("🚀 ✅ Service API Énergie prêt et opérationnel sur le port dédié.")
