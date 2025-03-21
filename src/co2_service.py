# ============================================================
# 🌿 **API BentoML : Service Émissions de CO₂**
# 📚 **Commenté, structuré, avec port explicitement configuré**
# ============================================================

# ============================================================
# 🔧 **Service : Émissions de CO₂ (src/co2_service.py)**
# ------------------------------------------------------------
# 🎯 **Objectif :** Prédire les émissions de CO₂ (tonnes)
# 📌 **Rôle :** Estimer les émissions en temps réel selon les
#     caractéristiques du bâtiment.
# ✅ **Port dédié :** 3001 (défini explicitement dans la config ou via CLI)
# ============================================================

# 📦 **Imports nécessaires et leur rôle**
import bentoml  # Framework pour le déploiement rapide de modèles ML
from bentoml.io import JSON  # Gestion des entrées/sorties au format JSON
from pydantic import BaseModel, Field, validator  # Validation robuste des données
import numpy as np  # Manipulation numérique efficace
from loguru import logger  # Gestion avancée et lisible des logs
from src.config import CO2_SERVICE_PORT  # Port d'écoute configuré

# 🔗 **Chargement du modèle CO₂ depuis BentoML**
logger.info(f"🔄 Chargement du modèle CO₂ sur le port {CO2_SERVICE_PORT}...")
model_co2_ref = bentoml.sklearn.get("ghg_emissions_model:latest")  # Dernier modèle CO₂ enregistré
features_co2 = model_co2_ref.custom_objects.get("features", [])  # Features utilisées lors de l'entraînement
logger.info(f"📋 Features CO₂ : {features_co2}")
co2_runner = model_co2_ref.to_runner()  # Création du runner pour exécution asynchrone
logger.info("🌿 Runner CO₂ configuré.")
logger.info(f"🚀 ✅ Service CO₂ disponible sur http://127.0.0.1:{CO2_SERVICE_PORT}")

# 🌐 **Définition du service BentoML (port explicitement configuré)**
co2_prediction_service = bentoml.Service(
    name="co2_prediction_service",
    runners=[co2_runner]
)

# 📜 **Validation des données entrantes pour le CO₂**
class CO2InputData(BaseModel):
    """
    📄 **Description :**
    - Reçoit une liste de valeurs numériques correspondant aux features du modèle CO₂.
    - Valide la longueur de la liste pour correspondre aux features attendues.

    💡 **Exemple JSON attendu :**
    ```json
    {
        "features": [120000.0, 0.6, 0.4, 3, 2010, 1, 75.0, 0, 150.0, 1]
    }
    ```
    🔍 **Rôle de chaque feature :**
    1. `site_energy_use` : Consommation énergétique totale du bâtiment (kBtu).
    2. `electricity_ratio` : Proportion d'électricité dans la consommation totale.
    3. `gas_ratio` : Proportion de gaz dans la consommation totale.
    4. `floors_cat` : Catégorie du nombre d'étages.
    5. `year_built_cat` : Catégorie de l'année de construction.
    6. `f_is_large_building` : Indicateur binaire si le bâtiment est de grande taille.
    7. `site_eui` : Intensité d'utilisation énergétique du site (kBtu/pi²).
    8. `property_use_list_office` : Indicateur pour usage de type bureau.
    9. `building_density` : Densité du bâtiment (surface/volume).
    10. `property_type_office` : Indicateur pour propriété de type bureau.
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
    - 🔄 Transforme les données entrantes en tableau numpy.
    - 🏃 Exécute la prédiction via le runner.
    - 🌟 Retourne la prédiction sous forme JSON.

    💡 **Exemple JSON attendu :**
    ```json
    {
        "features": [120000.0, 0.6, 0.4, 3, 2010, 1, 75.0, 0, 150.0, 1]
    }
    ```
    📝 **Réponse JSON exemple :**
    ```json
    {
        "ghg_emissions_total": 250.75
    }
    ```
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
# 🏃 **Commandes d'exécution locale (port explicitement défini)**
# ------------------------------------------------------------
# 🌿 CO₂ :
#    ➔ bentoml serve src.co2_service:co2_prediction_service --reload --port 3001
# 🌐 Swagger UI :
#    ➔ CO₂ : http://127.0.0.1:3001
# ============================================================

if __name__ == "__main__":
    logger.info(f"🚀 ✅ Service API CO₂ prêt et opérationnel sur le port {CO2_SERVICE_PORT}.")
    logger.info("📝 Lancement manuel : bentoml serve src.co2_service:co2_prediction_service --reload --port 3001")
