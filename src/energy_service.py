# ============================================================
# ⚡ **API BentoML : Service Consommation d'Énergie**
# 📚 **Commenté, structuré, avec port explicitement configuré**
# ============================================================

# ============================================================
# 🔧 **Service : Consommation d'Énergie (src/energy_service.py)**
# ------------------------------------------------------------
# 🎯 **Objectif :** Prédire la consommation énergétique (kBtu)
# 📌 **Rôle :** Estimer la consommation en temps réel selon les
#     caractéristiques du bâtiment.
# ✅ **Port dédié :** 3000 (défini explicitement dans la config ou via CLI)
# ============================================================

# 📦 **Imports nécessaires et leur rôle**
import bentoml  # Framework pour le déploiement rapide de modèles ML
from bentoml.io import JSON  # Gestion des entrées/sorties au format JSON
from pydantic import BaseModel, Field, validator  # Validation robuste des données
import numpy as np  # Manipulation numérique efficace
from loguru import logger  # Gestion avancée et lisible des logs
from src.config import ENERGY_SERVICE_PORT  # Port d'écoute configuré

# 🔗 **Chargement du modèle Énergie depuis BentoML**
logger.info(f"🔄 Chargement du modèle Énergie sur le port {ENERGY_SERVICE_PORT}...")
model_energy_ref = bentoml.sklearn.get("site_energy_use_model:latest")  # Dernier modèle Énergie enregistré
features_energy = model_energy_ref.custom_objects.get("features", [])  # Features utilisées lors de l'entraînement
logger.info(f"📋 Features Énergie : {features_energy}")
energy_runner = model_energy_ref.to_runner()  # Création du runner pour exécution asynchrone
logger.info("⚡ Runner Énergie configuré.")
logger.info(f"🚀 ✅ Service Énergie disponible sur http://127.0.0.1:{ENERGY_SERVICE_PORT}")

# 🌐 **Définition du service BentoML (port explicitement configuré)**
energy_prediction_service = bentoml.Service(
    name="energy_prediction_service",
    runners=[energy_runner]
)

# 📜 **Validation des données entrantes pour l'Énergie**
class EnergyInputData(BaseModel):
    """
    📄 **Description :**
    - Reçoit une liste de valeurs numériques correspondant aux features du modèle Énergie.
    - Valide la longueur de la liste pour correspondre aux features attendues.

    💡 **Exemple JSON attendu :**
    ```json
    {
        "features": [0.1, 0.5, 2, 1995, 1, 65.0, 0, 120.0, 1, 0]
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
    - 🔄 Transforme les données entrantes en tableau numpy.
    - 🏃 Exécute la prédiction via le runner.
    - 🌟 Retourne la prédiction sous forme JSON.

    💡 **Exemple JSON attendu :**
    ```json
    {
        "features": [0.1, 0.5, 2, 1995, 1, 65.0, 0, 120.0, 1, 0]
    }
    ```
    📝 **Réponse JSON exemple :**
    ```json
    {
        "site_energy_use": 135000.50
    }
    ```
    """
    try:
        logger.info("🔍 Prédiction Énergie en cours...")
        input_features = np.array(data.features).reshape(1, -1)
        energy_pred = await energy_runner.predict.async_run(input_features)
        logger.info(f"⚡ Résultat : {energy_pred[0]:.2f} kBtu")
        return {"site_energy_use": float(energy_pred[0])}
    except Exception as e:
        logger.error(f"❌ Erreur Énergie : {str(e)}")
        return {"error": str(e)}

# ============================================================
# 🏃 **Commandes d'exécution locale (port explicitement défini)**
# ------------------------------------------------------------
# ⚡ Énergie :
#    ➔ bentoml serve src.energy_service:energy_prediction_service --reload --port 3000
# 🌐 Swagger UI :
#    ➔ Énergie : http://127.0.0.1:3000
# ============================================================

if __name__ == "__main__":
    logger.info(f"🚀 ✅ Service API Énergie prêt et opérationnel sur le port {ENERGY_SERVICE_PORT}.")
    logger.info("📝 Lancement manuel : bentoml serve src.energy_service:energy_prediction_service --reload --port 3000")

