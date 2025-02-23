import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field, ValidationError
import numpy as np

# ✅ Chargement des modèles depuis BentoML
model_energy_ref = bentoml.sklearn.get("site_energy_use_model:latest")
model_co2_ref = bentoml.sklearn.get("ghg_emissions_model:latest")

model_energy_runner = model_energy_ref.to_runner()
model_co2_runner = model_co2_ref.to_runner()

features_energy = model_energy_ref.custom_objects["features"]
features_co2 = model_co2_ref.custom_objects["features"]

# 📌 Définition de la validation d'entrée avec Pydantic
class EnergyInputData(BaseModel):
    """Modèle d'entrée pour la prédiction de la consommation d'énergie."""
    features: list[float] = Field(..., description=f"Liste de {len(features_energy)} valeurs correspondant aux features du modèle de consommation d'énergie.")

class CO2InputData(BaseModel):
    """Modèle d'entrée pour la prédiction des émissions de CO₂."""
    features: list[float] = Field(..., description=f"Liste de {len(features_co2)} valeurs correspondant aux features du modèle de CO₂.")

# 📡 Création du service BentoML
svc = bentoml.Service(
    "energy_co2_prediction_service",
    runners=[model_energy_runner, model_co2_runner]
)

@svc.api(input=JSON(pydantic_model=EnergyInputData), output=JSON())
def predict_energy(data: EnergyInputData):
    """
    Endpoint de prédiction pour la consommation d'énergie.
    """
    try:
        if len(data.features) != len(features_energy):
            raise ValueError(f"❌ Le nombre de features fournies ({len(data.features)}) ne correspond pas à celui attendu ({len(features_energy)}).")

        input_features = np.array(data.features).reshape(1, -1)
        energy_pred = model_energy_runner.predict.run(input_features)[0]

        return {"site_energy_use": float(energy_pred)}

    except Exception as e:
        return {"error": str(e)}

@svc.api(input=JSON(pydantic_model=CO2InputData), output=JSON())
def predict_co2(data: CO2InputData):
    """
    Endpoint de prédiction pour les émissions de CO₂.
    """
    try:
        if len(data.features) != len(features_co2):
            raise ValueError(f"❌ Le nombre de features fournies ({len(data.features)}) ne correspond pas à celui attendu ({len(features_co2)}).")

        input_features = np.array(data.features).reshape(1, -1)
        co2_pred = model_co2_runner.predict.run(input_features)[0]

        return {"ghg_emissions_total": float(co2_pred)}

    except Exception as e:
        return {"error": str(e)}

print("🚀 API prête : lancez avec 'bentoml serve service.py:svc' et accédez à Swagger UI sur http://127.0.0.1:3000")
print(f"📝 Features attendues pour la consommation d'énergie : {features_energy}")
print(f"📝 Features attendues pour les émissions de CO₂ : {features_co2}")
