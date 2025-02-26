# ============================================================
# 1️⃣ Configuration initiale et importations
# ============================================================
# 📦 Script : preprocess_data_for_models.py
#     - Nettoyage et transformation des données pour :
#       ✅ GHG Emissions Total
#       ✅ Site Energy Use
#     - Chemins et logs via config.py
#     - Étapes commentées par sections numérotées
#     - 🔍 **Astuce** : `RAW_DATA_PATH` correspond au chemin du fichier CSV brut
#       initial, contenant toutes les données sources à transformer.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from loguru import logger
from config import (
    RAW_DATA_PATH,  # 🔍 Fichier CSV brut initial importé pour transformation
    PROCESSED_ENERGY_PATH,
    PROCESSED_CO2_PATH,
    RANDOM_STATE
)

# ============================================================
# 2️⃣ Chargement des données initiales
# ============================================================
logger.info(f"📂 Chargement des données depuis : {RAW_DATA_PATH} (fichier brut initial)")
try:
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info("✅ Données initiales chargées avec succès.")
except FileNotFoundError:
    logger.error("❌ Fichier introuvable. Vérifiez le chemin dans config.py.")
    raise

# ============================================================
# 3️⃣ Renommage des colonnes principales
# ============================================================
logger.info("🔄 Renommage des colonnes nécessaires...")
columns_mapping = {
    "SiteEnergyUse(kBtu)": "site_energy_use",
    "Electricity(kWh)": "electricity_kwh",
    "Electricity(kBtu)": "electricity_kbtu",
    "NaturalGas(kBtu)": "natural_gas_kbtu",
    "SiteEUI(kBtu/sf)": "site_eui",
    "PropertyGFATotal": "gfa_total",
    "NumberofFloors": "num_floors",
    "YearBuilt": "year_built"
}
df.rename(columns=columns_mapping, inplace=True)
logger.info("✏️ Colonnes renommées avec succès.")

# ============================================================
# 4️⃣ Imputation des valeurs manquantes
# ============================================================
logger.info("🩹 Imputation des valeurs manquantes...")
medians = {
    'site_energy_use': 2554947.25,
    'electricity_kwh': 472415.34,
    'electricity_kbtu': 1611881.0,
    'natural_gas_kbtu': 498263.0,
    'num_floors': 3,
    'year_built': 1977,
    'site_eui': 51.90,
    'gfa_total': df["gfa_total"].median()
}
for col, median_val in medians.items():
    df[col] = df[col].fillna(median_val)
    logger.info(f"🩹 '{col}' imputée avec médiane : {median_val}.")

# ============================================================
# 5️⃣ Traitement des valeurs aberrantes (Winsorizing)
# ============================================================
logger.info("📉 Traitement des valeurs aberrantes (Winsorizing)...")
def winsorize(series, lower=0.01, upper=0.99):
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    logger.info(f"📊 '{series.name}' : bornes [{lower_bound}, {upper_bound}]")
    return series.clip(lower=lower_bound, upper=upper_bound)

cols_to_winsorize = ["site_energy_use", "electricity_kwh", "electricity_kbtu", "natural_gas_kbtu", "num_floors", "site_eui", "gfa_total"]
for col in cols_to_winsorize:
    df[col] = winsorize(df[col])
    logger.info(f"✅ '{col}' : valeurs aberrantes traitées.")

# ============================================================
# 6️⃣ Création des variables dérivées (features clés)
# ============================================================
logger.info("🧮 Création des variables dérivées...")
df['electricity_ratio'] = df['electricity_kbtu'] / (df['site_energy_use'] + 1e-9)
logger.info("⚡ 'electricity_ratio' (électricité/énergie) créée.")
df['gas_ratio'] = df['natural_gas_kbtu'] / (df['site_energy_use'] + 1e-9)
logger.info("🔥 'gas_ratio' (gaz/énergie) créée.")
df["f_is_large_building"] = np.where(df["gfa_total"] > 100000, 1, 0)
logger.info("🏢 'f_is_large_building' (>100000 sq ft) créée.")
df["building_density"] = df["gfa_total"] / (df["site_energy_use"] + 1e-9)
logger.info("🏗️ 'building_density' (surface/énergie) créée.")

# ============================================================
# 7️⃣ Transformation catégorielle : num_floors
# ============================================================
logger.info("🏗️ Transformation catégorielle de 'num_floors'...")
conditions_floors = [
    (df["num_floors"] <= 4),
    (df["num_floors"] > 4) & (df["num_floors"] <= 8),
    (df["num_floors"] > 8)
]
values_floors = ["0-4 étages", "5-8 étages", "8+ étages"]
df["floors_cat"] = np.select(conditions_floors, values_floors)
encoder_floors = OrdinalEncoder(categories=[["0-4 étages", "5-8 étages", "8+ étages"]])
df["floors_cat"] = encoder_floors.fit_transform(df[["floors_cat"]]).astype(int) + 1
logger.info("🔢 'floors_cat' encodée : 1='0-4', 2='5-8', 3='8+'.")

# ============================================================
# 8️⃣ Transformation catégorielle : year_built
# ============================================================
logger.info("🏡 Transformation catégorielle de 'year_built'...")
def year_built_category(year):
    if year <= 1960:
        return '1900-1960'
    elif year <= 1976:
        return '1961-1976'
    elif year <= 1980:
        return '1977-1980'
    elif year <= 1994:
        return '1981-1994'
    else:
        return '1995-2015'

df['year_built_cat'] = df['year_built'].apply(year_built_category)
encoder_year = OrdinalEncoder(categories=[["1900-1960", "1961-1976", "1977-1980", "1981-1994", "1995-2015"]])
df['year_built_cat'] = encoder_year.fit_transform(df[['year_built_cat']]).astype(int) + 1
logger.info("🔢 'year_built_cat' encodée : 1='1900-1960', ..., 5='1995-2015'.")

# ============================================================
# 9️⃣ Standardisation des variables continues
# ============================================================
logger.info("📏 Standardisation des variables continues...")
scaler = StandardScaler()
continuous_cols = ["site_energy_use", "electricity_ratio", "gas_ratio", "site_eui", "building_density"]
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
for col in continuous_cols:
    logger.info(f"✅ '{col}' standardisée.")

# ============================================================
# 🔟 Export des datasets finaux pour les modèles
# ============================================================
logger.info("💾 Sauvegarde des datasets finaux...")
final_columns_ghg = ["site_energy_use", "electricity_ratio", "gas_ratio", "floors_cat", "year_built_cat"]
final_columns_energy = ["site_eui", "f_is_large_building", "floors_cat", "building_density", "gas_ratio"]

df[final_columns_ghg].to_csv(PROCESSED_CO2_PATH, index=False)
logger.info(f"✅ Dataset GHG exporté : {PROCESSED_CO2_PATH}")

df[final_columns_energy].to_csv(PROCESSED_ENERGY_PATH, index=False)
logger.info(f"✅ Dataset Site Energy Use exporté : {PROCESSED_ENERGY_PATH}")

# ============================================================
# 🎉 1️⃣1️⃣ Fin du prétraitement : Données prêtes pour modélisation
# ============================================================
logger.info("🚀 Prétraitement terminé avec succès. Les datasets sont prêts pour l'entraînement des modèles.")
