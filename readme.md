# 📊 Préparation des données pour la prédiction de la consommation énergétique (`site_energy_use`)

Ce notebook est dédié à la **préparation des données** en vue de prédire la consommation énergétique des bâtiments (`site_energy_use`).  
Il suit un pipeline clair et structuré : **chargement, exploration, nettoyage, transformation** et **export final** des données prêtes à être modélisées.

---

## 1️⃣ Chargement des données et réduction du périmètre

- **1.1** Définition des répertoires et configuration des logs  
- **1.2** Importation du dataset d’origine  
- **1.3** Filtrage : sélection des bâtiments **non résidentiels situés à Seattle**  
- **1.4** Suppression des variables non pertinentes pour la variable cible `site_energy_use`  

---

## 2️⃣ Exploration des données

- **2.1** Aperçu du dataset : dimensions, types, valeurs uniques  
- **2.2** Vérification de la cohérence des données :  
  - Valeurs aberrantes  
  - Erreurs typographiques  
  - Formats incorrects  
- **2.3** Résumé statistique  
- **2.4** Harmonisation des noms de colonnes  

---

## 3️⃣ Nettoyage et préparation des données

- **3.1** Suppression des doublons  
- **3.2** Analyse des valeurs manquantes  
- **3.3** Suppression des colonnes trop incomplètes  
- **3.4** Analyse des corrélations avec `site_energy_use`  
- **3.5** Imputation des valeurs manquantes **guidée par les corrélations**

---

## 4️⃣ Analyse exploratoire avancée (EDA)

- **4.1** Analyse de la distribution de `site_energy_use` (détection d’outliers)  
- **4.2** Sélection des variables explicatives les plus pertinentes (post-analyse des corrélations et valeurs manquantes)  

---

## 5️⃣ Transformation et préparation finale

- **5.1** Encodage des variables catégoriques  
- **5.2** Normalisation / standardisation des variables numériques  
- **5.3** Détection et traitement des outliers  

---

## 6️⃣ Sauvegarde du jeu de données préparé

- **6.1** Export du dataset final nettoyé, prêt pour la phase de modélisation  

---

## ✅ Résultat attendu

Un fichier de données propre, complet, et prêt à être injecté dans un modèle de machine learning pour prédire efficacement `site_energy_use`.
