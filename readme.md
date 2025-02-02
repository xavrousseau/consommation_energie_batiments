### 📌 Préparation des données pour la prédiction de la consommation énergétique (`site_energy_use`)

Ce notebook est dédié à la préparation des données spécifiques à la prédiction de la consommation énergétique des bâtiments (`site_energy_use`). Il suit un pipeline clair, du chargement des données jusqu'à l'exportation des données nettoyées prêtes à être utilisées pour la modélisation.

---

### **1️⃣ Chargement des données et réduction du périmètre**
- 1.1 Définition des répertoires et configuration des logs
- 1.2 Importation du dataset
- 1.3 Filtrage des bâtiments non résidentiels à Seattle
- 1.4 Suppression des variables non pertinentes pour `site_energy_use`

### **2️⃣ Exploration des données**
- 2.1 Aperçu du dataset (dimensions, types de variables, valeurs uniques, etc.)
- 2.2 Vérification de la cohérence des données (valeurs aberrantes, erreurs typographiques, formats incorrects)
- 2.3 Résumé statistique
- 2.4 Harmonisation des noms de colonnes

3️⃣ Nettoyage et préparation des données
3.1 Suppression des doublons
3.2 Analyse des valeurs manquantes
3.3 Suppression des colonnes avec trop de valeurs manquantes
3.4 Analyse des corrélations avec site_energy_use 🔄 (Ajouté ici)
3.5 Imputation des valeurs manquantes en tenant compte des corrélations avec site_energy_use (Utilisation des résultats de la 3.4 pour ajuster l’imputation)

4️⃣ Analyse exploratoire avancée (EDA)
4.1 Distribution de site_energy_use (Peut aider à détecter des outliers)
4.2 Sélection des variables explicatives pertinentes (Maintenant qu’on a vu la corrélation et l’impact des valeurs manquantes, on peut affiner le choix des variables finales)

### **5️⃣ Transformation et préparation finale des données**
- 5.1 Encodage des variables catégoriques
- 5.2 Normalisation/standardisation des variables numériques
- 5.3 Détection et gestion des outliers

### **6️⃣ Sauvegarde des données préparées**
- 6.1 Export du dataset nettoyé pour la modélisation

---