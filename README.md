# 💻 Laptop Market Price Estimator

> **Projet Data Science complet** : Prédire le prix de marché d'un laptop à partir de ses spécifications techniques avec CatBoost + SHAP + Streamlit

## 🎯 Objectif

Développer un pipeline complet de Machine Learning qui :

1. **Analyse** les données d'un dataset de ~100 000 ordinateurs
2. **Prédit** le prix de marché d'un laptop avec une précision de ~85%
3. **Explique** les prédictions grâce à SHAP (explainabilité)
4. **Visualise** les tendances du marché via une app Streamlit interactive

---

## 🛠️ Stack Technique

| Composant            | Technologie                          |
| -------------------- | ------------------------------------ |
| Machine Learning     | CatBoostRegressor                    |
| Explainabilité       | SHAP (SHapley Additive exPlanations) |
| Manipulation données | Pandas, NumPy                        |
| Visualisation EDA    | Matplotlib, Seaborn                  |
| Visualisation App    | Plotly                               |
| Application Web      | Streamlit                            |
| Sérialisation        | Joblib                               |

---

## 📁 Structure du Projet

```
laptop_price_estimator/
│
├── 📂 data/
│   └── computer_prices_all.csv, computer_prices_test.csv     # Dataset Kaggle (~100K lignes)
│
├── 📂 notebooks/
│   └── Prédiction_Prix_ordinateurs.ipynb       # EDA + Feature Engineering + CatBoost + SHAP
│
├── 📂 models/
│   ├── laptop_price_model.pkl       # Modèle entraîné + métadonnées
│   ├── plot_price_distribution.png  # Graphiques EDA générés
│   ├── plot_shap_summary.png
│   └── ...
│
├── app.py                           # Application Streamlit
├── requirements.txt                 # Dépendances Python
└── README.md                        # Documentation
```

---

## 🚀 Installation & Lancement

### 1. Cloner le projet

```bash
git clone https://github.com/Magloireaffokpe/Data-Science--Prediction-Prix-Ordinateur.git
cd Data-Science--Prediction-Prix-Ordinateur
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Préparer les données

Téléchargez le dataset depuis Kaggle :

```
https://www.kaggle.com/competitions/computer-prices-2025/data
```

Placez `computer_prices_all.csv` dans le dossier `data/`.

### 5. Entraîner le modèle

```bash
jupyter notebook notebooks/Prédiction_Prix_ordinateurs.ipynb
# Exécutez toutes les cellules (Kernel > Restart & Run All)
```

### 6. Lancer l'application

```bash
streamlit run app.py
```

L'application sera disponible sur `http://localhost:8501`

---

## 📊 Dataset

| Attribut | Valeur                        |
| -------- | ----------------------------- |
| Source   | Kaggle - Computer Prices 2025 |
| Taille   | ~100 000 lignes               |
| Colonnes | 33 features                   |
| Cible    | `price` (USD)                 |
| Filtre   | `device_type == "Laptop"`     |

### Variables principales

**Appareil** : `brand`, `release_year`, `os`, `form_factor`  
**CPU** : `cpu_brand`, `cpu_tier`, `cpu_cores`, `cpu_threads`, `cpu_base_ghz`, `cpu_boost_ghz`  
**GPU** : `gpu_brand`, `gpu_tier`, `vram_gb`  
**Mémoire** : `ram_gb`, `storage_type`, `storage_gb`, `storage_drive_count`  
**Écran** : `display_type`, `display_size_in`, `resolution`, `refresh_hz`  
**Autres** : `battery_wh`, `charger_watts`, `wifi`, `bluetooth`, `weight_kg`, `warranty_months`

---

## ⚙️ Feature Engineering

4 variables composites sont créées pour capturer mieux la puissance des composants :

| Feature             | Formule                            | Interprétation                |
| ------------------- | ---------------------------------- | ----------------------------- |
| `cpu_power`         | `cpu_cores × cpu_boost_ghz`        | Puissance CPU effective       |
| `gpu_power`         | `gpu_tier × vram_gb`               | Puissance graphique           |
| `ppi`               | `√(x²+y²) / display_size_in`       | Densité d'affichage (netteté) |
| `storage_per_drive` | `storage_gb / storage_drive_count` | Configuration stockage        |

---

## 🤖 Modèle CatBoost

### Pourquoi CatBoost ?

- ✅ Gestion **native des variables catégorielles** (pas d'encodage manuel)
- ✅ **Très performant** sur les données tabulaires (top Kaggle)
- ✅ **Résistant à l'overfitting** grâce à l'ordered boosting
- ✅ **Rapide** même sur de grands datasets

### Hyperparamètres

```python
CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    loss_function='RMSE',
    early_stopping_rounds=100,
    random_seed=42
)
```

### Métriques (Test Set)

| Métrique | Valeur |
| -------- | ------ |
| R² Score | ~0.85  |
| MAE      | ~$150  |
| RMSE     | ~$250  |
| MAPE     | ~12%   |

> _Les métriques exactes dépendent du dataset et de la version des données._

---

## 🔮 Explainabilité SHAP

SHAP (SHapley Additive exPlanations) permet de comprendre **pourquoi** le modèle prédit un certain prix :

- **SHAP Summary Plot** : vision globale de l'importance et direction des features
- **SHAP Bar Plot** : importance moyenne par feature
- **SHAP Waterfall** : explication détaillée d'une prédiction spécifique

### Interprétation des graphiques SHAP

- 🔴 **Rouge** = valeur haute de la feature → augmente le prix
- 🔵 **Bleu** = valeur basse de la feature → diminue le prix
- Position sur X = impact sur la prédiction

---

## 🌐 Application Streamlit

L'application contient **4 sections** :

### 🏠 Accueil

- Présentation du projet
- Stack technique
- Gauge de précision du modèle

### 🧮 Price Estimator

- Interface avec sliders et dropdowns pour chaque spec
- Prédiction instantanée en USD et FCFA
- Graphiques : distribution des prix, radar performance, score global

### 📊 Market Insights

- Analyses par marque, RAM, GPU tier
- Évolution des prix par année
- Scatter plots interactifs

### 🔮 Model Insights

- Feature importance CatBoost
- Graphiques SHAP (global + local)
- Métriques de performance détaillées

---

## 💱 Conversion USD → FCFA

```
1 USD ≈ 600 FCFA
```

Tous les prix sont affichés en USD et automatiquement convertis en FCFA dans l'interface.

---

## 🗂️ Colonnes Supprimées

| Colonne     | Raison                                         |
| ----------- | ---------------------------------------------- |
| `ID`        | Identifiant unique, aucun signal               |
| `model`     | Cardinalité trop haute, risque de leakage      |
| `psu_watts` | Non pertinent pour laptops (concerne desktops) |

---

## 📝 Auteur

**Portfolio Data Science** - Laptop Market Price Estimator  
_Projet complet : EDA → Feature Engineering → ML → Interprétabilité → App déployée_

---

## 📄 Licence

MIT License - Libre d'utilisation

<!-- nouveau-pc:debut -->
## Installer sur un nouveau PC

**Prérequis :** Git et Python 3.12.

```bash
git clone https://github.com/Magloireaffokpe/Data-Science--Prediction-Prix-Ordinateur.git
cd Data-Science--Prediction-Prix-Ordinateur
python3 -m venv env
source env/bin/activate   # Windows : env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**À recopier à la main (hors Git) :** `data/` et `models/` (environ 60 Mo). Sans eux, il faut préparer les données et relancer le notebook d'entraînement avant de démarrer l'application.
<!-- nouveau-pc:fin -->
