# 🏎️ F1 Prediction App

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-prediction-2025.streamlit.app)

> 🏆 Application de Machine Learning pour prédire les résultats des courses de Formule 1

---

## 🌐 Démonstration en Ligne

### 👉 [Accéder à l'Application](https://f1-prediction-2025.streamlit.app) 👈

| Fonctionnalité | Description |
|----------------|-------------|
| 🏆 **Podium** | Prédiction du podium avec probabilités de victoire |
| ⏱️ **Temps de Course** | Estimation du temps total de course par pilote |
| 🏢 **Performance Équipe** | Classification des constructeurs (Top, Mid, Back) |
| 📊 **Classements** | Standings pilotes et constructeurs |
| 📅 **Calendrier** | Calendrier complet F1 2025 |

---

## 📋 Table des Matières

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Technologies](#-technologies)
- [Les Modèles ML](#-les-modèles-ml)
- [Structure du Projet](#-structure-du-projet)
- [Installation Locale](#-installation-locale)
- [Utilisation](#-utilisation)
- [Dataset](#-dataset)
- [Auteurs](#-auteurs)
- [Améliorations Futures](#-améliorations-futures)
- [Licence](#-licence)

---

## 🎯 Aperçu

**F1 Prediction App** est une application web interactive qui utilise le Machine Learning pour prédire les résultats des courses de Formule 1.

Ce projet a été développé en binôme dans le cadre de notre formation en **2ème année de cycle d'ingénieur spécialisé en Intelligence Artificielle**.

### 🎯 Objectifs du projet

- Prédire les probabilités de victoire de chaque pilote
- Estimer les temps de course
- Classifier les équipes selon leur performance
- Fournir une interface utilisateur intuitive et interactive

### 💡 Ce qui rend ce projet unique

- Utilisation du modèle probabiliste **Plackett-Luce** pour des probabilités cohérentes
- Comparaison automatique d'algorithmes avec **GridSearchCV**
- Interface moderne avec visualisations interactives **Plotly**
- Données réelles de la saison F1 2025

---

## ✨ Fonctionnalités

### 🏆 Prédiction de Victoire

| Aspect | Détail |
|--------|--------|
| **Modèle** | Ridge Regression + Plackett-Luce |
| **Output** | Probabilités de victoire (somme = 100%) |
| **Visualisation** | Podium interactif + graphique des probabilités |

- Calcul d'un score de "force" (θ) pour chaque pilote
- Conversion en probabilités via le modèle Plackett-Luce
- Affichage du podium prédit avec pourcentages

### ⏱️ Prédiction du Temps de Course

| Aspect | Détail |
|--------|--------|
| **Algorithmes** | KNN, Random Forest, Linear Regression |
| **Optimisation** | GridSearchCV (validation croisée) |
| **Métrique** | Mean Absolute Error (MAE) |

- Comparaison automatique de 3 algorithmes
- Sélection du meilleur modèle
- Prédiction en millisecondes, affichage formaté

### 🏢 Classification des Équipes

| Aspect | Détail |
|--------|--------|
| **Modèle** | K-Means Clustering (k=3) |
| **Catégories** | Top Teams, Mid-field, Back-markers |
| **Features** | Points saison + Quali Pace Ratio |

- Classification automatique des 10 équipes F1
- Visualisation scatter plot des clusters
- Analyse comparative des performances

### 📊 Classements & Calendrier

- Classement pilotes par année
- Classement constructeurs par année
- Calendrier F1 2025 (courses terminées et à venir)

---

## 🛠️ Technologies

### Langages & Frameworks

| Technologie | Utilisation |
|-------------|-------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Langage principal |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interface web interactive |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Manipulation de données |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Calculs numériques |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | Visualisations interactives |

### Outils de Développement

| Outil | Utilisation |
|-------|-------------|
| Google Colab | Entraînement des modèles |
| Joblib | Sérialisation des modèles |
| Git & GitHub | Versioning et collaboration |
| Streamlit Cloud | Déploiement |

---

## 🤖 Les Modèles ML

### 1. 🏆 Modèle de Probabilité de Victoire

**Algorithme** : Ridge Regression + Plackett-Luce

#### Principe

1. Ridge Regression → Score de force θ (theta) pour chaque pilote

2. Plackett-Luce → Conversion en probabilités :

    P(pilote i gagne) = exp(θᵢ) / Σ exp(θⱼ)


#### Features utilisées

| Feature | Description |
|---------|-------------|
| `grid` | Position sur la grille de départ |
| `laps` | Nombre de tours de la course |
| `q1_sec` | Temps en Q1 (secondes) |
| `q2_sec` | Temps en Q2 (secondes) |
| `q3_sec` | Temps en Q3 (secondes) |
| `fastestLapTime` | Meilleur tour en course (ms) |
| `avg_lap_ms` | Temps moyen au tour (ms) |
| `pit_stop_count` | Nombre d'arrêts aux stands |
| `avg_pit_duration_s` | Durée moyenne d'un pit stop (s) |

#### Avantages

- ✅ Probabilités cohérentes (somme = 100%)
- ✅ Modèle théoriquement fondé
- ✅ Score θ interprétable (plus élevé = pilote plus fort)

---

### 2. ⏱️ Modèle de Temps de Course

**Approche** : Comparaison de 3 algorithmes avec sélection automatique

#### Algorithmes comparés

| Algorithme | Hyperparamètres optimisés |
|------------|---------------------------|
| **KNN** | n_neighbors, weights |
| **Random Forest** | n_estimators, max_depth |
| **Linear Regression** | - |

#### Processus

1. GridSearchCV avec cv=3
2. Scoring : neg_mean_absolute_error
3. Sélection du modèle avec le plus petit MAE


#### Features utilisées

| Feature | Description |
|---------|-------------|
| `grid` | Position de départ |
| `circuitId` | Identifiant du circuit |
| `constructorId` | Identifiant de l'équipe |
| `number_driver` | Numéro du pilote |
| `year` | Année de la course |

#### Target

`milliseconds` → Temps total de course en millisecondes

---

### 3. 🏢 Modèle de Classification des Équipes

**Algorithme** : K-Means Clustering (k=3)

#### Catégories

| Tier | Emoji | Description | Exemple |
|------|-------|-------------|---------|
| **Top Teams** | 🏆 | Équipes de tête | Red Bull, Ferrari, McLaren |
| **Mid-field** | 🔵 | Milieu de grille | Aston Martin, Alpine |
| **Back-markers** | ⬇️ | Fond de grille | Williams, Sauber |

#### Features utilisées

| Feature | Description |
|---------|-------------|
| `points` | Points totaux sur la saison |
| `Quali_Pace_Ratio` | Ratio temps quali / temps pole (1.0 = pole) |

#### Visualisation

Scatter plot avec les clusters colorés et les centres de chaque groupe.

---

## 📁 Structure du Projet

```text
f1-prediction/
│
├── 📄 app.py                   # Application Streamlit principale
├── 📄 requirements.txt         # Dépendances Python
├── 📄 README.md                # Documentation (ce fichier)
├── 📄 LICENSE                  # Licence MIT
├── 📄 .gitignore               # Fichiers ignorés par Git
│
├── 📁 .streamlit/              # Configuration Streamlit
│   └── config.toml             # Thème et paramètres
│
├── 📁 models/                  # Modèles ML sauvegardés
│   ├── model_driver_win.pkl    # Modèle probabilité de victoire
│   ├── model_driver_time.pkl   # Modèle temps de course
│   ├── model_team_perf.pkl     # Modèle performance équipe
│   ├── scaler_driver_win.pkl   # Scaler win probability
│   ├── scaler_driver_time.pkl  # Scaler temps de course
│   └── scaler_team_perf.pkl    # Scaler team performance
│
├── 📁 data/                    # Données
│   ├── __init__.py
│   ├── data_loader.py          # Chargement CSV et mappings
│   └── FinalCombinedCleanFinal.csv
│
└── 📁 src/                     # Code source
    ├── __init__.py
    ├── models.py               # Chargement et prédiction
    └── features.py             # Préparation des features

```
---

## 🚀 Installation Locale

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de packages)
- Git

### Étapes

#### 1. Cloner le repository

git clone https://github.com/VOTRE_USERNAME/f1-prediction.git
cd f1-prediction

#### 2. Créer un environnement virtuel

Windows :

python -m venv venv
venv\Scripts\activate

Linux / Mac :

python -m venv venv
source venv/bin/activate

#### 3. Installer les dépendances

pip install -r requirements.txt

#### 4. Lancer l'application

streamlit run app.py

#### 5. Ouvrir dans le navigateur
http://localhost:8501

## 📖 Utilisation

🏆 Prédiction du Podium
- Sélectionnez un Grand Prix dans la liste
- Configurez les paramètres de course (tours, durée pit stop)
- (Optionnel) Modifiez les positions de grille
- Cliquez sur "Calculer les Probabilités de Victoire"
- Visualisez :
    Le podium prédit (1er, 2ème, 3ème)
    Les scores θ de chaque pilote
    Le graphique des probabilités

⏱️ Temps de Course
- Sélectionnez un pilote
- Choisissez un circuit
- Définissez la position de grille
- Cliquez sur "Prédire le Temps"
- Obtenez le temps estimé en format HH:MM:SS.mmm

🏢 Performance Équipe
- Sélectionnez une équipe ou comparez toutes
- Ajustez les paramètres :
- Points : Points accumulés sur la saison
- Quali Pace Ratio : Performance en qualification
- Visualisez la classification :
    🏆 Top Teams
    🔵 Mid-field
    ⬇️ Back-markers

## 📊 Dataset
Source
Données historiques de Formule 1 compilées et nettoyées.

Fichier :
FinalCombinedCleanFinal.csv

Colonnes principales :

| Colonne | Description |
|--------|-------------|
| `raceId` | Identifiant unique de la course |
| `driverId` | Identifiant du pilote |
| `constructorId` | Identifiant de l'équipe |
| `circuitId` | Identifiant du circuit |
| `year` | Année de la course |
| `number_driver` | Numéro du pilote |
| `grid` | Position de départ |
| `positionOrder` | Position finale |
| `points` | Points marqués |
| `milliseconds` | Temps de course (ms) |
| `status` | Statut (Finished, DNF, etc.) |
| `laps` | Nombre de tours |
| `fastestLapTime` | Meilleur tour |
| `avg_lap_ms` | Temps moyen au tour |
| `pit_stop_count` | Nombre de pit stops |
| `avg_pit_duration_s` | Durée moyenne d’un pit stop |
| `q1, q2, q3` | Temps de qualification |
| `location, country` | Lieu du circuit |


Mappings inclus
Le fichier data/data_loader.py contient les mappings :

- number_driver → Nom du pilote (ex: 1 → "Max Verstappen")
- constructorId → Nom de l'équipe (ex: 9 → "Red Bull")
- circuitId → Nom du circuit (ex: 6 → "Monte Carlo")

## 👥 Auteurs
<table> <tr> <td align="center"> <a href="https://github.com/Rami-Yacoub"> <sub><b>Rami Yacoub</b></sub> </a> <br /> <a href="https://linkedin.com/in/rami-yacoub3">LinkedIn</a> </td> <td align="center"><a href="https://github.com/Onsguidara"><br /> <sub><b>Ons Guidara</b></sub> </a> <br /> <a href="https://www.linkedin.com/in/ons-guidara-3308a1219/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app">LinkedIn</a> </td> </tr> </table>
Formation : 2ème année Cycle Ingénieur - Spécialité Intelligence Artificielle

Année : 2025-2026

## 📄 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

MIT License

Copyright (c) 2025 Rami Yacoub & Ons Guidara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

## ⭐ Support
Si ce projet vous a été utile, n'hésitez pas à :

- ⭐ Mettre une étoile sur le repository
- 🐛 Signaler un bug via les Issues
- 💡 Proposer une amélioration via une Pull Request
- 📢 Partager avec votre réseau


<p align="center"> <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love"> <img src="https://img.shields.io/badge/and-🏎️-black.svg" alt="and F1"> <img src="https://img.shields.io/badge/by-AI%20Students-blue.svg" alt="by AI Students"> </p><p align="center"> <a href="#-f1-prediction-app">⬆️ Retour en haut</a> </p>