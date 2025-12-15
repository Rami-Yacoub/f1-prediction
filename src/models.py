"""
Chargement des modèles et scalers
"""
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Mapping des clusters vers les noms de tiers
TIER_NAMES = {
    0: "Back-markers",
    1: "Mid-field", 
    2: "Top Teams"
}


class ModelLoader:
    """Classe pour charger et gérer les modèles."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.loaded = False
        self.errors = []
    
    def load_all(self):
        """Charge tous les modèles et scalers."""
        self.errors = []
        
        # Modèle Win Probability (Ridge pour theta)
        try:
            self.models["driver_win"] = joblib.load(MODELS_DIR / "model_driver_win.pkl")
            self.scalers["driver_win"] = joblib.load(MODELS_DIR / "scaler_driver_win.pkl")
        except Exception as e:
            self.errors.append(f"driver_win: {str(e)}")
        
        # Modèle Temps de Course
        try:
            self.models["driver_time"] = joblib.load(MODELS_DIR / "model_driver_time.pkl")
            self.scalers["driver_time"] = joblib.load(MODELS_DIR / "scaler_driver_time.pkl")
        except Exception as e:
            self.errors.append(f"driver_time: {str(e)}")
        
        # Modèle Team Performance (K-Means)
        try:
            self.models["team_perf"] = joblib.load(MODELS_DIR / "model_team_perf.pkl")
            self.scalers["team_perf"] = joblib.load(MODELS_DIR / "scaler_team_perf.pkl")
        except Exception as e:
            self.errors.append(f"team_perf: {str(e)}")
        
        self.loaded = len(self.errors) == 0
        return self.loaded
    
    def predict_theta(self, X):
        """
        Prédit le score theta (force) pour un pilote.
        
        Args:
            X: DataFrame avec les features
        
        Returns:
            Score theta (float)
        """
        model = self.models.get("driver_win")
        scaler = self.scalers.get("driver_win")
        
        if model is None or scaler is None:
            raise ValueError("Modèle driver_win non chargé")
        
        X_scaled = scaler.transform(X)
        theta = model.predict(X_scaled)[0]
        
        return theta
    
    def predict_win_probabilities(self, drivers_features_list):
        """
        Prédit les probabilités de victoire pour tous les pilotes d'une course
        en utilisant le modèle Plackett-Luce.
        
        Args:
            drivers_features_list: Liste de tuples (driver_name, X_features)
        
        Returns:
            Liste de tuples (driver_name, theta, win_probability)
        """
        model = self.models.get("driver_win")
        scaler = self.scalers.get("driver_win")
        
        if model is None or scaler is None:
            raise ValueError("Modèle driver_win non chargé")
        
        # Calculer theta pour chaque pilote
        results = []
        thetas = []
        
        for driver_name, X in drivers_features_list:
            X_scaled = scaler.transform(X)
            theta = model.predict(X_scaled)[0]
            thetas.append(theta)
            results.append({"driver": driver_name, "theta": theta})
        
        # Appliquer Plackett-Luce pour obtenir les probabilités
        thetas = np.array(thetas)
        exp_theta = np.exp(thetas - np.max(thetas))  # Stabilisation numérique
        probabilities = exp_theta / exp_theta.sum()
        
        # Ajouter les probabilités aux résultats
        for i, prob in enumerate(probabilities):
            results[i]["win_probability"] = prob
        
        return results
    
    def predict_race_time(self, X):
        """Prédit le temps de course en millisecondes."""
        model = self.models.get("driver_time")
        scaler = self.scalers.get("driver_time")
        
        if model is None or scaler is None:
            raise ValueError("Modèle driver_time non chargé")
        
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)
    
    def predict_team_tier(self, X):
        """
        Prédit le tier (catégorie) d'une équipe via K-Means.
        
        Returns:
            cluster: Numéro du cluster (0, 1, 2)
            tier_name: Nom du tier ("Back-markers", "Mid-field", "Top Teams")
        """
        model = self.models.get("team_perf")
        scaler = self.scalers.get("team_perf")
        
        if model is None or scaler is None:
            raise ValueError("Modèle team_perf non chargé")
        
        X_scaled = scaler.transform(X)
        cluster = model.predict(X_scaled)[0]
        
        # Déterminer le tier basé sur les centres des clusters
        centers = model.cluster_centers_
        centers_original = scaler.inverse_transform(centers)
        
        # Trier les clusters par points (colonne 0)
        sorted_clusters = sorted(range(len(centers_original)), 
                                  key=lambda i: centers_original[i][0], 
                                  reverse=True)
        
        tier_mapping = {
            sorted_clusters[0]: 2,  # Top Teams
            sorted_clusters[1]: 1,  # Mid-field
            sorted_clusters[2]: 0,  # Back-markers
        }
        
        tier_label = tier_mapping.get(cluster, 1)
        tier_name = TIER_NAMES.get(tier_label, "Unknown")
        
        return cluster, tier_label, tier_name