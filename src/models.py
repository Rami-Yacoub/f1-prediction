"""
Chargement des modèles et scalers
"""
import joblib
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
        self.tier_mapping = {}  # Pour stocker le mapping cluster -> tier
    
    def load_all(self):
        """Charge tous les modèles et scalers."""
        self.errors = []
        
        # Modèle Win Probability
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
    
    def predict_win_probability(self, X):
        """Prédit la probabilité de victoire."""
        model = self.models.get("driver_win")
        scaler = self.scalers.get("driver_win")
        
        if model is None or scaler is None:
            raise ValueError("Modèle driver_win non chargé")
        
        X_scaled = scaler.transform(X)
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_scaled)[:, 1]
        else:
            return model.predict(X_scaled)
    
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
        # Le cluster avec le plus de points = Top Teams
        # Le cluster avec le moins de points = Back-markers
        centers = model.cluster_centers_
        
        # Inverser le scaling pour obtenir les vraies valeurs
        centers_original = scaler.inverse_transform(centers)
        
        # Trier les clusters par points (colonne 0)
        # Plus de points = meilleur tier
        sorted_clusters = sorted(range(len(centers_original)), 
                                  key=lambda i: centers_original[i][0], 
                                  reverse=True)
        
        # Créer le mapping : cluster avec plus de points = Top Teams (2)
        tier_mapping = {
            sorted_clusters[0]: 2,  # Top Teams
            sorted_clusters[1]: 1,  # Mid-field
            sorted_clusters[2]: 0,  # Back-markers
        }
        
        tier_label = tier_mapping.get(cluster, 1)
        tier_name = TIER_NAMES.get(tier_label, "Unknown")
        
        return cluster, tier_label, tier_name
    
    def get_cluster_centers(self):
        """Retourne les centres des clusters pour l'affichage."""
        model = self.models.get("team_perf")
        scaler = self.scalers.get("team_perf")
        
        if model is None or scaler is None:
            return None
        
        centers_scaled = model.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        
        return centers_original