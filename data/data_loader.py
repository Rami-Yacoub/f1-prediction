"""
Chargement et traitement du dataset F1
Adapté aux colonnes du fichier FinalCombinedCleanFinal.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Chemin vers le fichier CSV
DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "FinalCombinedCleanFinal.csv"

# =============================================================================
# MAPPINGS
# =============================================================================

# Mapping number_driver -> Nom du pilote
DRIVERS_MAP = {
    1: "Max Verstappen",
    3: "Daniel Ricciardo",
    4: "Lando Norris",
    5: "Sebastian Vettel",
    6: "Nicholas Latifi",
    7: "Kimi Räikkönen",
    8: "Romain Grosjean",
    9: "Marcus Ericsson",
    10: "Pierre Gasly",
    11: "Sergio Perez",
    14: "Fernando Alonso",
    16: "Charles Leclerc",
    18: "Lance Stroll",
    20: "Kevin Magnussen",
    21: "Nyck de Vries",
    22: "Yuki Tsunoda",
    23: "Alex Albon",
    24: "Guanyu Zhou",
    26: "Daniil Kvyat",
    27: "Nico Hulkenberg",
    31: "Esteban Ocon",
    33: "Max Verstappen",  # Ancien numéro
    44: "Lewis Hamilton",
    47: "Mick Schumacher",
    55: "Carlos Sainz",
    63: "George Russell",
    77: "Valtteri Bottas",
    81: "Oscar Piastri",
    87: "Oliver Bearman",
    99: "Antonio Giovinazzi",
    # Pilotes 2025
    12: "Kimi Antonelli",
    30: "Liam Lawson",
    38: "Oliver Bearman",
    43: "Franco Colapinto",
    50: "Gabriel Bortoleto",
    61: "Isack Hadjar",
}

# Mapping constructorId -> Nom de l'équipe
CONSTRUCTORS_MAP = {
    1: "McLaren",
    3: "Williams",
    4: "Renault",
    5: "Toro Rosso",
    6: "Ferrari",
    9: "Red Bull",
    10: "Force India",
    15: "BMW Sauber",
    17: "Lotus",
    21: "Haas F1 Team",
    25: "Caterham",
    51: "Haas F1 Team",
    117: "Lotus F1",
    131: "Mercedes",
    210: "Aston Martin",
    211: "Alpine",
    213: "AlphaTauri",
    214: "Alfa Romeo",
    215: "RB",
    216: "Kick Sauber",
    217: "Racing Point",
}

# Mapping circuitId -> Nom du circuit
CIRCUITS_MAP = {
    1: "Albert Park (Melbourne)",
    2: "Sepang",
    3: "Sakhir (Bahrain)",
    4: "Barcelona",
    5: "Istanbul",
    6: "Monte Carlo (Monaco)",
    7: "Montreal (Canada)",
    9: "Silverstone",
    10: "Hockenheim",
    11: "Hungaroring (Budapest)",
    12: "Valencia",
    13: "Spa-Francorchamps",
    14: "Monza",
    15: "Marina Bay (Singapore)",
    17: "Suzuka (Japan)",
    18: "Yas Marina (Abu Dhabi)",
    21: "Austin (COTA)",
    22: "Spielberg (Austria)",
    23: "Sochi",
    24: "Mexico City",
    25: "Interlagos (São Paulo)",
    26: "Baku",
    27: "Shanghai",
    29: "Zandvoort",
    30: "Jeddah",
    31: "Losail (Qatar)",
    32: "Miami",
    33: "Las Vegas",
    34: "Imola",
}

# Couleurs des équipes
TEAM_COLORS = {
    "Red Bull": "#1E41FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#FF00A2",
    "Williams": "#005AFF",
    "RB": "#6692FF",
    "AlphaTauri": "#6692FF",
    "Toro Rosso": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
    "Kick Sauber": "#52E252",
    "Alfa Romeo": "#52E252",
    "Sauber": "#52E252",
    "Renault": "#FF00A2",
    "Racing Point": "#FF69B4",
    "Force India": "#FF69B4",
    "Lotus": "#000000",
    "Lotus F1": "#FFD700",
}

# Calendrier F1 2025
F1_CALENDAR_2025 = [
    {"name": "Australian Grand Prix", "circuit": "Albert Park", "circuit_id": 1, "date": "16 Mar", "laps": 58, "completed": True},
    {"name": "Chinese Grand Prix", "circuit": "Shanghai", "circuit_id": 27, "date": "23 Mar", "laps": 56, "completed": True},
    {"name": "Japanese Grand Prix", "circuit": "Suzuka", "circuit_id": 17, "date": "6 Apr", "laps": 53, "completed": True},
    {"name": "Bahrain Grand Prix", "circuit": "Sakhir", "circuit_id": 3, "date": "13 Apr", "laps": 57, "completed": True},
    {"name": "Saudi Arabian Grand Prix", "circuit": "Jeddah", "circuit_id": 30, "date": "20 Apr", "laps": 50, "completed": True},
    {"name": "Miami Grand Prix", "circuit": "Miami", "circuit_id": 32, "date": "4 May", "laps": 57, "completed": True},
    {"name": "Emilia Romagna Grand Prix", "circuit": "Imola", "circuit_id": 34, "date": "18 May", "laps": 63, "completed": True},
    {"name": "Monaco Grand Prix", "circuit": "Monte Carlo", "circuit_id": 6, "date": "25 May", "laps": 78, "completed": True},
    {"name": "Spanish Grand Prix", "circuit": "Barcelona", "circuit_id": 4, "date": "1 Jun", "laps": 66, "completed": True},
    {"name": "Canadian Grand Prix", "circuit": "Montreal", "circuit_id": 7, "date": "15 Jun", "laps": 70, "completed": True},
    {"name": "Austrian Grand Prix", "circuit": "Spielberg", "circuit_id": 22, "date": "29 Jun", "laps": 71, "completed": True},
    {"name": "British Grand Prix", "circuit": "Silverstone", "circuit_id": 9, "date": "6 Jul", "laps": 52, "completed": True},
    {"name": "Belgian Grand Prix", "circuit": "Spa-Francorchamps", "circuit_id": 13, "date": "27 Jul", "laps": 44, "completed": False},
    {"name": "Hungarian Grand Prix", "circuit": "Budapest", "circuit_id": 11, "date": "3 Aug", "laps": 70, "completed": False},
    {"name": "Dutch Grand Prix", "circuit": "Zandvoort", "circuit_id": 29, "date": "31 Aug", "laps": 72, "completed": False},
    {"name": "Italian Grand Prix", "circuit": "Monza", "circuit_id": 14, "date": "7 Sep", "laps": 53, "completed": False},
    {"name": "Azerbaijan Grand Prix", "circuit": "Baku", "circuit_id": 26, "date": "21 Sep", "laps": 51, "completed": False},
    {"name": "Singapore Grand Prix", "circuit": "Marina Bay", "circuit_id": 15, "date": "5 Oct", "laps": 62, "completed": False},
    {"name": "United States Grand Prix", "circuit": "Austin", "circuit_id": 21, "date": "19 Oct", "laps": 56, "completed": False},
    {"name": "Mexico City Grand Prix", "circuit": "Mexico City", "circuit_id": 24, "date": "26 Oct", "laps": 71, "completed": False},
    {"name": "São Paulo Grand Prix", "circuit": "Interlagos", "circuit_id": 25, "date": "9 Nov", "laps": 71, "completed": False},
    {"name": "Las Vegas Grand Prix", "circuit": "Las Vegas", "circuit_id": 33, "date": "22 Nov", "laps": 50, "completed": False},
    {"name": "Qatar Grand Prix", "circuit": "Losail", "circuit_id": 31, "date": "30 Nov", "laps": 57, "completed": False},
    {"name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina", "circuit_id": 18, "date": "7 Dec", "laps": 58, "completed": False},
]

# Grille 2025 (pour les prédictions)
GRID_2025 = [
    {"name": "Max Verstappen", "number": 1, "team": "Red Bull", "constructor_id": 9},
    {"name": "Yuki Tsunoda", "number": 22, "team": "Red Bull", "constructor_id": 9},
    {"name": "Charles Leclerc", "number": 16, "team": "Ferrari", "constructor_id": 6},
    {"name": "Lewis Hamilton", "number": 44, "team": "Ferrari", "constructor_id": 6},
    {"name": "George Russell", "number": 63, "team": "Mercedes", "constructor_id": 131},
    {"name": "Kimi Antonelli", "number": 12, "team": "Mercedes", "constructor_id": 131},
    {"name": "Lando Norris", "number": 4, "team": "McLaren", "constructor_id": 1},
    {"name": "Oscar Piastri", "number": 81, "team": "McLaren", "constructor_id": 1},
    {"name": "Fernando Alonso", "number": 14, "team": "Aston Martin", "constructor_id": 210},
    {"name": "Lance Stroll", "number": 18, "team": "Aston Martin", "constructor_id": 210},
    {"name": "Pierre Gasly", "number": 10, "team": "Alpine", "constructor_id": 211},
    {"name": "Franco Colapinto", "number": 43, "team": "Alpine", "constructor_id": 211},
    {"name": "Alex Albon", "number": 23, "team": "Williams", "constructor_id": 3},
    {"name": "Carlos Sainz", "number": 55, "team": "Williams", "constructor_id": 3},
    {"name": "Liam Lawson", "number": 30, "team": "RB", "constructor_id": 215},
    {"name": "Isack Hadjar", "number": 61, "team": "RB", "constructor_id": 215},
    {"name": "Esteban Ocon", "number": 31, "team": "Haas F1 Team", "constructor_id": 51},
    {"name": "Oliver Bearman", "number": 87, "team": "Haas F1 Team", "constructor_id": 51},
    {"name": "Nico Hulkenberg", "number": 27, "team": "Kick Sauber", "constructor_id": 216},
    {"name": "Gabriel Bortoleto", "number": 50, "team": "Kick Sauber", "constructor_id": 216},
]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_driver_name(driver_number):
    """Retourne le nom du pilote à partir de son numéro."""
    return DRIVERS_MAP.get(driver_number, f"Pilote #{driver_number}")

def get_constructor_name(constructor_id):
    """Retourne le nom du constructeur à partir de son ID."""
    return CONSTRUCTORS_MAP.get(constructor_id, f"Constructeur #{constructor_id}")

def get_circuit_name(circuit_id):
    """Retourne le nom du circuit à partir de son ID."""
    return CIRCUITS_MAP.get(circuit_id, f"Circuit #{circuit_id}")

def get_team_color(team_name):
    """Retourne la couleur d'une équipe."""
    for team, color in TEAM_COLORS.items():
        if team.lower() in team_name.lower() or team_name.lower() in team.lower():
            return color
    return "#FFFFFF"

def get_driver_number(driver_name):
    """Retourne le numéro d'un pilote à partir de son nom."""
    for number, name in DRIVERS_MAP.items():
        if name.lower() == driver_name.lower():
            return number
    return None

def get_constructor_id(constructor_name):
    """Retourne l'ID d'un constructeur à partir de son nom."""
    for cid, name in CONSTRUCTORS_MAP.items():
        if name.lower() == constructor_name.lower():
            return cid
    return None

def get_circuit_id(circuit_name):
    """Retourne l'ID d'un circuit à partir de son nom."""
    for cid, name in CIRCUITS_MAP.items():
        if circuit_name.lower() in name.lower():
            return cid
    return None


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class F1DataLoader:
    """Classe pour charger et gérer les données F1."""
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path or CSV_PATH
        self.df = None
        self.loaded = False
        
    def load(self):
        """Charge le CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Ajouter les colonnes de noms
            if 'number_driver' in self.df.columns:
                self.df['driver_name'] = self.df['number_driver'].map(DRIVERS_MAP)
                self.df['driver_name'] = self.df['driver_name'].fillna(
                    self.df['number_driver'].apply(lambda x: f"Pilote #{x}")
                )
            
            if 'constructorId' in self.df.columns:
                self.df['constructor_name'] = self.df['constructorId'].map(CONSTRUCTORS_MAP)
                self.df['constructor_name'] = self.df['constructor_name'].fillna(
                    self.df['constructorId'].apply(lambda x: f"Constructeur #{x}")
                )
            
            if 'circuitId' in self.df.columns:
                self.df['circuit_name'] = self.df['circuitId'].map(CIRCUITS_MAP)
                # Utiliser 'location' si circuit_name est manquant
                if 'location' in self.df.columns:
                    self.df['circuit_name'] = self.df['circuit_name'].fillna(self.df['location'])
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"Erreur de chargement: {e}")
            return False
    
    def get_all_drivers(self):
        """Retourne la liste de tous les pilotes uniques."""
        if self.df is not None and 'driver_name' in self.df.columns:
            return sorted(self.df['driver_name'].dropna().unique().tolist())
        # Fallback sur la grille 2025
        return [d["name"] for d in GRID_2025]
    
    def get_all_constructors(self):
        """Retourne la liste de tous les constructeurs uniques."""
        if self.df is not None and 'constructor_name' in self.df.columns:
            return sorted(self.df['constructor_name'].dropna().unique().tolist())
        return list(set(CONSTRUCTORS_MAP.values()))
    
    def get_all_circuits(self):
        """Retourne la liste de tous les circuits uniques."""
        if self.df is not None:
            if 'circuit_name' in self.df.columns:
                circuits = self.df['circuit_name'].dropna().unique().tolist()
            elif 'location' in self.df.columns:
                circuits = self.df['location'].dropna().unique().tolist()
            else:
                circuits = list(CIRCUITS_MAP.values())
            return sorted(circuits)
        return list(CIRCUITS_MAP.values())
    
    def get_driver_info(self, driver_name):
        """Retourne les infos d'un pilote."""
        # Chercher dans la grille 2025
        for driver in GRID_2025:
            if driver["name"].lower() == driver_name.lower():
                return driver
        
        # Chercher dans le dataset
        if self.df is not None and 'driver_name' in self.df.columns:
            driver_df = self.df[self.df['driver_name'] == driver_name]
            if not driver_df.empty:
                row = driver_df.iloc[-1]  # Prendre la dernière entrée
                return {
                    "name": driver_name,
                    "number": row.get('number_driver', 0),
                    "team": row.get('constructor_name', 'Unknown'),
                    "constructor_id": row.get('constructorId', 0)
                }
        
        # Chercher dans le mapping
        for number, name in DRIVERS_MAP.items():
            if name.lower() == driver_name.lower():
                return {
                    "name": name,
                    "number": number,
                    "team": "Unknown",
                    "constructor_id": 0
                }
        
        return None
    
    def get_driver_constructor(self, driver_name, year=2025):
        """Retourne le constructeur d'un pilote pour une année donnée."""
        # Chercher dans la grille 2025
        if year >= 2025:
            for driver in GRID_2025:
                if driver["name"].lower() == driver_name.lower():
                    return driver["team"]
        
        # Chercher dans le dataset
        if self.df is not None:
            df_filtered = self.df.copy()
            
            if 'year' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['year'] == year]
            
            if 'driver_name' in df_filtered.columns:
                driver_df = df_filtered[df_filtered['driver_name'] == driver_name]
                if not driver_df.empty:
                    return driver_df['constructor_name'].iloc[-1]
        
        return None
    
    def get_circuit_id_from_name(self, circuit_name):
        """Retourne l'ID du circuit à partir de son nom."""
        # Chercher dans le mapping
        for cid, name in CIRCUITS_MAP.items():
            if circuit_name.lower() in name.lower():
                return cid
        
        # Chercher dans le dataset via location
        if self.df is not None and 'location' in self.df.columns:
            circuit_df = self.df[self.df['location'].str.lower() == circuit_name.lower()]
            if not circuit_df.empty:
                return circuit_df['circuitId'].iloc[0]
        
        return None
    
    def get_unique_years(self):
        """Retourne les années disponibles."""
        if self.df is not None and 'year' in self.df.columns:
            return sorted(self.df['year'].dropna().unique().tolist(), reverse=True)
        return [2025, 2024, 2023, 2022, 2021, 2020]
    
    def get_latest_standings(self, year=2024):
        """Retourne le classement pilotes pour une année."""
        if self.df is None or 'year' not in self.df.columns:
            return None
        
        df_year = self.df[self.df['year'] == year]
        
        if df_year.empty:
            return None
        
        # Calculer les points totaux par pilote
        if 'driver_name' in df_year.columns and 'points' in df_year.columns:
            standings = df_year.groupby('driver_name')['points'].sum().reset_index()
            standings = standings.sort_values('points', ascending=False).reset_index(drop=True)
            standings['position'] = range(1, len(standings) + 1)
            standings.columns = ['Pilote', 'Points', 'Position']
            return standings[['Position', 'Pilote', 'Points']]
        
        return None
    
    def get_constructor_standings(self, year=2024):
        """Retourne le classement constructeurs pour une année."""
        if self.df is None or 'year' not in self.df.columns:
            return None
        
        df_year = self.df[self.df['year'] == year]
        
        if df_year.empty:
            return None
        
        if 'constructor_name' in df_year.columns and 'points' in df_year.columns:
            standings = df_year.groupby('constructor_name')['points'].sum().reset_index()
            standings = standings.sort_values('points', ascending=False).reset_index(drop=True)
            standings['position'] = range(1, len(standings) + 1)
            standings.columns = ['Constructeur', 'Points', 'Position']
            return standings[['Position', 'Constructeur', 'Points']]
        
        return None
    
    def get_columns(self):
        """Retourne la liste des colonnes."""
        if self.df is not None:
            return self.df.columns.tolist()
        return []
    
    def get_sample_data(self, n=5):
        """Retourne un échantillon des données."""
        if self.df is not None:
            return self.df.head(n)
        return None


# Instance globale pour réutilisation
data_loader = F1DataLoader()