"""
Préparation des features pour chaque modèle
"""
import pandas as pd
import numpy as np


def time_to_seconds(time_str):
    """Convertit un temps MM:SS.mmm en secondes."""
    if pd.isna(time_str) or time_str in ['DNQ', 'DNS', '', None, 'nan']:
        return np.nan
    
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return np.nan


def build_features_driver_win(grid, year, laps, q1_sec, q2_sec, q3_sec,
                               fastest_lap_time, avg_lap_ms, pit_stop_count, avg_pit_duration_s):
    """
    Features pour le modèle de probabilité de victoire.
    
    Features :
    ['grid', 'year', 'laps', 'q1_sec', 'q2_sec', 'q3_sec', 
     'fastestLapTime', 'avg_lap_ms', 'pit_stop_count', 'avg_pit_duration_s']
    """
    data = {
        'grid': [int(grid)],
        'year': [int(year)],
        'laps': [int(laps)],
        'q1_sec': [float(q1_sec) if q1_sec and q1_sec != 999 else 999.0],
        'q2_sec': [float(q2_sec) if q2_sec and q2_sec != 999 else 999.0],
        'q3_sec': [float(q3_sec) if q3_sec and q3_sec != 999 else 999.0],
        'fastestLapTime': [float(fastest_lap_time)],
        'avg_lap_ms': [float(avg_lap_ms)],
        'pit_stop_count': [int(pit_stop_count)],
        'avg_pit_duration_s': [float(avg_pit_duration_s)],
    }
    return pd.DataFrame(data)


def build_features_driver_time(grid, circuit_id, constructor_id, driver_number, year):
    """
    Features pour le modèle de temps de course.
    
    Features :
    ['grid', 'circuitId', 'constructorId', 'number_driver', 'year']
    """
    data = {
        'grid': [int(grid)],
        'circuitId': [int(circuit_id)],
        'constructorId': [int(constructor_id)],
        'number_driver': [int(driver_number)],
        'year': [int(year)],
    }
    return pd.DataFrame(data)


def build_features_team_perf(points, quali_pace_ratio):
    """
    Features pour le modèle de performance équipe (K-Means Clustering).
    
    Features (selon ton notebook) :
    ['points', 'Quali_Pace_Ratio']
    
    Args:
        points: Points totaux de l'équipe sur la saison
        quali_pace_ratio: Ratio entre le meilleur temps de qualification et le temps pole
                          (1.0 = pole, 1.01 = 1% plus lent que la pole)
    """
    data = {
        'points': [float(points)],
        'Quali_Pace_Ratio': [float(quali_pace_ratio)],
    }
    return pd.DataFrame(data)


def milliseconds_to_time_string(ms):
    """Convertit des millisecondes en format lisible."""
    if pd.isna(ms) or ms <= 0:
        return "N/A"
    
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes}:{seconds:06.3f}"


def seconds_to_time_string(seconds):
    """Convertit des secondes en format lisible."""
    if pd.isna(seconds) or seconds <= 0:
        return "N/A"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    return f"{minutes}:{secs:06.3f}"