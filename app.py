"""
F1 Prediction App - Application Streamlit
Utilise le dataset FinalCombinedCleanFinal.csv avec mappings
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Imports locaux
from src.models import ModelLoader
from src.features import (
    build_features_driver_win,
    build_features_driver_time,
    build_features_team_perf,
    time_to_seconds,
    milliseconds_to_time_string,
)
from data.data_loader import (
    F1DataLoader,
    F1_CALENDAR_2025,
    GRID_2025,
    TEAM_COLORS,
    DRIVERS_MAP,
    CONSTRUCTORS_MAP,
    CIRCUITS_MAP,
    get_driver_name,
    get_constructor_name,
    get_circuit_name,
    get_team_color,
    get_driver_number,
    get_constructor_id,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="F1 Prediction",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E10600;
        margin-bottom: 1rem;
    }
    .podium-gold {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: black;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    }
    .podium-silver {
        background: linear-gradient(135deg, #C0C0C0, #A8A8A8);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        color: black;
        box-shadow: 0 4px 15px rgba(192, 192, 192, 0.4);
    }
    .podium-bronze {
        background: linear-gradient(135deg, #CD7F32, #8B4513);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(205, 127, 50, 0.4);
    }
    .result-card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #E10600;
        margin: 15px 0;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00FF00;
    }
    .team-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT
# =============================================================================
@st.cache_resource
def load_data():
    """Charge les données du CSV."""
    loader = F1DataLoader()
    success = loader.load()
    return loader, success

@st.cache_resource
def load_models():
    """Charge les modèles ML."""
    loader = ModelLoader()
    loader.load_all()
    return loader

data_loader, data_loaded = load_data()
model_loader = load_models()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
    st.markdown("---")
    
    # Statut des données
    st.header("📁 Données")
    if data_loaded:
        st.success("✅ CSV chargé")
        st.caption(f"📊 {len(data_loader.df)} lignes")
        st.caption(f"👤 {len(data_loader.get_all_drivers())} pilotes")
        st.caption(f"🏢 {len(data_loader.get_all_constructors())} constructeurs")
    else:
        st.error("❌ CSV non trouvé")
    
    st.markdown("---")
    
    # Statut des modèles
    st.header("🤖 Modèles")
    
    model_status = {
        "driver_win": "Win %",
        "driver_time": "Temps",
        "team_perf": "Team"
    }
    
    for key, name in model_status.items():
        if key in model_loader.models:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name}")
    
    if model_loader.errors:
        with st.expander("Voir les erreurs"):
            for err in model_loader.errors:
                st.caption(err)
    
    st.markdown("---")
    
    # Stats calendrier
    races_done = sum(1 for r in F1_CALENDAR_2025 if r["completed"])
    races_left = len(F1_CALENDAR_2025) - races_done
    
    col1, col2 = st.columns(2)
    col1.metric("✅", races_done)
    col2.metric("🔜", races_left)

# =============================================================================
# TITRE
# =============================================================================
st.markdown('<h1 class="main-title">🏎️ F1 Prediction App</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Prédictions basées sur Machine Learning</p>", unsafe_allow_html=True)

# =============================================================================
# ONGLETS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Podium & Victoire",
    "⏱️ Temps de Course",
    "🏢 Performance Équipe",
    "📊 Classements",
    "📅 Calendrier"
])

# =============================================================================
# TAB 1: PODIUM
# =============================================================================
with tab1:
    st.header("🏆 Prédiction du Podium")
    st.markdown("""
    Ce modèle utilise **Ridge Regression + Plackett-Luce** pour calculer les probabilités de victoire.
    
    1. **Ridge** calcule un score de "force" (θ) pour chaque pilote
    2. **Plackett-Luce** convertit ces scores en probabilités de victoire
    """)
    
    # Sélection du Grand Prix
    upcoming_races = [r for r in F1_CALENDAR_2025 if not r["completed"]]
    
    if upcoming_races:
        race_options = [f"{r['name']} - {r['circuit']}" for r in upcoming_races]
    else:
        race_options = [f"{r['name']} - {r['circuit']}" for r in F1_CALENDAR_2025]
        st.info("Toutes les courses sont terminées. Sélectionnez une course pour simuler.")
    
    selected_race_full = st.selectbox("🏁 Sélectionnez un Grand Prix", race_options)
    selected_race_name = selected_race_full.split(" - ")[0]
    
    race_info = next((r for r in F1_CALENDAR_2025 if r["name"] == selected_race_name), None)
    
    if race_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📍 Circuit", race_info['circuit'])
        col2.metric("🔄 Tours", race_info['laps'])
        col3.metric("📅 Date", race_info['date'])
        col4.metric("🆔 ID", race_info['circuit_id'])
    
    st.markdown("---")
    
    # Paramètres globaux
    st.subheader("⚙️ Paramètres de Course")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        laps = st.number_input("Nombre de tours", value=race_info["laps"] if race_info else 55, min_value=1)
    with col2:
        avg_pit_duration = st.slider("Durée pit stop (s)", 20.0, 35.0, 25.0)
    with col3:
        pit_stops = st.number_input("Arrêts aux stands", value=2, min_value=0, max_value=5)
    
    st.markdown("---")
    
    # Configuration par pilote
    st.subheader("🏎️ Configuration des Pilotes")
    
    # Valeurs par défaut basées sur la performance relative
    default_configs = {
        "Max Verstappen": {"grid": 1, "q_delta": 0.0},
        "Charles Leclerc": {"grid": 2, "q_delta": 0.15},
        "Lando Norris": {"grid": 3, "q_delta": 0.20},
        "Oscar Piastri": {"grid": 4, "q_delta": 0.25},
        "Lewis Hamilton": {"grid": 5, "q_delta": 0.30},
        "Carlos Sainz": {"grid": 6, "q_delta": 0.35},
        "George Russell": {"grid": 7, "q_delta": 0.40},
        "Fernando Alonso": {"grid": 8, "q_delta": 0.50},
        "Yuki Tsunoda": {"grid": 9, "q_delta": 0.60},
        "Lance Stroll": {"grid": 10, "q_delta": 0.65},
        "Pierre Gasly": {"grid": 11, "q_delta": 0.70},
        "Esteban Ocon": {"grid": 12, "q_delta": 0.75},
        "Alex Albon": {"grid": 13, "q_delta": 0.80},
        "Nico Hulkenberg": {"grid": 14, "q_delta": 0.85},
        "Liam Lawson": {"grid": 15, "q_delta": 0.90},
        "Franco Colapinto": {"grid": 16, "q_delta": 0.95},
        "Kimi Antonelli": {"grid": 17, "q_delta": 1.00},
        "Oliver Bearman": {"grid": 18, "q_delta": 1.05},
        "Isack Hadjar": {"grid": 19, "q_delta": 1.10},
        "Gabriel Bortoleto": {"grid": 20, "q_delta": 1.15},
    }
    
    with st.expander("🔧 Modifier les positions de grille et temps de qualification", expanded=False):
        st.markdown("""
        **Position Grille** : Position de départ (1 = pole position)
        
        **Delta Q (secondes)** : Écart par rapport à la pole en qualification
        - 0.0 = Temps de la pole
        - 0.5 = 0.5 secondes plus lent que la pole
        """)
        
        driver_configs = {}
        
        cols = st.columns(4)
        
        for idx, driver in enumerate(GRID_2025):
            name = driver["name"]
            defaults = default_configs.get(name, {"grid": idx + 1, "q_delta": 1.0})
            
            with cols[idx % 4]:
                st.markdown(f"**{name}**")
                
                grid = st.number_input(
                    "Grille",
                    min_value=1,
                    max_value=20,
                    value=defaults["grid"],
                    key=f"grid_{name}"
                )
                
                q_delta = st.number_input(
                    "Delta Q (s)",
                    min_value=0.0,
                    max_value=5.0,
                    value=defaults["q_delta"],
                    step=0.05,
                    key=f"qdelta_{name}"
                )
                
                driver_configs[name] = {
                    "grid": grid,
                    "q_delta": q_delta,
                    "team": driver["team"],
                    "number": driver["number"]
                }
    
    # Si l'expander n'est pas ouvert, utiliser les valeurs par défaut
    if not driver_configs:
        for driver in GRID_2025:
            name = driver["name"]
            defaults = default_configs.get(name, {"grid": GRID_2025.index(driver) + 1, "q_delta": 1.0})
            driver_configs[name] = {
                "grid": defaults["grid"],
                "q_delta": defaults["q_delta"],
                "team": driver["team"],
                "number": driver["number"]
            }
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🎯 Calculer les Probabilités de Victoire", type="primary", use_container_width=True):
        
        if "driver_win" not in model_loader.models:
            st.error("❌ Modèle driver_win non chargé. Placez model_driver_win.pkl dans models/")
        else:
            with st.spinner("Calcul des scores θ et probabilités Plackett-Luce..."):
                
                # Temps de référence pour la pole (en secondes)
                base_pole_time = 80.0  # ~1:20.000
                
                # Préparer les features pour chaque pilote
                drivers_features = []
                
                for name, config in driver_configs.items():
                    # Calculer les temps de qualification basés sur le delta
                    pole_time = base_pole_time
                    driver_q_time = pole_time + config["q_delta"]
                    
                    # Q1, Q2, Q3 (avec progression typique)
                    q1_time = driver_q_time + 1.5  # Q1 environ 1.5s plus lent
                    q2_time = driver_q_time + 0.7  # Q2 environ 0.7s plus lent
                    q3_time = driver_q_time if config["grid"] <= 10 else 999  # Hors Q3 si grille > 10
                    
                    # Fastest lap time estimé (en millisecondes)
                    fastest_lap_ms = driver_q_time * 1000
                    
                    # Avg lap time (un peu plus lent que le fastest)
                    avg_lap_ms = fastest_lap_ms + 2500  # ~2.5s plus lent en moyenne
                    
                    try:
                        X = build_features_driver_win(
                            grid=config["grid"],
                            laps=laps,
                            q1_sec=q1_time,
                            q2_sec=q2_time,
                            q3_sec=q3_time,
                            fastest_lap_time=fastest_lap_ms,
                            avg_lap_ms=avg_lap_ms,
                            pit_stop_count=pit_stops,
                            avg_pit_duration_s=avg_pit_duration
                        )
                        
                        drivers_features.append((name, X, config))
                        
                    except Exception as e:
                        st.warning(f"Erreur pour {name}: {e}")
                
                # Calculer les probabilités avec Plackett-Luce
                if drivers_features:
                    try:
                        # Calculer theta pour chaque pilote
                        results = model_loader.predict_win_probabilities(
                            [(name, X) for name, X, _ in drivers_features]
                        )
                        
                        # Ajouter les infos supplémentaires
                        for i, (name, X, config) in enumerate(drivers_features):
                            results[i]["team"] = config["team"]
                            results[i]["number"] = config["number"]
                            results[i]["grid"] = config["grid"]
                        
                        # Créer le DataFrame et trier
                        df_pred = pd.DataFrame(results)
                        df_pred = df_pred.sort_values("win_probability", ascending=False).reset_index(drop=True)
                        
                        # PODIUM
                        st.markdown("## 🏆 Podium Prédit")
                        
                        podium = df_pred.head(3)
                        
                        col1, col2, col3 = st.columns([1, 1.3, 1])
                        
                        # 2ème
                        with col1:
                            if len(podium) >= 2:
                                d = podium.iloc[1]
                                color = get_team_color(d['team'])
                                st.markdown(f"""
                                <div class="podium-silver">
                                    <h2>🥈 2ème</h2>
                                    <h3>{d['driver']}</h3>
                                    <p style="color: {color}; font-weight: bold;">{d['team']}</p>
                                    <p style="font-size: 1.5rem;">{d['win_probability']*100:.2f}%</p>
                                    <p>θ = {d['theta']:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 1er
                        with col2:
                            if len(podium) >= 1:
                                d = podium.iloc[0]
                                color = get_team_color(d['team'])
                                st.markdown(f"""
                                <div class="podium-gold">
                                    <h1>🥇 VAINQUEUR</h1>
                                    <h2>{d['driver']}</h2>
                                    <p style="color: {color}; font-weight: bold;">{d['team']}</p>
                                    <p style="font-size: 2rem;">{d['win_probability']*100:.2f}%</p>
                                    <p>θ = {d['theta']:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 3ème
                        with col3:
                            if len(podium) >= 3:
                                d = podium.iloc[2]
                                color = get_team_color(d['team'])
                                st.markdown(f"""
                                <div class="podium-bronze">
                                    <h2>🥉 3ème</h2>
                                    <h3>{d['driver']}</h3>
                                    <p style="font-weight: bold;">{d['team']}</p>
                                    <p style="font-size: 1.5rem;">{d['win_probability']*100:.2f}%</p>
                                    <p>θ = {d['theta']:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Vérification : somme des probabilités
                        total_prob = df_pred["win_probability"].sum()
                        st.success(f"✅ Somme des probabilités : {total_prob*100:.2f}% (doit être ~100%)")
                        
                        st.markdown("---")
                        
                        # Graphique
                        st.subheader("📊 Probabilités de Victoire (Plackett-Luce)")
                        
                        df_sorted = df_pred.sort_values("win_probability", ascending=True)
                        
                        color_map = {team: get_team_color(team) for team in df_sorted["team"].unique()}
                        
                        fig = px.bar(
                            df_sorted,
                            x="win_probability",
                            y="driver",
                            orientation="h",
                            color="team",
                            color_discrete_map=color_map,
                            title=f"Probabilités de Victoire - {selected_race_name}",
                            labels={"win_probability": "Probabilité", "driver": "Pilote", "team": "Équipe"},
                            hover_data={"theta": ":.3f", "grid": True}
                        )
                        
                        fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                        fig.update_layout(
                            height=700,
                            yaxis={'categoryorder': 'total ascending'},
                            xaxis_tickformat='.0%'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Graphique des scores theta
                        st.subheader("📈 Scores de Force (θ)")
                        
                        fig_theta = px.bar(
                            df_sorted,
                            x="theta",
                            y="driver",
                            orientation="h",
                            color="team",
                            color_discrete_map=color_map,
                            title="Score θ (Force du pilote) - Plus élevé = Plus fort",
                            labels={"theta": "Score θ", "driver": "Pilote"}
                        )
                        
                        fig_theta.update_traces(texttemplate='%{x:.2f}', textposition='outside')
                        fig_theta.update_layout(
                            height=700,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig_theta, use_container_width=True)
                        
                        # Tableau complet
                        st.subheader("📋 Classement Complet")
                        
                        df_display = df_pred.copy()
                        df_display["Position"] = range(1, len(df_display) + 1)
                        df_display["Probabilité"] = df_display["win_probability"].apply(lambda x: f"{x*100:.2f}%")
                        df_display["Score θ"] = df_display["theta"].apply(lambda x: f"{x:.3f}")
                        
                        df_display = df_display.rename(columns={
                            "driver": "Pilote",
                            "team": "Équipe",
                            "number": "Numéro",
                            "grid": "Grille"
                        })
                        
                        st.dataframe(
                            df_display[["Position", "Pilote", "Équipe", "Numéro", "Grille", "Score θ", "Probabilité"]],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                    except Exception as e:
                        st.error(f"Erreur lors du calcul: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# =============================================================================
# TAB 2: TEMPS DE COURSE
# =============================================================================
with tab2:
    st.header("⏱️ Prédiction du Temps de Course")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Pilote")
        
        driver_names = [d["name"] for d in GRID_2025]
        selected_driver = st.selectbox("Sélectionnez un pilote", driver_names, key="time_driver")
        
        # Infos du pilote
        driver_info = next((d for d in GRID_2025 if d["name"] == selected_driver), None)
        
        if driver_info:
            color = get_team_color(driver_info["team"])
            st.markdown(f"""
            <div class="team-card" style="background-color: {color}30; border-left: 4px solid {color};">
                <strong>{driver_info['name']}</strong> #{driver_info['number']}<br>
                🏎️ {driver_info['team']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🏁 Circuit")
        
        circuit_options = [f"{r['name']} - {r['circuit']}" for r in F1_CALENDAR_2025]
        selected_circuit_full = st.selectbox("Circuit", circuit_options, key="time_circuit")
        selected_circuit_name = selected_circuit_full.split(" - ")[1]
        
        race_info = next((r for r in F1_CALENDAR_2025 if r["circuit"] == selected_circuit_name), None)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grid_pos = st.slider("Position grille", 1, 20, 1, key="time_grid")
    with col2:
        year = st.number_input("Année", value=2025, key="time_year")
    with col3:
        if race_info:
            st.metric("Circuit ID", race_info['circuit_id'])
    
    st.markdown("---")
    
    if st.button("⏱️ Prédire le Temps", type="primary", use_container_width=True):
        
        if "driver_time" not in model_loader.models:
            st.error("❌ Modèle temps non chargé")
        elif driver_info is None:
            st.error("Pilote non trouvé")
        else:
            with st.spinner("Calcul..."):
                try:
                    X = build_features_driver_time(
                        grid=grid_pos,
                        circuit_id=race_info['circuit_id'] if race_info else 1,
                        constructor_id=driver_info['constructor_id'],
                        driver_number=driver_info['number'],
                        year=year
                    )
                    
                    time_ms = model_loader.predict_race_time(X)[0]
                    time_str = milliseconds_to_time_string(time_ms)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>⏱️ Temps de Course Prédit</h2>
                        <p class="metric-big">{time_str}</p>
                        <p>{selected_driver} | {selected_circuit_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Temps (ms)", f"{time_ms:,.0f}")
                    col2.metric("Position grille", grid_pos)
                    if race_info:
                        avg_lap = time_ms / race_info['laps']
                        col3.metric("Moy/tour", milliseconds_to_time_string(avg_lap))
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")

# =============================================================================
# TAB 3: PERFORMANCE ÉQUIPE
# =============================================================================
# =============================================================================
# TAB 3: PERFORMANCE ÉQUIPE (K-Means Clustering)
# =============================================================================
with tab3:
    st.header("🏢 Classification des Équipes")
    st.markdown("""
    Ce modèle utilise le **K-Means Clustering** pour classifier les équipes F1 en 3 catégories :
    - 🏆 **Top Teams** : Les meilleures équipes (Red Bull, Ferrari, Mercedes, McLaren)
    - 🔵 **Mid-field** : Les équipes de milieu de grille
    - ⬇️ **Back-markers** : Les équipes du fond de grille
    """)
    
    st.markdown("---")
    
    # Valeurs par défaut estimées pour chaque équipe 2025
    team_defaults = {
        "Red Bull": {"points": 860, "quali_pace_ratio": 1.000},
        "Ferrari": {"points": 652, "quali_pace_ratio": 1.003},
        "McLaren": {"points": 640, "quali_pace_ratio": 1.002},
        "Mercedes": {"points": 425, "quali_pace_ratio": 1.005},
        "Aston Martin": {"points": 86, "quali_pace_ratio": 1.012},
        "RB": {"points": 46, "quali_pace_ratio": 1.018},
        "Haas F1 Team": {"points": 31, "quali_pace_ratio": 1.022},
        "Alpine": {"points": 13, "quali_pace_ratio": 1.020},
        "Williams": {"points": 17, "quali_pace_ratio": 1.025},
        "Kick Sauber": {"points": 0, "quali_pace_ratio": 1.030},
    }
    
    # Couleurs pour les tiers
    tier_colors = {
        "Top Teams": "#FFD700",      # Or
        "Mid-field": "#C0C0C0",      # Argent
        "Back-markers": "#CD7F32",   # Bronze
    }
    
    tier_emojis = {
        "Top Teams": "🏆",
        "Mid-field": "🔵",
        "Back-markers": "⬇️",
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Sélection d'une Équipe")
        
        teams = list(team_defaults.keys())
        selected_team = st.selectbox("Équipe", teams, key="team_cluster_select")
        
        defaults = team_defaults.get(selected_team, {"points": 100, "quali_pace_ratio": 1.015})
        
        color = get_team_color(selected_team)
        team_drivers = [d for d in GRID_2025 if d["team"] == selected_team]
        
        st.markdown(f"""
        <div class="team-card" style="background-color: {color}30; border-left: 4px solid {color}; padding: 15px; border-radius: 10px;">
            <h4>{selected_team}</h4>
            <p><strong>Pilotes 2025:</strong></p>
            {''.join([f"<p>• {d['name']} #{d['number']}</p>" for d in team_drivers])}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚙️ Paramètres de l'Équipe")
        
        st.markdown("""
        **Points** : Points totaux accumulés sur la saison.
        
        **Quali Pace Ratio** : Ratio entre le meilleur temps de qualification de l'équipe et le temps de la pole position.
        - `1.000` = Pole position
        - `1.010` = 1% plus lent que la pole
        - `1.020` = 2% plus lent que la pole
        """)
        
        points = st.number_input(
            "Points de l'équipe",
            min_value=0,
            max_value=1000,
            value=int(defaults["points"]),
            step=10,
            key="team_points"
        )
        
        quali_pace_ratio = st.slider(
            "Quali Pace Ratio",
            min_value=0.98,
            max_value=1.10,
            value=float(defaults["quali_pace_ratio"]),
            step=0.001,
            format="%.3f",
            key="team_quali_ratio"
        )
    
    st.markdown("---")
    
    # Bouton pour classifier l'équipe sélectionnée
    if st.button("🎯 Classifier cette Équipe", type="secondary", use_container_width=True):
        
        if "team_perf" not in model_loader.models:
            st.error("❌ Modèle team_perf non chargé. Placez model_team_perf.pkl dans models/")
        else:
            try:
                X = build_features_team_perf(
                    points=points,
                    quali_pace_ratio=quali_pace_ratio
                )
                
                cluster, tier_label, tier_name = model_loader.predict_team_tier(X)
                
                color = get_team_color(selected_team)
                tier_color = tier_colors.get(tier_name, "#FFFFFF")
                tier_emoji = tier_emojis.get(tier_name, "")
                
                st.markdown(f"""
                <div class="result-card" style="border-left-color: {color};">
                    <h2>{tier_emoji} Classification: {tier_name}</h2>
                    <p style="font-size: 1.5rem;"><strong>{selected_team}</strong></p>
                    <p>Points: {points} | Quali Pace Ratio: {quali_pace_ratio:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    st.markdown("---")
    
    # Classification de toutes les équipes
    st.subheader("📊 Classification de Toutes les Équipes")
    
    if st.button("🔄 Classifier Toutes les Équipes (Données 2024)", type="primary", use_container_width=True):
        
        if "team_perf" not in model_loader.models:
            st.error("❌ Modèle team_perf non chargé")
        else:
            with st.spinner("Classification en cours..."):
                
                results = []
                
                for team, defaults in team_defaults.items():
                    try:
                        X = build_features_team_perf(
                            points=defaults["points"],
                            quali_pace_ratio=defaults["quali_pace_ratio"]
                        )
                        
                        cluster, tier_label, tier_name = model_loader.predict_team_tier(X)
                        
                        results.append({
                            "Équipe": team,
                            "Points": defaults["points"],
                            "Quali Pace Ratio": defaults["quali_pace_ratio"],
                            "Tier": tier_name,
                            "Tier_Label": tier_label,
                        })
                        
                    except Exception as e:
                        st.warning(f"Erreur pour {team}: {e}")
                
                if results:
                    df_results = pd.DataFrame(results)
                    df_results = df_results.sort_values("Points", ascending=False).reset_index(drop=True)
                    
                    # Affichage par catégorie
                    st.markdown("### 🏆 Top Teams")
                    top_teams = df_results[df_results["Tier"] == "Top Teams"]
                    if not top_teams.empty:
                        for _, row in top_teams.iterrows():
                            color = get_team_color(row["Équipe"])
                            st.markdown(f"""
                            <div style="background-color: {color}30; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {color};">
                                <strong>{row['Équipe']}</strong> - {row['Points']} pts
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune équipe dans cette catégorie")
                    
                    st.markdown("### 🔵 Mid-field")
                    mid_teams = df_results[df_results["Tier"] == "Mid-field"]
                    if not mid_teams.empty:
                        for _, row in mid_teams.iterrows():
                            color = get_team_color(row["Équipe"])
                            st.markdown(f"""
                            <div style="background-color: {color}30; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {color};">
                                <strong>{row['Équipe']}</strong> - {row['Points']} pts
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune équipe dans cette catégorie")
                    
                    st.markdown("### ⬇️ Back-markers")
                    back_teams = df_results[df_results["Tier"] == "Back-markers"]
                    if not back_teams.empty:
                        for _, row in back_teams.iterrows():
                            color = get_team_color(row["Équipe"])
                            st.markdown(f"""
                            <div style="background-color: {color}30; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {color};">
                                <strong>{row['Équipe']}</strong> - {row['Points']} pts
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune équipe dans cette catégorie")
                    
                    st.markdown("---")
                    
                    # Graphique Scatter
                    st.subheader("📈 Visualisation des Clusters")
                    
                    color_map_tier = {
                        "Top Teams": "#FFD700",
                        "Mid-field": "#1E90FF", 
                        "Back-markers": "#FF6347"
                    }
                    
                    fig = px.scatter(
                        df_results,
                        x="Points",
                        y="Quali Pace Ratio",
                        color="Tier",
                        color_discrete_map=color_map_tier,
                        hover_name="Équipe",
                        size=[50] * len(df_results),
                        title="Classification des Équipes (Points vs Quali Pace Ratio)",
                        labels={
                            "Points": "Points Saison",
                            "Quali Pace Ratio": "Ratio Pace Qualification"
                        }
                    )
                    
                    # Ajouter les noms des équipes
                    fig.update_traces(
                        textposition='top center',
                        marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey'))
                    )
                    
                    # Ajouter les annotations pour chaque équipe
                    for _, row in df_results.iterrows():
                        fig.add_annotation(
                            x=row["Points"],
                            y=row["Quali Pace Ratio"],
                            text=row["Équipe"],
                            showarrow=False,
                            yshift=15,
                            font=dict(size=10)
                        )
                    
                    fig.update_layout(
                        height=500,
                        yaxis=dict(autorange="reversed")  # Inverser l'axe Y (plus petit ratio = mieux)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau récapitulatif
                    st.subheader("📋 Tableau Récapitulatif")
                    
                    df_display = df_results[["Équipe", "Points", "Quali Pace Ratio", "Tier"]].copy()
                    df_display["Quali Pace Ratio"] = df_display["Quali Pace Ratio"].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: CLASSEMENTS
# =============================================================================
with tab4:
    st.header("📊 Classements")
    
    available_years = data_loader.get_unique_years() if data_loaded else [2024, 2023, 2022]
    selected_year = st.selectbox("Année", available_years, key="standings_year")
    
    subtab1, subtab2 = st.tabs(["🏆 Pilotes", "🏢 Constructeurs"])
    
    with subtab1:
        st.subheader(f"Classement Pilotes {selected_year}")
        
        if data_loaded:
            standings = data_loader.get_latest_standings(selected_year)
            
            if standings is not None and not standings.empty:
                st.dataframe(standings, use_container_width=True, hide_index=True)
                
                fig = px.bar(
                    standings.head(10),
                    x="Pilote",
                    y="Points",
                    title="Top 10 Pilotes",
                    color="Points",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Pas de données pour {selected_year}")
        else:
            st.warning("Données non chargées")
    
    with subtab2:
        st.subheader(f"Classement Constructeurs {selected_year}")
        
        if data_loaded:
            standings = data_loader.get_constructor_standings(selected_year)
            
            if standings is not None and not standings.empty:
                # Ajouter les couleurs
                standings["Couleur"] = standings["Constructeur"].apply(get_team_color)
                
                st.dataframe(
                    standings[["Position", "Constructeur", "Points"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                color_map = {row["Constructeur"]: row["Couleur"] for _, row in standings.iterrows()}
                
                fig = px.bar(
                    standings.sort_values("Points"),
                    x="Points",
                    y="Constructeur",
                    orientation="h",
                    color="Constructeur",
                    color_discrete_map=color_map,
                    title="Points par Constructeur"
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Pas de données pour {selected_year}")
        else:
            st.warning("Données non chargées")

# =============================================================================
# TAB 5: CALENDRIER
# =============================================================================
with tab5:
    st.header("📅 Calendrier F1 2025")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Terminées")
        
        for race in F1_CALENDAR_2025:
            if race["completed"]:
                st.markdown(f"""
                <div style="background-color: #1E3A1E; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #00FF00;">
                    <strong>{race['name']}</strong><br>
                    📍 {race['circuit']} | 📅 {race['date']}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔜 À Venir")
        
        for race in F1_CALENDAR_2025:
            if not race["completed"]:
                st.markdown(f"""
                <div style="background-color: #3A2A1E; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #FF8700;">
                    <strong>{race['name']}</strong><br>
                    📍 {race['circuit']} | 📅 {race['date']}
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    🏎️ F1 Prediction App | Dataset: FinalCombinedCleanFinal.csv | 
    Créé avec Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)