"""
╔══════════════════════════════════════════════════════════════════╗
║          💻 LAPTOP MARKET PRICE ESTIMATOR - app.py               ║
║          Application Streamlit interactive & moderne             ║
╚══════════════════════════════════════════════════════════════════╝

Architecture :
  ├── Page 1 : 🏠 Accueil
  ├── Page 2 : 🧮 Price Estimator  (prédiction interactive)
  ├── Page 3 : 📊 Market Insights  (tendances du marché)
  └── Page 4 : 🔮 Model Insights   (SHAP & Feature Importance)
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from catboost import CatBoostRegressor, Pool

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Price Estimator",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '💻 Laptop Market Price Estimator | Data Science Portfolio'
    }
)

# ─── CSS Personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Variables & Thème ── */
:root {
    --primary: #4F8EF7;
    --primary-dark: #2563EB;
    --accent: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --bg-card: #1E293B;
    --bg-card2: #0F172A;
    --text-muted: #94A3B8;
    --border: rgba(79, 142, 247, 0.25);
}

/* ── Fond général ── */
.stApp {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D1B2A 50%, #0A0F1E 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1E293B 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* ── Labels des widgets : sliders, selectbox, radio, etc. ── */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
div[data-testid="stWidgetLabel"] label {
    color: #F1F5F9 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Valeurs et steps affichés sous les sliders ── */
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span {
    color: #CBD5E1 !important;
}

/* ── Valeur sélectionnée dans les selectbox ── */
div[data-baseweb="select"] span {
    color: #F1F5F9 !important;
}

/* ── Texte général (paragraphes dans le main) ── */
.stApp p {
    color: #CBD5E1;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(79, 142, 247, 0.2);
}
div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.85rem; }
div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-weight: 700; }

/* ── Cards personnalisées ── */
.price-card {
    background: linear-gradient(135deg, #1E3A5F, #0F172A);
    border: 2px solid #4F8EF7;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(79, 142, 247, 0.3);
    animation: glowPulse 3s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(79, 142, 247, 0.3); }
    50% { box-shadow: 0 0 50px rgba(79, 142, 247, 0.6); }
}
.price-card h1 { color: #4F8EF7; font-size: 3.5rem; margin: 0; font-weight: 800; }
.price-card .label { color: #94A3B8; font-size: 1rem; margin-bottom: 0.5rem; }
.price-card .fcfa { color: #10B981; font-size: 1.6rem; font-weight: 600; margin-top: 0.5rem; }

.info-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F1F5F9;
    border-left: 4px solid #4F8EF7;
    padding-left: 1rem;
    margin-bottom: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-blue { background: rgba(79, 142, 247, 0.2); color: #4F8EF7; border: 1px solid #4F8EF7; }
.badge-green { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
.badge-orange { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4F8EF7, #10B981, #F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

/* Sliders & inputs */
.stSlider > div > div > div { background: var(--primary) !important; }
.stSelectbox, .stSlider { background: transparent !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    background: transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(79, 142, 247, 0.15) !important;
    color: #4F8EF7 !important;
    border-bottom: 2px solid #4F8EF7;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────────────────────
USD_TO_FCFA = 600
MODEL_PATH = "models/laptop_price_model.pkl"

# ─── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle CatBoost et ses métadonnées depuis le fichier PKL."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None

# ─── Données de démonstration (si pas de modèle) ─────────────────────────────
def get_demo_data():
    """Génère des données de démonstration pour le mode preview."""
    np.random.seed(42)
    n = 1000
    brands = ['Apple', 'Dell', 'HP', 'Lenovo', 'ASUS', 'Acer', 'MSI', 'Samsung', 'Razer', 'LG']
    return pd.DataFrame({
        'brand': np.random.choice(brands, n),
        'ram_gb': np.random.choice([8, 16, 32, 64], n, p=[0.3, 0.4, 0.2, 0.1]),
        'storage_gb': np.random.choice([256, 512, 1024, 2048], n),
        'gpu_tier': np.random.randint(1, 8, n),
        'cpu_tier': np.random.randint(1, 6, n),
        'display_size_in': np.random.choice([13.3, 14.0, 15.6, 16.0, 17.3], n),
        'release_year': np.random.choice([2021, 2022, 2023, 2024], n),
        'price': np.random.lognormal(7.5, 0.7, n).clip(300, 8000)
    })

# ─── Fonctions utilitaires ─────────────────────────────────────────────────────
def format_price_usd(price: float) -> str:
    return f"${price:,.0f}"

def format_price_fcfa(price: float) -> str:
    fcfa = price * USD_TO_FCFA
    if fcfa >= 1_000_000:
        return f"{fcfa/1_000_000:.1f}M FCFA"
    return f"{fcfa:,.0f} FCFA"

def get_price_segment(price: float) -> tuple[str, str]:
    if price < 500: return "Budget", "#F59E0B"
    elif price < 1000: return "Milieu de gamme", "#4F8EF7"
    elif price < 2000: return "Premium", "#10B981"
    else: return "Ultra-Premium", "#EF4444"

def compute_performance_score(cpu_cores, cpu_boost_ghz, gpu_tier, vram_gb, ram_gb) -> float:
    """Score de performance normalisé [0-100] basé sur les composants clés."""
    cpu_score  = (cpu_cores * cpu_boost_ghz) / (16 * 5.5)   # Normalise par un i9
    gpu_score  = (gpu_tier * vram_gb) / (10 * 24)            # Normalise par RTX 4090
    ram_score  = np.log2(ram_gb) / np.log2(128)              # Normalise par 128GB
    score = (0.35 * cpu_score + 0.40 * gpu_score + 0.25 * ram_score)
    return min(round(score * 100, 1), 100.0)

def plotly_dark_layout(**kwargs):
    """Template Plotly commun pour un look cohérent."""
    return dict(
        template='plotly_dark',
        paper_bgcolor='rgba(14,20,37,0)',
        plot_bgcolor='rgba(14,20,37,0)',
        font=dict(family='Inter, sans-serif', color='#94A3B8', size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        **kwargs
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>💻</div>
        <h2 style='color:#4F8EF7; margin:0; font-size:1.1rem;'>Laptop Price<br>Estimator</h2>
        <p style='color:#94A3B8; font-size:0.75rem; margin-top:0.3rem;'>Powered by CatBoost + SHAP</p>
    </div>
    <hr style='border-color: rgba(79,142,247,0.3); margin: 0.5rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🧮 Price Estimator", "📊 Market Insights", "🔮 Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(79,142,247,0.2);'>", unsafe_allow_html=True)

    # Chargement du modèle
    with st.spinner("Chargement du modèle..."):
        metadata = load_model()

    if metadata:
        metrics = metadata.get('metrics', {})
        st.markdown("**📊 Performances Modèle**")
        st.metric("R² Score", f"{metrics.get('r2_test', 0):.3f}")
        st.metric("MAE", format_price_usd(metrics.get('mae_test', 0)))
        st.metric("MAPE", f"{metrics.get('mape_test', 0):.1f}%")
        st.markdown(f"""
        <div style='background:rgba(16,185,129,0.1); border:1px solid #10B981;
                    border-radius:8px; padding:0.8rem; margin-top:0.5rem;'>
            <p style='color:#10B981; font-size:0.75rem; margin:0;'>
            ✅ Modèle chargé<br>
            🔢 {metadata.get('train_size',0):,} exemples d'entraînement
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.1); border:1px solid #F59E0B;
                    border-radius:8px; padding:0.8rem;'>
            <p style='color:#F59E0B; font-size:0.8rem; margin:0;'>
            ⚠️ Mode Démo<br>Entraînez le modèle pour activer toutes les fonctionnalités.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("👉 Exécutez le notebook `eda_and_training.ipynb` d'abord")

    st.markdown("""
    <div style='position:absolute; bottom:1rem; left:1rem; right:1rem; text-align:center;'>
        <p style='color:#475569; font-size:0.7rem;'>
        💡 1 USD ≈ 600 FCFA<br>
        Portfolio Data Science 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    col_hero, col_img = st.columns([3, 2])

    with col_hero:
        st.markdown("""
        <div class='hero-title'>
            Laptop Market<br>Price Estimator
        </div>
        <p style='color:#94A3B8; font-size:1.15rem; margin: 1rem 0 2rem 0; max-width:500px;'>
        Un modèle de Machine Learning avancé qui prédit le prix de marché d'un laptop
        à partir de ses spécifications techniques, avec une précision de ~85%.
        </p>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("""
            <div class='badge badge-blue'>⚡ CatBoost Regressor</div>
            <div class='badge badge-green' style='margin-top:0.5rem;'>🔮 SHAP Explainability</div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown("""
            <div class='badge badge-orange'>📊 ~100K Laptops</div>
            <div class='badge badge-blue' style='margin-top:0.5rem;'>🌍 33 Features</div>
            """, unsafe_allow_html=True)

    with col_img:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=85.4,
            title={'text': "Précision Modèle (%)", 'font': {'size': 14, 'color': '#94A3B8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#4F8EF7'},
                'bar': {'color': '#4F8EF7'},
                'bgcolor': 'rgba(30,41,59,0.8)',
                'steps': [
                    {'range': [0, 60],  'color': 'rgba(239,68,68,0.2)'},
                    {'range': [60, 80], 'color': 'rgba(245,158,11,0.2)'},
                    {'range': [80, 100],'color': 'rgba(16,185,129,0.2)'}
                ],
                'threshold': {'line': {'color': '#10B981', 'width': 3}, 'value': 80}
            },
            number={'suffix': "%", 'font': {'color': '#4F8EF7', 'size': 40}}
        ))
        fig.update_layout(**plotly_dark_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Features cards
    st.markdown("<div class='section-title'>🎯 Ce que vous pouvez faire</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    features = [
        ("🧮", "Price Estimator", "Saisissez les specs d'un laptop et obtenez son prix estimé en USD et FCFA instantanément."),
        ("📊", "Market Insights", "Explorez les tendances du marché : prix par marque, RAM, GPU, année de sortie."),
        ("🔮", "Model Insights", "Comprenez quels facteurs influencent le plus le prix grâce à SHAP et la feature importance."),
        ("📍", "Positionnement", "Visualisez où votre laptop se situe dans la distribution globale des prix."),
    ]

    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"""
            <div class='info-card' style='text-align:center;'>
                <div style='font-size:2rem;'>{icon}</div>
                <h3 style='color:#F1F5F9; font-size:1rem; margin:0.5rem 0;'>{title}</h3>
                <p style='color:#94A3B8; font-size:0.85rem; margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech stack
    st.markdown("<div class='section-title'>🛠️ Stack Technique</div>", unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color:#4F8EF7;'>🤖 Machine Learning</h4>
            <ul style='color:#94A3B8; font-size:0.9rem;'>
                <li>CatBoostRegressor (gradient boosting)</li>
                <li>SHAP (explicabilité)</li>
                <li>Scikit-learn (évaluation)</li>
                <li>Joblib (sérialisation)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color:#10B981;'>📊 Data & Visualisation</h4>
            <ul style='color:#94A3B8; font-size:0.9rem;'>
                <li>Pandas & NumPy</li>
                <li>Plotly (graphiques interactifs)</li>
                <li>Seaborn & Matplotlib (EDA)</li>
                <li>Feature Engineering custom</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color:#F59E0B;'>🌐 Application</h4>
            <ul style='color:#94A3B8; font-size:0.9rem;'>
                <li>Streamlit (framework web)</li>
                <li>CSS animations personnalisées</li>
                <li>Design responsive moderne</li>
                <li>UX orientée utilisateur</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : PRICE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Price Estimator":
    st.markdown("<div class='section-title'>🧮 Estimateur de Prix Laptop</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Configurez les spécifications techniques pour obtenir une estimation de prix.</p>",
                unsafe_allow_html=True)

    # Valeurs par défaut
    default_brands = ['Acer', 'Apple', 'ASUS', 'Dell', 'HP', 'Huawei', 'Lenovo',
                      'LG', 'MSI', 'Razer', 'Samsung', 'Toshiba', 'Other']
    default_os = ['Windows 11', 'Windows 10', 'macOS', 'ChromeOS', 'Linux', 'FreeDOS']
    default_cpu_brands = ['Intel', 'AMD', 'Apple', 'Qualcomm']
    default_gpu_brands = ['NVIDIA', 'AMD', 'Intel', 'Apple']
    default_display_types = ['IPS', 'OLED', 'VA', 'TN', 'AMOLED', 'Mini-LED']
    default_storage_types = ['SSD', 'HDD', 'eMMC', 'NVMe SSD', 'Hybrid']
    default_wifi = ['WiFi 6', 'WiFi 6E', 'WiFi 5', 'WiFi 4', 'WiFi 7']
    default_form = ['Clamshell', '2-in-1', 'Gaming', 'Ultrabook', 'Workstation']

    # Récupérer options depuis métadonnées si disponibles
    if metadata:
        brand_options = metadata.get('brand_options', default_brands) or default_brands
        os_options = metadata.get('os_options', default_os) or default_os
    else:
        brand_options, os_options = default_brands, default_os

    # ─── INPUTS ───────────────────────────────────────────────────────────────
    with st.container():
        tab_device, tab_cpu, tab_gpu, tab_display, tab_storage, tab_other = st.tabs([
            "📱 Appareil", "⚙️ CPU", "🎮 GPU", "🖥️ Écran", "💾 Stockage", "🔧 Autres"
        ])

        with tab_device:
            c1, c2, c3 = st.columns(3)
            with c1:
                brand = st.selectbox("🏷️ Marque", sorted(brand_options))
                os_sel = st.selectbox("💿 Système d'exploitation", default_os)
            with c2:
                release_year = st.slider("📅 Année de sortie", 2015, 2025, 2023)
                form_factor = st.selectbox("📐 Form Factor", default_form)
            with c3:
                weight_kg = st.slider("⚖️ Poids (kg)", 0.8, 4.5, 1.8, step=0.1)
                warranty_months = st.select_slider("🛡️ Garantie (mois)", [0, 12, 24, 36, 48], value=12)

        with tab_cpu:
            c1, c2, c3 = st.columns(3)
            with c1:
                cpu_brand = st.selectbox("🏭 Marque CPU", default_cpu_brands)
                cpu_tier = st.slider("📊 Tier CPU (1=bas, 5=haut)", 1, 5, 3)
            with c2:
                cpu_cores = st.select_slider("🔢 Nombre de cœurs", [2, 4, 6, 8, 10, 12, 14, 16, 20, 24], value=8)
                cpu_threads = st.select_slider("🧵 Threads", [4, 8, 12, 16, 20, 24, 28, 32, 40, 48], value=16)
            with c3:
                cpu_base_ghz = st.slider("📉 Fréquence Base (GHz)", 0.8, 4.5, 2.4, step=0.1)
                cpu_boost_ghz = st.slider("📈 Fréquence Boost (GHz)", 1.5, 6.0, 4.5, step=0.1)

            # Preview CPU Power
            cpu_power = cpu_cores * cpu_boost_ghz
            st.metric("⚡ CPU Power Score", f"{cpu_power:.1f}")

        with tab_gpu:
            c1, c2, c3 = st.columns(3)
            with c1:
                gpu_brand = st.selectbox("🏭 Marque GPU", default_gpu_brands)
                gpu_tier = st.slider("📊 Tier GPU (1=intégré, 10=haut de gamme)", 1, 10, 4)
            with c2:
                vram_gb = st.select_slider("🎨 VRAM (GB)", [0, 2, 4, 6, 8, 12, 16, 24], value=4)
            with c3:
                gpu_power = gpu_tier * vram_gb
                st.metric("🎮 GPU Power Score", f"{gpu_power}")
                st.markdown(f"""
                <div class='info-card' style='padding:0.8rem;'>
                    <p style='color:#94A3B8; font-size:0.8rem; margin:0;'>
                    {'🟢 Haute performance' if gpu_tier >= 7 else '🟡 Milieu de gamme' if gpu_tier >= 4 else '🔵 Usage basique'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab_display:
            c1, c2, c3 = st.columns(3)
            with c1:
                display_type = st.selectbox("🖥️ Type d'écran", default_display_types)
                display_size = st.slider("📏 Taille (pouces)", 11.0, 18.0, 15.6, step=0.1)
            with c2:
                resolutions = ['1280x720', '1366x768', '1920x1080', '2560x1440',
                               '2560x1600', '3840x2160', '3840x2400']
                resolution = st.selectbox("📺 Résolution", resolutions, index=2)
                x_res, y_res = map(int, resolution.split('x'))
            with c3:
                refresh_hz = st.select_slider("🔄 Taux de rafraîchissement (Hz)",
                                              [60, 90, 120, 144, 165, 240, 360], value=60)
                # Calcul PPI
                ppi = np.sqrt(x_res**2 + y_res**2) / display_size
                st.metric("🔬 PPI (Densité)", f"{ppi:.0f}")

        with tab_storage:
            c1, c2, c3 = st.columns(3)
            with c1:
                storage_type = st.selectbox("💾 Type de stockage", default_storage_types)
                storage_gb = st.select_slider("📦 Capacité (GB)",
                                              [128, 256, 512, 1024, 2048, 4096], value=512)
            with c2:
                storage_drives = st.slider("🔢 Nombre de disques", 1, 4, 1)
                ram_gb = st.select_slider("🧠 RAM (GB)", [4, 8, 16, 32, 64, 128], value=16)
            with c3:
                storage_per_drive = storage_gb / storage_drives
                st.metric("💡 Stockage/Disque", f"{storage_per_drive:.0f} GB")

        with tab_other:
            c1, c2, c3 = st.columns(3)
            with c1:
                wifi = st.selectbox("📶 WiFi", default_wifi)
                bluetooth = st.selectbox("🔵 Bluetooth", ["5.3", "5.2", "5.1", "5.0", "4.2"])
            with c2:
                battery_wh = st.slider("🔋 Batterie (Wh)", 20, 150, 60, step=5)
                charger_watts = st.select_slider("⚡ Chargeur (W)", [30, 45, 60, 65, 90, 100, 140, 180, 230], value=65)

    st.markdown("---")

    # ─── BOUTON PRÉDICTION ────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("🚀 Estimer le Prix", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("🔮 Le modèle calcule le prix optimal..."):
            # Construction du vecteur de features
            input_data = {
                'brand': brand,
                'release_year': release_year,
                'os': os_sel,
                'form_factor': form_factor,
                'cpu_brand': cpu_brand,
                'cpu_tier': cpu_tier,
                'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads,
                'cpu_base_ghz': cpu_base_ghz,
                'cpu_boost_ghz': cpu_boost_ghz,
                'gpu_brand': gpu_brand,
                'gpu_tier': gpu_tier,
                'vram_gb': vram_gb,
                'ram_gb': ram_gb,
                'storage_type': storage_type,
                'storage_gb': storage_gb,
                'storage_drive_count': storage_drives,
                'display_type': display_type,
                'display_size_in': display_size,
                'refresh_hz': refresh_hz,
                'x_resolution': x_res,
                'y_resolution': y_res,
                'battery_wh': battery_wh,
                'charger_watts': charger_watts,
                'wifi': wifi,
                'bluetooth': bluetooth,
                'weight_kg': weight_kg,
                'warranty_months': warranty_months,
                # Features engineered
                'cpu_power': cpu_cores * cpu_boost_ghz,
                'gpu_power': gpu_tier * vram_gb,
                'ppi': ppi,
                'storage_per_drive': storage_per_drive,
            }

            input_df = pd.DataFrame([input_data])

            # Prédiction
            if metadata:
                model_obj = metadata['model']
                feat_cols = metadata['feature_columns']
                cat_feats = metadata['cat_features']

                # Aligner les colonnes
                for col in feat_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                for col in cat_feats:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)

                input_df = input_df[feat_cols]
                cat_idx = [input_df.columns.get_loc(c) for c in cat_feats if c in input_df.columns]
                pred_pool = Pool(input_df, cat_features=cat_idx)
                predicted_price = float(model_obj.predict(pred_pool)[0])

                # Récupérer les prix du dataset pour le graphique
                all_prices = metadata['price_stats']['all_prices']
            else:
                # Mode démo : simulation
                base_price = 200 + ram_gb * 15 + storage_gb * 0.3 + gpu_tier * 80 + cpu_tier * 60
                predicted_price = base_price * (0.8 + np.random.random() * 0.4)
                all_prices = list(np.random.lognormal(7.5, 0.7, 1000).clip(200, 8000))

        # ─── AFFICHAGE DU RÉSULTAT ─────────────────────────────────────────
        predicted_price = max(200, min(predicted_price, 15000))
        price_fcfa = predicted_price * USD_TO_FCFA
        segment, seg_color = get_price_segment(predicted_price)
        perf_score = compute_performance_score(cpu_cores, cpu_boost_ghz, gpu_tier, vram_gb, ram_gb)

        # Carte prix principale
        st.markdown(f"""
        <div class='price-card'>
            <p class='label'>💰 PRIX ESTIMÉ</p>
            <h1>${predicted_price:,.0f}</h1>
            <p class='fcfa'>{format_price_fcfa(predicted_price)}</p>
            <div class='badge' style='background:rgba({','.join(str(int(c*255)) for c in [0.31,0.56,0.97])},0.2);
                 color:{seg_color}; border:1px solid {seg_color}; margin-top:0.5rem;'>
                {segment}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Métriques rapides
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 USD", f"${predicted_price:,.0f}")
        m2.metric("🌍 FCFA", format_price_fcfa(predicted_price))
        m3.metric("🏆 Segment", segment)
        m4.metric("⚡ Perf Score", f"{perf_score}/100")

        st.markdown("---")

        # ─── GRAPHIQUES POST-PRÉDICTION ────────────────────────────────────
        st.markdown("### 📊 Positionnement sur le Marché")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Distribution des prix avec position du laptop
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=all_prices, nbinsx=80,
                marker_color='#4F8EF7', opacity=0.7,
                name='Marché'
            ))
            fig_dist.add_vline(
                x=predicted_price, line_width=3,
                line_color='#EF4444', line_dash='dash',
                annotation_text=f"Votre laptop: ${predicted_price:,.0f}",
                annotation_font_color='#EF4444',
                annotation_font_size=12
            )
            percentile = sum(1 for p in all_prices if p < predicted_price) / len(all_prices) * 100
            fig_dist.update_layout(
                title=f"📍 Position dans la distribution (top {100-percentile:.0f}%)",
                xaxis_title="Prix (USD)", yaxis_title="Fréquence",
                **plotly_dark_layout(height=350)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_g2:
            # Radar chart performance
            categories = ['CPU', 'GPU', 'RAM', 'Stockage', 'Écran', 'Batterie']
            cpu_score_norm = min(cpu_power / 88, 1) * 100
            gpu_score_norm = min(gpu_power / 80, 1) * 100
            ram_score_norm = min(ram_gb / 128, 1) * 100
            storage_score_norm = min(storage_gb / 4096, 1) * 100
            display_score_norm = min(ppi / 400, 1) * 100
            battery_score_norm = min(battery_wh / 150, 1) * 100

            values = [cpu_score_norm, gpu_score_norm, ram_score_norm,
                      storage_score_norm, display_score_norm, battery_score_norm]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(79, 142, 247, 0.25)',
                line=dict(color='#4F8EF7', width=2),
                name='Votre laptop'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor='rgba(79,142,247,0.2)', tickfont=dict(color='#94A3B8')),
                    angularaxis=dict(gridcolor='rgba(79,142,247,0.2)', tickfont=dict(color='#94A3B8'))
                ),
                title="⚡ Profil de Performance",
                showlegend=False,
                **plotly_dark_layout(height=350)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ─── Score de performance détaillé ────────────────────────────────
        st.markdown("### 🏆 Analyse du Laptop")
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            # Gauge de performance
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=perf_score,
                delta={'reference': 50, 'increasing': {'color': '#10B981'},
                       'decreasing': {'color': '#EF4444'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#4F8EF7'},
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(239,68,68,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 80], 'color': 'rgba(79,142,247,0.15)'},
                        {'range': [80, 100],'color': 'rgba(16,185,129,0.15)'}
                    ],
                    'threshold': {'line': {'color': '#10B981', 'width': 3}, 'value': 80}
                },
                title={'text': "Score Performance Global", 'font': {'size': 14, 'color': '#94A3B8'}},
                number={'font': {'color': '#4F8EF7', 'size': 50}, 'suffix': '/100'}
            ))
            fig_gauge.update_layout(**plotly_dark_layout(height=300))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_a2:
            # Fiche récapitulative
            st.markdown(f"""
            <div class='info-card'>
                <h4 style='color:#4F8EF7; margin-bottom:1rem;'>📋 Résumé de Configuration</h4>
                <table style='width:100%; color:#E2E8F0; font-size:0.9rem;'>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>Marque</td><td><b>{brand}</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>CPU</td><td><b>{cpu_brand} Tier {cpu_tier} ({cpu_cores}c/{cpu_threads}t @ {cpu_boost_ghz}GHz)</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>GPU</td><td><b>{gpu_brand} Tier {gpu_tier} ({vram_gb}GB VRAM)</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>RAM</td><td><b>{ram_gb} GB</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>Stockage</td><td><b>{storage_gb} GB {storage_type}</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>Écran</td><td><b>{display_size}" {resolution} {refresh_hz}Hz</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>Batterie</td><td><b>{battery_wh} Wh</b></td></tr>
                    <tr><td style='color:#94A3B8; padding:0.3rem 0;'>Année</td><td><b>{release_year}</b></td></tr>
                    <tr style='border-top:1px solid rgba(79,142,247,0.3); padding-top:0.5rem;'>
                        <td style='color:#F59E0B; padding:0.5rem 0;'><b>Prix Estimé</b></td>
                        <td><b style='color:#4F8EF7; font-size:1.2rem;'>${predicted_price:,.0f}</b></td>
                    </tr>
                    <tr>
                        <td style='color:#94A3B8;'>Équivalent FCFA</td>
                        <td><b style='color:#10B981;'>{format_price_fcfa(predicted_price)}</b></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Insights":
    st.markdown("<div class='section-title'>📊 Market Insights</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Explorez les tendances du marché des laptops.</p>", unsafe_allow_html=True)

    # Charger les données
    demo = get_demo_data()
    if metadata and 'price_stats' in metadata:
        ps = metadata['price_stats']
        # Reconstituer un mini-df si possible
        df_insight = demo  # Fallback sur données de démo
        mean_p = ps['mean']
        median_p = ps['median']
        min_p = ps['min']
        max_p = ps['max']
    else:
        df_insight = demo
        mean_p = demo['price'].mean()
        median_p = demo['price'].median()
        min_p = demo['price'].min()
        max_p = demo['price'].max()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Prix Moyen", format_price_usd(mean_p))
    k2.metric("📊 Prix Médian", format_price_usd(median_p))
    k3.metric("📉 Prix Min", format_price_usd(min_p))
    k4.metric("📈 Prix Max", format_price_usd(max_p))

    st.markdown("---")

    tab_brand, tab_specs, tab_trends = st.tabs(["🏷️ Par Marque", "⚙️ Par Specs", "📅 Tendances"])

    with tab_brand:
        col1, col2 = st.columns(2)
        with col1:
            brand_price = df_insight.groupby('brand')['price'].agg(['median', 'count']).reset_index()
            brand_price = brand_price[brand_price['count'] > 5].sort_values('median', ascending=True)
            fig = px.bar(brand_price, x='median', y='brand', orientation='h',
                         color='median', color_continuous_scale='Blues',
                         title="💰 Prix Médian par Marque",
                         labels={'median': 'Prix Médian (USD)', 'brand': 'Marque'})
            fig.update_layout(**plotly_dark_layout(height=400), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            brand_vol = df_insight['brand'].value_counts().head(10).reset_index()
            brand_vol.columns = ['brand', 'count']
            fig = px.pie(brand_vol, values='count', names='brand',
                         title="📦 Volume par Marque (Top 10)",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(**plotly_dark_layout(height=400))
            st.plotly_chart(fig, use_container_width=True)

    with tab_specs:
        col1, col2 = st.columns(2)
        with col1:
            ram_med = df_insight.groupby('ram_gb')['price'].median().reset_index()
            fig = px.bar(ram_med, x='ram_gb', y='price',
                         title="🧠 Prix Médian par RAM (GB)",
                         color='price', color_continuous_scale='Viridis',
                         labels={'ram_gb': 'RAM (GB)', 'price': 'Prix Médian (USD)'})
            fig.update_layout(**plotly_dark_layout(height=350), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gpu_med = df_insight.groupby('gpu_tier')['price'].median().reset_index()
            fig = px.line(gpu_med, x='gpu_tier', y='price',
                          title="🎮 Prix Médian par GPU Tier",
                          markers=True,
                          labels={'gpu_tier': 'GPU Tier', 'price': 'Prix Médian (USD)'})
            fig.update_traces(line=dict(color='#4F8EF7', width=3),
                              marker=dict(size=10, color='#F59E0B'))
            fig.update_layout(**plotly_dark_layout(height=350))
            st.plotly_chart(fig, use_container_width=True)

        # Scatter RAM vs Prix coloré par GPU tier
        fig_scatter = px.scatter(df_insight, x='ram_gb', y='price', color='gpu_tier',
                                 size='storage_gb', opacity=0.6,
                                 title="🔬 RAM vs Prix (couleur = GPU Tier, taille = Stockage)",
                                 color_continuous_scale='Viridis',
                                 labels={'ram_gb': 'RAM (GB)', 'price': 'Prix (USD)',
                                         'gpu_tier': 'GPU Tier', 'storage_gb': 'Stockage (GB)'})
        fig_scatter.update_layout(**plotly_dark_layout(height=400))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_trends:
        year_med = df_insight.groupby('release_year')['price'].agg(
            ['median', 'mean', 'min', 'max']).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=year_med['release_year'], y=year_med['max'],
            fill=None, mode='lines', line=dict(color='rgba(79,142,247,0.3)'),
            name='Max'
        ))
        fig_trend.add_trace(go.Scatter(
            x=year_med['release_year'], y=year_med['min'],
            fill='tonexty', mode='lines',
            fillcolor='rgba(79,142,247,0.1)',
            line=dict(color='rgba(79,142,247,0.3)'),
            name='Min'
        ))
        fig_trend.add_trace(go.Scatter(
            x=year_med['release_year'], y=year_med['median'],
            mode='lines+markers', name='Médiane',
            line=dict(color='#4F8EF7', width=3),
            marker=dict(size=10, color='#F59E0B')
        ))
        fig_trend.add_trace(go.Scatter(
            x=year_med['release_year'], y=year_med['mean'],
            mode='lines', name='Moyenne',
            line=dict(color='#10B981', width=2, dash='dash')
        ))
        fig_trend.update_layout(
            title="📅 Évolution des Prix par Année de Sortie",
            xaxis_title="Année", yaxis_title="Prix (USD)",
            **plotly_dark_layout(height=400)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Model Insights":
    st.markdown("<div class='section-title'>🔮 Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Comprenez ce qui influence le plus le prix d'un laptop.</p>",
                unsafe_allow_html=True)

    if metadata:
        metrics = metadata.get('metrics', {})
        feat_imp = metadata.get('feature_importance', [])

        # Métriques modèle
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 R² Score", f"{metrics.get('r2_test', 0):.4f}")
        m2.metric("📏 MAE", format_price_usd(metrics.get('mae_test', 0)))
        m3.metric("📐 RMSE", format_price_usd(metrics.get('rmse_test', 0)))
        m4.metric("📊 MAPE", f"{metrics.get('mape_test', 0):.2f}%")

        st.markdown("---")

        tab_feat, tab_shap, tab_about = st.tabs(["🏆 Feature Importance", "🔮 SHAP Analysis", "📖 À propos"])

        with tab_feat:
            if feat_imp:
                fi_df = pd.DataFrame(feat_imp).sort_values('importance', ascending=True)
                fig = px.bar(fi_df.tail(20), x='importance', y='feature',
                             orientation='h',
                             title="🏆 Top 20 Features les plus Importantes",
                             color='importance', color_continuous_scale='RdYlGn',
                             labels={'importance': 'Importance (%)', 'feature': 'Variable'})
                fig.update_layout(**plotly_dark_layout(height=500), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 Entraînez le modèle pour voir les feature importances.")

            # Afficher images sauvegardées depuis le notebook
            for img_name, title in [
                ("plot_shap_summary.png", "🔮 SHAP Summary Plot"),
                ("plot_shap_importance.png", "📊 SHAP Feature Importance"),
                ("plot_shap_waterfall.png", "💧 SHAP Waterfall - Explication Locale"),
            ]:
                img_path = f"models/{img_name}"
                if os.path.exists(img_path):
                    st.markdown(f"**{title}**")
                    st.image(img_path, use_column_width=True)

        with tab_shap:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color:#4F8EF7;'>🔮 Qu'est-ce que SHAP ?</h4>
                <p style='color:#94A3B8;'>
                SHAP (SHapley Additive exPlanations) est une méthode d'interprétabilité basée
                sur la théorie des jeux coopératifs de Shapley (Nobel d'économie 2012).<br><br>
                Pour chaque prédiction, SHAP calcule la contribution de chaque feature
                au résultat final, permettant de répondre à la question :<br>
                <em>"Pourquoi le modèle a-t-il prédit ce prix ?"</em>
                </p>
                <h4 style='color:#10B981; margin-top:1rem;'>📖 Comment lire le SHAP Summary Plot ?</h4>
                <ul style='color:#94A3B8;'>
                    <li><b style='color:#EF4444;'>Rouge</b> = valeur haute de la feature</li>
                    <li><b style='color:#4F8EF7;'>Bleu</b> = valeur basse de la feature</li>
                    <li>Position sur l'axe X = impact sur le prix (positif = augmente le prix)</li>
                    <li>Ordre des features = importance décroissante</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            for img_name, title, desc in [
                ("plot_model_evaluation.png", "🎯 Prédit vs Réel & Distribution des Résidus",
                 "Le modèle suit bien la diagonale, avec des résidus centrés en 0."),
                ("plot_feature_importance.png", "📊 Feature Importance CatBoost",
                 "Variables classées par importance dans le modèle."),
            ]:
                img_path = f"models/{img_name}"
                if os.path.exists(img_path):
                    col_i, col_d = st.columns([3, 1])
                    with col_i:
                        st.markdown(f"**{title}**")
                        st.image(img_path, use_column_width=True)
                    with col_d:
                        st.markdown(f"""
                        <div class='info-card'>
                            <p style='color:#94A3B8; font-size:0.85rem;'>{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_about:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color:#4F8EF7;'>⚙️ Paramètres du Modèle CatBoost</h4>
                <table style='width:100%; color:#E2E8F0;'>
                    <tr><td style='color:#94A3B8;'>Algorithme</td><td>CatBoostRegressor (Gradient Boosting)</td></tr>
                    <tr><td style='color:#94A3B8;'>Iterations</td><td>1500 (avec early stopping 100 rounds)</td></tr>
                    <tr><td style='color:#94A3B8;'>Learning Rate</td><td>0.05</td></tr>
                    <tr><td style='color:#94A3B8;'>Depth</td><td>8</td></tr>
                    <tr><td style='color:#94A3B8;'>L2 Regularization</td><td>3</td></tr>
                    <tr><td style='color:#94A3B8;'>Loss Function</td><td>RMSE</td></tr>
                    <tr><td style='color:#94A3B8;'>Train/Test Split</td><td>80% / 20%</td></tr>
                </table>
            </div>
            <div class='info-card' style='margin-top:1rem;'>
                <h4 style='color:#10B981;'>⚙️ Feature Engineering</h4>
                <table style='width:100%; color:#E2E8F0;'>
                    <tr><td style='color:#94A3B8;'>cpu_power</td><td>cpu_cores × cpu_boost_ghz</td></tr>
                    <tr><td style='color:#94A3B8;'>gpu_power</td><td>gpu_tier × vram_gb</td></tr>
                    <tr><td style='color:#94A3B8;'>ppi</td><td>√(x²+y²) / display_size_in</td></tr>
                    <tr><td style='color:#94A3B8;'>storage_per_drive</td><td>storage_gb / storage_drive_count</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Modèle non chargé. Veuillez d'abord exécuter le notebook d'entraînement.")
        st.markdown("""
        <div class='info-card'>
            <h4 style='color:#F59E0B;'>📋 Instructions</h4>
            <ol style='color:#94A3B8;'>
                <li>Placez votre dataset dans <code>data/computer_prices_all.csv</code></li>
                <li>Ouvrez et exécutez <code>notebooks/eda_and_training.ipynb</code></li>
                <li>Le modèle sera sauvegardé dans <code>models/laptop_price_model.pkl</code></li>
                <li>Relancez l'application avec <code>streamlit run app.py</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)