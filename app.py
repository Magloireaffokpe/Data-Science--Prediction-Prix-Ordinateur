"""
Laptop Market Price Estimator
Application Streamlit — responsive, épurée, mobile-first
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from catboost import CatBoostRegressor, Pool

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Estimator",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary: #4F8EF7;
    --accent:  #10B981;
    --warn:    #F59E0B;
    --danger:  #EF4444;
    --bg:      #0B1120;
    --surface: #131D30;
    --surface2:#1A2640;
    --border:  rgba(79,142,247,0.18);
    --text:    #E2E8F0;
    --muted:   #7A8FAD;
}

.stApp { background: var(--bg) !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── Navbar ── */
#navbar {
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    position: sticky;
    top: 0;
    z-index: 999;
    flex-wrap: wrap;
}
#navbar .nav-brand {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--primary);
    margin-right: 16px;
    white-space: nowrap;
}
#navbar a {
    padding: 7px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--muted);
    text-decoration: none;
    border: 1px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
}
#navbar a:hover { color: var(--text); background: var(--surface2); }
#navbar a.active {
    color: var(--primary);
    background: rgba(79,142,247,0.12);
    border-color: var(--border);
}

/* Sidebar cachée */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Conteneur */
.main-wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 28px 16px 60px;
}

/* Typographie */
.page-title {
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.page-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin-bottom: 24px;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
}

/* Résultat prix */
.price-result {
    background: var(--surface);
    border: 1px solid rgba(79,142,247,0.4);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    margin-bottom: 20px;
}
.price-result .usd  { font-size: clamp(2.2rem,6vw,3.2rem); font-weight: 800; color: var(--primary); line-height:1; }
.price-result .fcfa { font-size: 1.1rem; color: var(--accent); margin-top: 6px; }
.price-result .seg  {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.8rem !important; }
div[data-testid="stMetricValue"]   { color: var(--text)  !important; font-size: 1.4rem !important; font-weight: 700 !important; }

/* Labels widgets */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] span { color: var(--muted) !important; }
div[data-baseweb="select"] span  { color: var(--text)  !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 2px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    background: transparent;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 14px;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    background: rgba(79,142,247,0.1) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Slider track */
.stSlider > div > div > div { background: var(--primary) !important; }

hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* Bouton */
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--primary);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 12px 24px;
    color: #fff;
    width: 100%;
    transition: opacity 0.15s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.85; }

/* Responsive mobile */
@media (max-width: 640px) {
    .main-wrap { padding: 16px 10px 50px; }
    #navbar     { padding: 8px 10px; }
    #navbar .nav-brand { font-size: 0.82rem; margin-right: 6px; }
    #navbar a   { padding: 5px 9px; font-size: 0.76rem; }
    .price-result { padding: 18px 12px; }
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────────────────────
USD_TO_FCFA = 600
MODEL_PATH  = "models/laptop_price_model.pkl"

# ─── Navigation via query params ──────────────────────────────────────────────
params = st.query_params
page = params.get("page", "estimator")
if page not in ("estimator", "insights", "model"):
    page = "estimator"

def nav_link(label, key):
    cls = "active" if page == key else ""
    return f'<a href="?page={key}" class="{cls}">{label}</a>'

st.markdown(f"""
<div id="navbar">
    <span class="nav-brand">💻 Laptop Price</span>
    {nav_link("Estimateur", "estimator")}
    {nav_link("Marché",     "insights")}
    {nav_link("Modèle",     "model")}
</div>
""", unsafe_allow_html=True)

# ─── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

metadata = load_model()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fmt_usd(p):
    return f"${p:,.0f}"

def fmt_fcfa(p):
    f = p * USD_TO_FCFA
    return f"{f/1_000_000:.2f}M FCFA" if f >= 1_000_000 else f"{f:,.0f} FCFA"

def segment(p):
    if p < 500:  return "Budget",         "#F59E0B"
    if p < 1000: return "Milieu de gamme","#4F8EF7"
    if p < 2000: return "Premium",        "#10B981"
    return             "Ultra-Premium",   "#EF4444"

def perf_score(cores, boost, gpu_tier, vram, ram):
    c = min((cores * boost) / (16 * 5.5), 1)
    g = min((gpu_tier * vram) / (10 * 24), 1)
    r = min(np.log2(max(ram, 1)) / np.log2(128), 1)
    return min(round((0.35*c + 0.40*g + 0.25*r) * 100, 1), 100.0)

def dark_layout(**kw):
    return dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7A8FAD', size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        **kw
    )

def demo_data():
    np.random.seed(42)
    n = 800
    brands = ['Apple','Dell','HP','Lenovo','ASUS','Acer','MSI','Razer','Samsung','LG']
    return pd.DataFrame({
        'brand':           np.random.choice(brands, n),
        'ram_gb':          np.random.choice([8,16,32,64], n, p=[0.25,0.45,0.22,0.08]),
        'storage_gb':      np.random.choice([256,512,1024,2048], n),
        'gpu_tier':        np.random.randint(1, 9, n),
        'cpu_tier':        np.random.randint(1, 6, n),
        'display_size_in': np.random.choice([13.3,14.0,15.6,16.0,17.3], n),
        'release_year':    np.random.choice([2020,2021,2022,2023,2024], n),
        'price':           np.random.lognormal(7.5, 0.68, n).clip(299, 7999),
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : ESTIMATEUR
# ══════════════════════════════════════════════════════════════════════════════
if page == "estimator":
    st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Estimateur de prix</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Configurez les composants pour obtenir une estimation en USD et FCFA.</div>",
                unsafe_allow_html=True)

    brands_default = ['Acer','Apple','ASUS','Dell','HP','Huawei','Lenovo','LG','MSI','Razer','Samsung','Toshiba']
    os_default     = ['Windows 11','Windows 10','macOS','ChromeOS','Linux']

    if metadata:
        brands_opts = metadata.get('brand_options') or brands_default
        os_opts     = metadata.get('os_options')    or os_default
    else:
        brands_opts, os_opts = brands_default, os_default

    tab_dev, tab_cpu, tab_gpu, tab_disp, tab_stor = st.tabs([
        "Appareil", "CPU", "GPU", "Écran", "Stockage"
    ])

    with tab_dev:
        c1, c2 = st.columns(2)
        with c1:
            brand        = st.selectbox("Marque", sorted(brands_opts))
            os_sel       = st.selectbox("Système", os_opts)
            form_factor  = st.selectbox("Catégorie", ['Ultrabook','Notebook','Clamshell','Gaming','2-in-1','Workstation'])
        with c2:
            release_year = st.slider("Année de sortie", 2015, 2025, 2023)
            weight_kg    = st.slider("Poids (kg)", 0.8, 4.5, 1.8, step=0.1)
            warranty     = st.select_slider("Garantie (mois)", [0,12,24,36,48], value=12)

    with tab_cpu:
        c1, c2 = st.columns(2)
        with c1:
            cpu_brand     = st.selectbox("Marque CPU", ['Intel','AMD','Apple','Qualcomm'])
            cpu_tier      = st.slider("Tier (1 bas → 5 haut)", 1, 5, 3)
            cpu_cores     = st.select_slider("Cœurs", [2,4,6,8,10,12,14,16,20,24], value=8)
        with c2:
            cpu_threads   = st.select_slider("Threads", [4,8,12,16,20,24,28,32,40,48], value=16)
            cpu_base_ghz  = st.slider("Fréquence base (GHz)", 0.8, 4.5, 2.4, step=0.1)
            cpu_boost_ghz = st.slider("Fréquence boost (GHz)", 1.5, 6.0, 4.5, step=0.1)

    with tab_gpu:
        c1, c2 = st.columns(2)
        with c1:
            gpu_brand = st.selectbox("Marque GPU", ['NVIDIA','AMD','Intel','Apple'])
            gpu_tier  = st.slider("Tier (1 intégré → 10 haut de gamme)", 1, 10, 4)
        with c2:
            vram_gb   = st.select_slider("VRAM (GB)", [0,2,4,6,8,12,16,24], value=4)
            gpu_label = "Haute performance" if gpu_tier >= 7 else "Milieu de gamme" if gpu_tier >= 4 else "Usage basique"
            st.metric("GPU Power", gpu_tier * vram_gb)
            st.caption(gpu_label)

    with tab_disp:
        c1, c2 = st.columns(2)
        with c1:
            display_type = st.selectbox("Type", ['IPS','OLED','VA','TN','Mini-LED','AMOLED'])
            display_size = st.slider("Taille (pouces)", 11.0, 18.0, 15.6, step=0.1)
        with c2:
            resolution  = st.selectbox("Résolution", ['1366x768','1920x1080','2560x1440','2560x1600','3840x2160'], index=1)
            refresh_hz  = st.select_slider("Rafraîchissement (Hz)", [60,90,120,144,165,240,360], value=60)
        x_res, y_res = map(int, resolution.split('x'))
        ppi = np.sqrt(x_res**2 + y_res**2) / display_size
        st.caption(f"Densité : {ppi:.0f} PPI")

    with tab_stor:
        c1, c2 = st.columns(2)
        with c1:
            ram_gb         = st.select_slider("RAM (GB)", [4,8,16,32,64,128], value=16)
            storage_type   = st.selectbox("Type stockage", ['NVMe SSD','SSD','HDD','eMMC','Hybrid'])
        with c2:
            storage_gb     = st.select_slider("Capacité (GB)", [128,256,512,1024,2048,4096], value=512)
            storage_drives = st.slider("Nombre de disques", 1, 4, 1)
            battery_wh     = st.slider("Batterie (Wh)", 20, 150, 60, step=5)

    st.markdown("---")

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run = st.button("Estimer le prix", type="primary", use_container_width=True)

    # ── Résultat ──────────────────────────────────────────────────────────────
    if run:
        cpu_power_val     = cpu_cores * cpu_boost_ghz
        gpu_power_val     = gpu_tier * vram_gb
        storage_per_drive = storage_gb / max(storage_drives, 1)

        if metadata:
            model_obj = metadata['model']
            feat_cols = metadata['feature_columns']
            cat_feats = metadata['cat_features']
            row = {
                'brand': brand, 'release_year': release_year, 'os': os_sel,
                'form_factor': form_factor, 'cpu_brand': cpu_brand,
                'cpu_tier': cpu_tier, 'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads, 'cpu_base_ghz': cpu_base_ghz,
                'cpu_boost_ghz': cpu_boost_ghz, 'gpu_brand': gpu_brand,
                'gpu_tier': gpu_tier, 'vram_gb': vram_gb, 'ram_gb': ram_gb,
                'storage_type': storage_type, 'storage_gb': storage_gb,
                'storage_drive_count': storage_drives, 'display_type': display_type,
                'display_size_in': display_size, 'refresh_hz': refresh_hz,
                'x_resolution': x_res, 'y_resolution': y_res,
                'battery_wh': battery_wh, 'charger_watts': 65,
                'wifi': 'WiFi 6', 'bluetooth': '5.0',
                'weight_kg': weight_kg, 'warranty_months': warranty,
                'cpu_power': cpu_power_val, 'gpu_power': gpu_power_val,
                'ppi': ppi, 'storage_per_drive': storage_per_drive,
            }
            df_in = pd.DataFrame([row])
            for c in feat_cols:
                if c not in df_in: df_in[c] = 0
            for c in cat_feats:
                if c in df_in: df_in[c] = df_in[c].astype(str)
            df_in   = df_in[feat_cols]
            cat_idx = [df_in.columns.get_loc(c) for c in cat_feats if c in df_in.columns]
            price   = float(model_obj.predict(Pool(df_in, cat_features=cat_idx))[0])
            all_px  = metadata['price_stats']['all_prices']
        else:
            price  = (300 + ram_gb*12 + storage_gb*0.25 + gpu_tier*75
                      + cpu_tier*55 + (release_year-2015)*15)
            price *= (1.3 if brand == 'Apple' else 1.0)
            price *= 0.85 + np.random.random() * 0.3
            all_px = list(np.random.lognormal(7.4, 0.7, 800).clip(199, 7999))

        price = round(max(199, min(price, 12000)), 2)
        seg_label, seg_color = segment(price)
        score = perf_score(cpu_cores, cpu_boost_ghz, gpu_tier, vram_gb, ram_gb)
        pct   = sum(1 for p in all_px if p < price) / len(all_px) * 100

        st.markdown(f"""
        <div class='price-result'>
            <div style='font-size:0.78rem;color:var(--muted);margin-bottom:6px;letter-spacing:.08em;'>PRIX ESTIMÉ</div>
            <div class='usd'>{fmt_usd(price)}</div>
            <div class='fcfa'>{fmt_fcfa(price)}</div>
            <div class='seg' style='color:{seg_color};border-color:{seg_color};background:{seg_color}18;'>{seg_label}</div>
        </div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Percentile marché",  f"Top {100-pct:.0f}%")
        mc2.metric("Score performance",  f"{score} / 100")
        mc3.metric("Équivalent FCFA",    fmt_fcfa(price))

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=all_px, nbinsx=60,
                                       marker_color='#4F8EF7', opacity=0.6, showlegend=False))
            fig.add_vline(x=price, line_width=2, line_color='#EF4444', line_dash='dash',
                          annotation_text=fmt_usd(price),
                          annotation_font_color='#EF4444', annotation_font_size=11)
            fig.update_layout(title="Position dans le marché",
                              xaxis_title="Prix (USD)", yaxis_title="",
                              **dark_layout(height=280))
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            cats = ['CPU','GPU','RAM','Stockage','Écran','Batterie']
            vals = [
                min(cpu_power_val / 88,   1) * 100,
                min(gpu_power_val / 80,   1) * 100,
                min(ram_gb / 128,         1) * 100,
                min(storage_gb / 4096,    1) * 100,
                min(ppi / 400,            1) * 100,
                min(battery_wh / 150,     1) * 100,
            ]
            fig2 = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(79,142,247,0.18)',
                line=dict(color='#4F8EF7', width=2),
            ))
            fig2.update_layout(
                title="Profil de performance",
                polar=dict(radialaxis=dict(
                    visible=True, range=[0,100],
                    gridcolor='rgba(79,142,247,0.15)',
                    tickfont=dict(color='#7A8FAD', size=9))),
                showlegend=False,
                **dark_layout(height=280)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Récap compact
        st.markdown(f"""
        <div class='card' style='font-size:0.88rem; color:#CBD5E1;'>
            <div class='card-title'>Récapitulatif configuration</div>
            <div style='display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px 24px;'>
                <div><span style='color:var(--muted);font-size:0.78rem;'>Marque</span><br><b>{brand}</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>CPU</span><br><b>{cpu_brand} T{cpu_tier} · {cpu_cores}c @ {cpu_boost_ghz}GHz</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>GPU</span><br><b>{gpu_brand} T{gpu_tier} · {vram_gb}GB VRAM</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>RAM</span><br><b>{ram_gb} GB</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>Stockage</span><br><b>{storage_gb} GB {storage_type}</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>Écran</span><br><b>{display_size}" · {resolution} · {refresh_hz}Hz</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>Batterie</span><br><b>{battery_wh} Wh</b></div>
                <div><span style='color:var(--muted);font-size:0.78rem;'>Année</span><br><b>{release_year}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MARCHÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "insights":
    st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Tendances du marché</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Analyse des prix sur la base des données d'entraînement.</div>",
                unsafe_allow_html=True)

    df = demo_data()
    if metadata and 'price_stats' in metadata:
        ps = metadata['price_stats']
        mean_p, med_p = ps['mean'], ps['median']
        min_p,  max_p = ps['min'],  ps['max']
    else:
        mean_p, med_p = df['price'].mean(), df['price'].median()
        min_p,  max_p = df['price'].min(),  df['price'].max()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prix moyen",  fmt_usd(mean_p))
    k2.metric("Prix médian", fmt_usd(med_p))
    k3.metric("Prix min",    fmt_usd(min_p))
    k4.metric("Prix max",    fmt_usd(max_p))

    st.markdown("---")

    tab_brand, tab_specs, tab_yr = st.tabs(["Par marque", "Par composants", "Par année"])

    with tab_brand:
        bp = df.groupby('brand')['price'].median().sort_values(ascending=True).reset_index()
        fig = px.bar(bp, x='price', y='brand', orientation='h',
                     color='price', color_continuous_scale='Blues',
                     labels={'price': 'Prix médian (USD)', 'brand': 'Marque'})
        fig.update_layout(showlegend=False, **dark_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    with tab_specs:
        c1, c2 = st.columns(2)
        with c1:
            rm  = df.groupby('ram_gb')['price'].median().reset_index()
            fig = px.bar(rm, x='ram_gb', y='price', color='price',
                         color_continuous_scale='Blues',
                         labels={'ram_gb': 'RAM (GB)', 'price': 'Prix médian (USD)'})
            fig.update_layout(showlegend=False, title="RAM vs Prix", **dark_layout(height=300))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gm  = df.groupby('gpu_tier')['price'].median().reset_index()
            fig = px.line(gm, x='gpu_tier', y='price', markers=True,
                          labels={'gpu_tier': 'GPU Tier', 'price': 'Prix médian (USD)'})
            fig.update_traces(line=dict(color='#4F8EF7', width=2.5),
                              marker=dict(size=8, color='#F59E0B'))
            fig.update_layout(title="GPU Tier vs Prix", **dark_layout(height=300))
            st.plotly_chart(fig, use_container_width=True)

    with tab_yr:
        ym  = df.groupby('release_year')['price'].agg(['median','min','max']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ym['release_year'], y=ym['max'], mode='lines',
                                 line=dict(color='rgba(79,142,247,0.2)'), showlegend=False))
        fig.add_trace(go.Scatter(x=ym['release_year'], y=ym['min'], mode='lines',
                                 fill='tonexty', fillcolor='rgba(79,142,247,0.07)',
                                 line=dict(color='rgba(79,142,247,0.2)'), showlegend=False))
        fig.add_trace(go.Scatter(x=ym['release_year'], y=ym['median'],
                                 mode='lines+markers', name='Médiane',
                                 line=dict(color='#4F8EF7', width=2.5),
                                 marker=dict(size=8, color='#F59E0B')))
        fig.update_layout(title="Évolution des prix par année",
                          xaxis_title="Année", yaxis_title="Prix (USD)",
                          **dark_layout(height=320))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "model":
    st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Analyse du modèle</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Performance et interprétabilité du modèle CatBoost.</div>",
                unsafe_allow_html=True)

    if not metadata:
        st.info("Entraînez d'abord le modèle via le notebook `eda_and_training.ipynb`.")
        st.markdown("""
        <div class='card' style='font-size:0.88rem;color:#CBD5E1;'>
            <div class='card-title'>Instructions</div>
            <ol style='color:#7A8FAD; padding-left:1.2rem; line-height:2;'>
                <li>Placez le dataset dans <code>data/computer_prices_all.csv</code></li>
                <li>Exécutez le notebook (Kernel → Run All)</li>
                <li>Relancez : <code>streamlit run app.py</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        metrics  = metadata.get('metrics', {})
        feat_imp = metadata.get('feature_importance', [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²",   f"{metrics.get('r2_test', 0):.3f}")
        m2.metric("MAE",  fmt_usd(metrics.get('mae_test', 0)))
        m3.metric("RMSE", fmt_usd(metrics.get('rmse_test', 0)))
        m4.metric("MAPE", f"{metrics.get('mape_test', 0):.1f}%")

        st.markdown("---")

        tab_fi, tab_shap, tab_params = st.tabs(["Feature importance", "SHAP", "Paramètres"])

        with tab_fi:
            if feat_imp:
                fi  = pd.DataFrame(feat_imp).sort_values('importance', ascending=True)
                fig = px.bar(fi.tail(15), x='importance', y='feature', orientation='h',
                             color='importance', color_continuous_scale='Blues',
                             labels={'importance': 'Importance (%)', 'feature': 'Variable'})
                fig.update_layout(showlegend=False, **dark_layout(height=420))
                st.plotly_chart(fig, use_container_width=True)

            for fname, label in [
                ("plot_shap_summary.png",   "SHAP Summary"),
                ("plot_shap_waterfall.png", "SHAP Waterfall — explication locale"),
            ]:
                p = f"models/{fname}"
                if os.path.exists(p):
                    st.markdown(f"**{label}**")
                    st.image(p, use_column_width=True)

        with tab_shap:
            st.markdown("""
            <div class='card' style='color:#CBD5E1; font-size:0.9rem; line-height:1.8;'>
                <div class='card-title'>Qu'est-ce que SHAP ?</div>
                SHAP quantifie la contribution de chaque variable à chaque prédiction,
                en s'appuyant sur la théorie des jeux de Shapley.<br><br>
                <b style='color:#EF4444;'>Rouge</b> = valeur haute &nbsp;·&nbsp;
                <b style='color:#4F8EF7;'>Bleu</b> = valeur basse &nbsp;·&nbsp;
                Position X = impact sur le prix prédit
            </div>
            """, unsafe_allow_html=True)

            for fname, label in [
                ("plot_model_evaluation.png",   "Prédit vs Réel"),
                ("plot_feature_importance.png", "Feature Importance CatBoost"),
            ]:
                p = f"models/{fname}"
                if os.path.exists(p):
                    st.markdown(f"**{label}**")
                    st.image(p, use_column_width=True)

        with tab_params:
            st.markdown("""
            <div class='card' style='font-size:0.88rem;'>
                <div class='card-title'>Hyperparamètres CatBoost</div>
                <table style='width:100%; border-collapse:collapse; color:#CBD5E1;'>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>Iterations</td><td>1500 (early stopping 100)</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>Learning rate</td><td>0.05</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>Depth</td><td>8</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>L2 regularization</td><td>3</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>Loss function</td><td>RMSE</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;'>Train / Test</td><td>80 % / 20 %</td></tr>
                </table>
            </div>
            <div class='card' style='font-size:0.88rem;'>
                <div class='card-title'>Features engineered</div>
                <table style='width:100%; border-collapse:collapse; color:#CBD5E1;'>
                    <tr><td style='color:#7A8FAD;padding:5px 0;font-family:monospace;'>cpu_power</td><td>cpu_cores × cpu_boost_ghz</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;font-family:monospace;'>gpu_power</td><td>gpu_tier × vram_gb</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;font-family:monospace;'>ppi</td><td>√(x²+y²) / display_size</td></tr>
                    <tr><td style='color:#7A8FAD;padding:5px 0;font-family:monospace;'>storage_per_drive</td><td>storage_gb / nb_disques</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)