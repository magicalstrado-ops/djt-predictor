
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import requests
import pandas.tseries.offsets as offsets
import warnings
import datetime
import os
import csv
warnings.filterwarnings('ignore')

# ── HISTORIAL ─────────────────────────────────────────────────────────────────
HISTORIAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historial_predicciones.csv")
HISTORIAL_COLS = ["fecha","ticker","mercado","precio_actual","precio_objetivo",
                  "variacion_pct","rango_min","rango_max","mae_usd","senal",
                  "precio_real_siguiente","error_real_usd","acierto"]

def cargar_historial():
    if not os.path.exists(HISTORIAL_CSV):
        return pd.DataFrame(columns=HISTORIAL_COLS)
    try:
        df_h = pd.read_csv(HISTORIAL_CSV)
        for col in HISTORIAL_COLS:
            if col not in df_h.columns:
                df_h[col] = None
        return df_h[HISTORIAL_COLS]
    except:
        return pd.DataFrame(columns=HISTORIAL_COLS)

def guardar_prediccion(fila_dict):
    existe = os.path.exists(HISTORIAL_CSV)
    with open(HISTORIAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORIAL_COLS)
        if not existe:
            writer.writeheader()
        writer.writerow({col: fila_dict.get(col, "") for col in HISTORIAL_COLS})

def actualizar_resultados_reales(df_hist):
    if df_hist.empty:
        return df_hist
    actualizados = False
    for i, row in df_hist.iterrows():
        if pd.isna(row["precio_real_siguiente"]) or str(row["precio_real_siguiente"]) == "":
            try:
                fecha_pred = pd.to_datetime(row["fecha"]).date()
                hoy = datetime.date.today()
                if (hoy - fecha_pred).days >= 1:
                    data_real = yf.download(str(row["ticker"]),
                        start=(fecha_pred + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                        end=(fecha_pred + datetime.timedelta(days=6)).strftime("%Y-%m-%d"),
                        progress=False)
                    if len(data_real) > 0:
                        close_col = data_real['Close']
                        precio_real = float(close_col.iloc[0].iloc[0] if hasattr(close_col.iloc[0], 'iloc') else close_col.iloc[0])
                        precio_obj  = float(row["precio_objetivo"])
                        precio_act  = float(row["precio_actual"])
                        error_real  = abs(precio_real - precio_obj)
                        dir_pred    = "sube" if float(row["variacion_pct"]) > 0 else "baja"
                        dir_real    = "sube" if precio_real > precio_act else "baja"
                        acierto     = "✅" if dir_pred == dir_real else "❌"
                        df_hist.at[i, "precio_real_siguiente"] = round(precio_real, 4)
                        df_hist.at[i, "error_real_usd"]        = round(error_real, 4)
                        df_hist.at[i, "acierto"]               = acierto
                        actualizados = True
            except:
                pass
    if actualizados:
        df_hist.to_csv(HISTORIAL_CSV, index=False)
    return df_hist

# ── BASE DE DATOS DE TICKERS ──────────────────────────────────────────────────
TICKERS_ARG = {
    "YPF":   ("YPF S.A.",                  "Energía · Argentina"),
    "PAM":   ("Pampa Energía",             "Energía · Argentina"),
    "CEPU":  ("Central Puerto",            "Utilities · Argentina"),
    "TGS":   ("Transportadora Gas Sur",    "Energía · Argentina"),
    "VIST":  ("Vista Energy",              "Energía · Argentina"),
    "EDN":   ("Edenor",                    "Utilities · Argentina"),
    "PAMP":  ("Pampa Energía (BA)",        "Energía · Argentina"),
    "GGAL":  ("Grupo Galicia",             "Banco · Argentina"),
    "BMA":   ("Banco Macro",               "Banco · Argentina"),
    "SUPV":  ("Supervielle",               "Banco · Argentina"),
    "BBAR":  ("BBVA Argentina",            "Banco · Argentina"),
    "VALO":  ("Grupo Supervielle",         "Banco · Argentina"),
    "BPAT":  ("Banco Patagonia",           "Banco · Argentina"),
    "MELI":  ("MercadoLibre",              "Tecnología · Argentina"),
    "GLOB":  ("Globant",                   "Tecnología · Argentina"),
    "DESP":  ("Despegar.com",              "Turismo · Argentina"),
    "IRSA":  ("IRSA Inversiones",          "Real Estate · Argentina"),
    "IRS":   ("IRSA Propiedades",          "Real Estate · Argentina"),
    "LOMA":  ("Loma Negra",                "Materiales · Argentina"),
    "ARCO":  ("Arcos Dorados",             "Consumo · Argentina"),
    "TXAR":  ("Ternium Argentina",         "Acero · Argentina"),
    "TX":    ("Ternium S.A.",              "Acero · Argentina"),
    "TECO2": ("Telecom Argentina",         "Telecom · Argentina"),
}

TICKERS_USA = {
    "AAPL":  ("Apple Inc.",                "Tecnología · EEUU"),
    "MSFT":  ("Microsoft",                 "Tecnología · EEUU"),
    "GOOGL": ("Alphabet / Google",         "Tecnología · EEUU"),
    "GOOG":  ("Alphabet Class C",          "Tecnología · EEUU"),
    "AMZN":  ("Amazon",                    "Tecnología · EEUU"),
    "NVDA":  ("NVIDIA",                    "Semiconductores · EEUU"),
    "META":  ("Meta Platforms",            "Redes Sociales · EEUU"),
    "TSLA":  ("Tesla",                     "Autos / IA · EEUU"),
    "AVGO":  ("Broadcom",                  "Semiconductores · EEUU"),
    "ORCL":  ("Oracle",                    "Software · EEUU"),
    "CRM":   ("Salesforce",                "Software · EEUU"),
    "NOW":   ("ServiceNow",                "Software · EEUU"),
    "ADBE":  ("Adobe",                     "Software · EEUU"),
    "INTU":  ("Intuit",                    "Fintech · EEUU"),
    "AMD":   ("AMD",                       "Semiconductores · EEUU"),
    "INTC":  ("Intel",                     "Semiconductores · EEUU"),
    "QCOM":  ("Qualcomm",                  "Semiconductores · EEUU"),
    "NFLX":  ("Netflix",                   "Streaming · EEUU"),
    "UBER":  ("Uber Technologies",         "Transporte · EEUU"),
    "ABNB":  ("Airbnb",                    "Turismo · EEUU"),
    "BKNG":  ("Booking Holdings",          "Turismo · EEUU"),
    "SHOP":  ("Shopify",                   "eCommerce · EEUU"),
    "CRWD":  ("CrowdStrike",               "Ciberseguridad · EEUU"),
    "NET":   ("Cloudflare",                "Redes · EEUU"),
    "DDOG":  ("Datadog",                   "Monitoreo · EEUU"),
    "SNOW":  ("Snowflake",                 "Data Cloud · EEUU"),
    "PLTR":  ("Palantir",                  "IA · EEUU"),
    "ARM":   ("Arm Holdings",              "Semiconductores · EEUU"),
    "JPM":   ("JPMorgan Chase",            "Banco · EEUU"),
    "BAC":   ("Bank of America",           "Banco · EEUU"),
    "GS":    ("Goldman Sachs",             "Banco · EEUU"),
    "MS":    ("Morgan Stanley",            "Banco · EEUU"),
    "V":     ("Visa",                      "Pagos · EEUU"),
    "MA":    ("Mastercard",                "Pagos · EEUU"),
    "PYPL":  ("PayPal",                    "Fintech · EEUU"),
    "COIN":  ("Coinbase",                  "Crypto · EEUU"),
    "XOM":   ("ExxonMobil",                "Energía · EEUU"),
    "CVX":   ("Chevron",                   "Energía · EEUU"),
    "COP":   ("ConocoPhillips",            "Energía · EEUU"),
    "OXY":   ("Occidental Petroleum",      "Energía · EEUU"),
    "LLY":   ("Eli Lilly",                 "Farmacéutico · EEUU"),
    "JNJ":   ("Johnson & Johnson",         "Salud · EEUU"),
    "PFE":   ("Pfizer",                    "Farmacéutico · EEUU"),
    "WMT":   ("Walmart",                   "Retail · EEUU"),
    "COST":  ("Costco",                    "Retail · EEUU"),
    "HD":    ("Home Depot",                "Retail · EEUU"),
    "NKE":   ("Nike",                      "Consumo · EEUU"),
    "KO":    ("Coca-Cola",                 "Consumo · EEUU"),
    "PEP":   ("PepsiCo",                   "Consumo · EEUU"),
    "MCD":   ("McDonald's",                "Restaurantes · EEUU"),
    "DIS":   ("Walt Disney",               "Entretenimiento · EEUU"),
    "BA":    ("Boeing",                    "Aeroespacial · EEUU"),
    "CAT":   ("Caterpillar",               "Maquinaria · EEUU"),
    "F":     ("Ford Motor",                "Autos · EEUU"),
    "GM":    ("General Motors",            "Autos · EEUU"),
    "RIVN":  ("Rivian",                    "EV · EEUU"),
    "SPY":   ("S&P 500 ETF (SPDR)",        "Índice · EEUU"),
    "QQQ":   ("Nasdaq 100 ETF",            "Índice · EEUU"),
    "GLD":   ("SPDR Gold Shares",          "ETF Oro · Commodities"),
    "TLT":   ("iShares 20Y Treasury",      "ETF Bonos · EEUU"),
    "ARKK":  ("ARK Innovation ETF",        "ETF Innovación · EEUU"),
    "TQQQ":  ("ProShares Ultra QQQ 3x",    "ETF Apalancado · EEUU"),
    "IBIT":  ("iShares Bitcoin Trust",     "ETF Bitcoin · Crypto"),
    "BTC-USD":  ("Bitcoin",               "Crypto"),
    "ETH-USD":  ("Ethereum",              "Crypto"),
    "SOL-USD":  ("Solana",                "Crypto"),
    "XRP-USD":  ("Ripple XRP",            "Crypto"),
    "DOGE-USD": ("Dogecoin",              "Crypto"),
    "TSM":   ("Taiwan Semiconductor",     "Semiconductores · Taiwan"),
    "ASML":  ("ASML Holding",             "Semiconductores · Países Bajos"),
    "BABA":  ("Alibaba Group",            "Tecnología · China"),
    "VALE":  ("Vale S.A.",                "Minería · Brasil"),
    "PBR":   ("Petrobras",                "Energía · Brasil"),
    "NU":    ("Nu Holdings",              "Fintech · Brasil"),
}

TODOS_TICKERS = {**TICKERS_ARG, **TICKERS_USA}
SET_ARG = set(TICKERS_ARG.keys())

def es_argentino(ticker):
    return ticker.upper() in SET_ARG

def buscar_tickers(query):
    if not query or len(query) < 1:
        return []
    q = query.upper()
    resultados = []
    for sym, (nombre, sector) in TODOS_TICKERS.items():
        if sym.startswith(q) or q in nombre.upper():
            resultados.append((sym, nombre, sector))
    return sorted(resultados, key=lambda x: (not x[0].startswith(q), x[0]))[:8]

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DJT Capital · Investigación Cuantitativa",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --navy:       #060d1a;
    --navy2:      #0a1423;
    --navy3:      #0f1c30;
    --navy4:      #142038;
    --border:     #162035;
    --border2:    #1e3050;
    --azul:       #1a56db;
    --azul2:      #1e40af;
    --acento:     #3b82f6;
    --acento2:    #60a5fa;
    --texto:      #dde6f5;
    --texto-dim:  #7b93b8;
    --texto-xs:   #3d5a80;
    --verde:      #05d890;
    --verde-dim:  #064e3b;
    --rojo:       #f43f5e;
    --rojo-dim:   #4c0519;
    --dorado:     #f59e0b;
    --blanco:     #f0f6ff;
    --naranja:    #fb923c;
}

html, body { scroll-behavior: smooth; }
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0c1a35 0%, var(--navy) 60%) !important;
    color: var(--texto);
    font-family: 'JetBrains Mono', monospace;
}
.main .block-container { padding: 0 2.2rem 3rem 2.2rem; max-width: 1700px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy2) 0%, var(--navy) 100%) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] * { color: var(--texto-dim) !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: var(--navy3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--blanco) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    border-radius: 3px !important;
    letter-spacing: 0.08em !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--acento) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.18) !important;
}

@keyframes pulso         { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }
@keyframes parpadeo      { 0%,100%{opacity:1} 50%{opacity:0.2} }
@keyframes entradaAbajo  { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }
@keyframes entradaIzq    { from{opacity:0;transform:translateX(-18px)} to{opacity:1;transform:translateX(0)} }
@keyframes contadorUp    { from{opacity:0;transform:translateY(10px) scale(0.95)} to{opacity:1;transform:translateY(0) scale(1)} }
@keyframes bordeGlow     { 0%,100%{box-shadow:0 0 0 rgba(5,216,144,0)} 50%{box-shadow:0 0 22px rgba(5,216,144,0.12)} }
@keyframes slideRight    { from{width:0} to{width:100%} }
@keyframes fadeIn        { from{opacity:0} to{opacity:1} }
@keyframes shimmer       { 0%{background-position:-200% center} 100%{background-position:200% center} }

.header-bar {
    background: linear-gradient(90deg, var(--navy2) 0%, var(--navy3) 100%);
    border-bottom: 1px solid var(--border2);
    padding: 16px 2.2rem 13px 2.2rem;
    margin: 0 -2.2rem 2.2rem -2.2rem;
    display: flex; align-items: center; justify-content: space-between;
    animation: entradaAbajo 0.5s ease both;
    position: relative; overflow: hidden;
}
.header-bar::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, var(--acento), transparent);
    animation: slideRight 1.2s ease 0.3s both;
}
.header-firma {
    font-family:'Playfair Display',serif; font-size:1.1rem;
    color:var(--blanco); letter-spacing:0.1em; text-transform:uppercase; font-weight:600;
}
.header-division {
    font-family:'JetBrains Mono',monospace; font-size:0.58rem;
    color:var(--texto-xs); letter-spacing:0.22em; text-transform:uppercase; margin-top:4px;
}
.header-derecha { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:var(--texto-xs); letter-spacing:0.1em; text-align:right; }
.punto-vivo {
    display:inline-block; width:7px; height:7px;
    background:var(--verde); border-radius:50%; margin-right:7px;
    animation:pulso 1.8s ease-in-out infinite; box-shadow:0 0 8px var(--verde);
}

.etiqueta-seccion {
    font-family:'JetBrains Mono',monospace; font-size:0.56rem;
    color:var(--texto-xs); letter-spacing:0.3em; text-transform:uppercase;
    border-left:2px solid var(--acento); padding-left:10px;
    margin-bottom:16px; margin-top:4px; animation:entradaIzq 0.4s ease both;
}

.badge-arg {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:0.54rem; letter-spacing:0.2em; text-transform:uppercase;
    background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
    color:var(--dorado); padding:3px 10px; border-radius:2px;
    animation:fadeIn 0.4s ease both;
}
.badge-usa {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:0.54rem; letter-spacing:0.2em; text-transform:uppercase;
    background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3);
    color:var(--acento2); padding:3px 10px; border-radius:2px;
    animation:fadeIn 0.4s ease both;
}

.ac-contenedor {
    background: var(--navy3);
    border: 1px solid var(--border2);
    border-top: none;
    border-radius: 0 0 4px 4px;
    overflow: hidden;
    animation: entradaAbajo 0.2s ease both;
    margin-top: -4px;
}
.ac-fila {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.15s ease;
    font-family: 'JetBrains Mono', monospace;
}
.ac-fila:last-child { border-bottom: none; }
.ac-fila:hover { background: var(--navy4); }
.ac-ticker {
    font-size: 0.8rem; font-weight: 600; color: var(--acento2);
    letter-spacing: 0.08em; min-width: 58px;
}
.ac-nombre { font-size: 0.68rem; color: var(--texto-dim); flex:1; padding: 0 10px; }
.ac-sector { font-size: 0.56rem; color: var(--texto-xs); letter-spacing: 0.06em; }
.ac-arg-dot {
    display:inline-block; width:5px; height:5px; border-radius:50%;
    background:var(--dorado); margin-right:6px; vertical-align:middle;
}
.ac-usa-dot {
    display:inline-block; width:5px; height:5px; border-radius:50%;
    background:var(--acento); margin-right:6px; vertical-align:middle;
}
.ac-header {
    font-family:'JetBrains Mono',monospace; font-size:0.52rem;
    color:var(--texto-xs); letter-spacing:0.2em; text-transform:uppercase;
    padding:7px 14px 5px 14px; background:var(--navy2);
    border-bottom:1px solid var(--border);
}
.ac-vacio {
    font-family:'JetBrains Mono',monospace; font-size:0.64rem;
    color:var(--texto-xs); padding:12px 14px; text-align:center;
    letter-spacing:0.06em;
}

.tarjeta-compra, .tarjeta-venta {
    padding:24px 30px; border-radius:3px; position:relative; overflow:hidden;
    animation:entradaAbajo 0.5s ease 0.2s both;
    transition:transform 0.3s ease, box-shadow 0.3s ease;
}
.tarjeta-compra:hover, .tarjeta-venta:hover { transform:translateY(-2px); box-shadow:0 12px 40px rgba(0,0,0,0.4); }
.tarjeta-compra {
    background:linear-gradient(135deg,#021a0e 0%,#032d17 60%,#042010 100%);
    border:1px solid #0d5c2e; border-left:3px solid var(--verde);
    animation:bordeGlow 3s infinite;
}
.tarjeta-venta {
    background:linear-gradient(135deg,#1a020a 0%,#2d0310 60%,#200310 100%);
    border:1px solid #6b1030; border-left:3px solid var(--rojo);
}
.tipo-senal { font-family:'JetBrains Mono',monospace; font-size:0.56rem; letter-spacing:0.3em; text-transform:uppercase; color:var(--texto-xs); margin-bottom:8px; }
.valor-senal-compra { font-family:'Playfair Display',serif; font-size:2.1rem; color:var(--verde); font-weight:700; letter-spacing:0.03em; animation:contadorUp 0.6s ease 0.4s both; text-shadow:0 0 20px rgba(5,216,144,0.3); }
.valor-senal-venta  { font-family:'Playfair Display',serif; font-size:2.1rem; color:var(--rojo);  font-weight:700; letter-spacing:0.03em; animation:contadorUp 0.6s ease 0.4s both; text-shadow:0 0 20px rgba(244,63,94,0.3); }
.sub-senal { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--texto-dim); margin-top:10px; letter-spacing:0.05em; animation:entradaAbajo 0.4s ease 0.6s both; }

.celda-dato {
    background:var(--navy2); border:1px solid var(--border); border-radius:3px;
    padding:15px 18px; height:100%; transition:all 0.3s ease;
    animation:entradaAbajo 0.5s ease both; position:relative; overflow:hidden;
}
.celda-dato:hover { border-color:var(--border2); background:var(--navy3); transform:translateY(-1px); box-shadow:0 6px 20px rgba(0,0,0,0.3); }
.celda-dato::after { content:''; position:absolute; top:0; left:0; width:100%; height:2px; background:linear-gradient(90deg,transparent,var(--acento),transparent); opacity:0; transition:opacity 0.3s; }
.celda-dato:hover::after { opacity:1; }
.etiqueta-celda { font-family:'JetBrains Mono',monospace; font-size:0.54rem; color:var(--texto-xs); letter-spacing:0.25em; text-transform:uppercase; margin-bottom:8px; }
.valor-celda-grande { font-family:'Playfair Display',serif; font-size:1.6rem; color:var(--blanco); font-weight:500; line-height:1; animation:contadorUp 0.5s ease both; }
.valor-celda-med    { font-family:'JetBrains Mono',monospace; font-size:0.98rem; color:var(--blanco); animation:contadorUp 0.5s ease both; }
.sub-celda { font-family:'JetBrains Mono',monospace; font-size:0.56rem; color:var(--texto-xs); margin-top:6px; }
.verde  { color:var(--verde)   !important; }
.rojo   { color:var(--rojo)    !important; }
.azul   { color:var(--acento2) !important; }
.dorado { color:var(--dorado)  !important; }

.separador { border:none; height:1px; background:linear-gradient(90deg,transparent 0%,var(--border2) 30%,var(--border2) 70%,transparent 100%); margin:24px 0; }

.tabla-modelos { background:var(--navy2); border:1px solid var(--border); border-radius:3px; overflow:hidden; animation:entradaAbajo 0.5s ease 0.3s both; }
.fila-modelo { display:flex; justify-content:space-between; align-items:center; padding:11px 20px; border-bottom:1px solid var(--border); font-family:'JetBrains Mono',monospace; font-size:0.76rem; transition:background 0.2s ease; }
.fila-modelo:hover { background:var(--navy3); }
.fila-modelo:last-child { border-bottom:none; }
.fila-encabezado { background:var(--navy3); font-size:0.54rem; letter-spacing:0.25em; text-transform:uppercase; color:var(--texto-xs); }
.nombre-modelo { color:var(--texto-dim); }
.mae-modelo   { color:var(--acento2); font-weight:500; }
.mae-ensamble { color:var(--verde); font-weight:600; }
.barra-peso   { display:inline-block; height:3px; background:var(--acento); border-radius:2px; margin-top:4px; transition:width 0.8s ease; }
.badge-activo { display:inline-flex; align-items:center; gap:4px; font-size:0.58rem; letter-spacing:0.1em; color:var(--verde); }
.badge-activo::before { content:'◉'; animation:parpadeo 2s ease-in-out infinite; }
.fila-ensamble { background:linear-gradient(90deg,rgba(5,216,144,0.04) 0%,transparent 100%); border-top:1px solid rgba(5,216,144,0.15) !important; }

.estado-inactivo { background:var(--navy2); border:1px solid var(--border); border-radius:3px; padding:80px 40px; text-align:center; margin:24px 0; animation:entradaAbajo 0.6s ease both; position:relative; overflow:hidden; }
.estado-inactivo::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--acento),transparent); }
.titulo-inactivo { font-family:'Playfair Display',serif; font-size:1.2rem; color:var(--texto-dim); letter-spacing:0.08em; margin-bottom:12px; }
.sub-inactivo { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--texto-xs); letter-spacing:0.12em; line-height:2; }
.icono-inactivo { font-size:2.2rem; color:var(--border2); margin-bottom:18px; display:block; animation:pulso 3s ease-in-out infinite; }

.stButton > button {
    background:linear-gradient(135deg,var(--azul) 0%,var(--azul2) 100%) !important;
    color:#fff !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.7rem !important; font-weight:600 !important; letter-spacing:0.2em !important;
    text-transform:uppercase !important; border:none !important; border-radius:3px !important;
    padding:13px 20px !important; width:100% !important; transition:all 0.3s ease !important;
}
.stButton > button:hover { background:linear-gradient(135deg,#1e64f0 0%,#1e3fc8 100%) !important; box-shadow:0 6px 25px rgba(26,86,219,0.45) !important; transform:translateY(-1px) !important; }

.aviso-legal { font-family:'JetBrains Mono',monospace; font-size:0.53rem; color:var(--texto-xs); letter-spacing:0.08em; text-align:center; padding:18px 0 10px 0; border-top:1px solid var(--border); margin-top:28px; line-height:2; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--navy); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--acento); }

.hist-tabla { background:var(--navy2); border:1px solid var(--border); border-radius:3px; overflow:hidden; animation:entradaAbajo 0.5s ease both; }
.hist-fila { display:grid; grid-template-columns:90px 65px 60px 85px 85px 75px 90px 85px 55px; align-items:center; padding:9px 16px; border-bottom:1px solid var(--border); font-family:'JetBrains Mono',monospace; font-size:0.72rem; transition:background 0.2s ease; gap:8px; }
.hist-fila:hover { background:var(--navy3); }
.hist-fila:last-child { border-bottom:none; }
.hist-encab { background:var(--navy3); font-size:0.5rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--texto-xs); padding:8px 16px; }
.hist-acierto-ok  { color:var(--verde); font-size:1.1rem; }
.hist-acierto-no  { color:var(--rojo);  font-size:1.1rem; }
.hist-pendiente   { color:var(--texto-xs); font-size:0.62rem; letter-spacing:0.1em; }
.hist-stat { background:var(--navy3); border:1px solid var(--border); border-radius:3px; padding:14px 18px; text-align:center; }
.hist-stat-val  { font-family:'Playfair Display',serif; font-size:1.5rem; color:var(--blanco); }
.hist-stat-lbl  { font-family:'JetBrains Mono',monospace; font-size:0.54rem; color:var(--texto-xs); letter-spacing:0.18em; text-transform:uppercase; margin-top:5px; }

.stCheckbox > label { font-family:'JetBrains Mono',monospace !important; font-size:0.76rem !important; color:var(--texto-dim) !important; letter-spacing:0.05em !important; }
footer { display:none !important; }
#MainMenu { display:none !important; }
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
TOKEN   = "8230884463:AAGwo4BgKirkvC9dTkS6Ef_xKnY-jcyJfQ8"
CHAT_ID = "7836739381"
def enviar_alerta(msg):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
    return r.status_code

@st.cache_data(ttl=300)
def obtener_dolar_blue():
    try:
        r = requests.get("https://dolarapi.com/v1/dolares/blue", timeout=5)
        d = r.json(); return float(d['venta']), float(d['compra'])
    except:
        try:
            r = requests.get("https://api.bluelytics.com.ar/v2/latest", timeout=5)
            d = r.json(); return float(d['blue']['value_sell']), float(d['blue']['value_buy'])
        except: return None, None

# ── PANEL LATERAL ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:26px 0 22px 0;'>
        <div style='font-family:"Playfair Display",serif;font-size:1.1rem;color:#f0f6ff;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;'>DJT Capital</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.54rem;color:#3d5a80;letter-spacing:0.25em;text-transform:uppercase;margin-top:5px;'>Investigación Cuantitativa</div>
    </div>
    <hr style='border:none;border-top:1px solid #162035;margin-bottom:18px;'>
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.54rem;color:#3d5a80;letter-spacing:0.25em;text-transform:uppercase;margin-bottom:8px;'>Instrumento</div>
    """, unsafe_allow_html=True)

    query = st.text_input("", value="", placeholder="Ej: YPF, AMZN, TSLA...", label_visibility="collapsed", key="buscador_ticker")

    if "ticker_sel" not in st.session_state:
        st.session_state.ticker_sel = "YPF"

    sugerencias = buscar_tickers(query) if query else []

    if sugerencias and query:
        html_ac = "<div class='ac-contenedor'>"
        html_ac += "<div class='ac-header'>Sugerencias</div>"
        for sym, nombre, sector in sugerencias:
            dot = "ac-arg-dot" if sym in SET_ARG else "ac-usa-dot"
            html_ac += f"""<div class='ac-fila'>
                <span class='ac-ticker'><span class='{dot}'></span>{sym}</span>
                <span class='ac-nombre'>{nombre}</span>
                <span class='ac-sector'>{sector}</span>
            </div>"""
        html_ac += "</div>"
        st.markdown(html_ac, unsafe_allow_html=True)

        opciones = [f"{s[0]} — {s[1]}" for s in sugerencias]
        elegido = st.selectbox("Seleccionar:", opciones, label_visibility="collapsed", key="sel_ac")
        if elegido:
            if st.session_state.get("ticker_sel") != elegido.split(" — ")[0]:
                st.session_state.correr = False
            st.session_state.ticker_sel = elegido.split(" — ")[0]
    elif query and len(query) >= 1 and not sugerencias:
        st.markdown("<div class='ac-contenedor'><div class='ac-vacio'>Sin resultados — se usará el símbolo tal cual</div></div>", unsafe_allow_html=True)
        st.session_state.ticker_sel = query.upper()
    elif not query:
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#3d5a80;padding:6px 2px;letter-spacing:0.06em;'>🔍 &nbsp; Empezá a escribir para ver sugerencias</div>", unsafe_allow_html=True)

    ticker = st.session_state.ticker_sel

    if es_argentino(ticker):
        nombre_ticker = TICKERS_ARG.get(ticker, (ticker, ""))[0]
        st.markdown(f"<div style='margin:10px 0 4px 0;'><span class='badge-arg'>🇦🇷 Mercado Argentino</span></div><div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#7b93b8;margin-top:4px;'>{nombre_ticker}</div>", unsafe_allow_html=True)
    else:
        nombre_ticker = TICKERS_USA.get(ticker, (ticker, ""))[0]
        st.markdown(f"<div style='margin:10px 0 4px 0;'><span class='badge-usa'>🇺🇸 Mercado EEUU</span></div><div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#7b93b8;margin-top:4px;'>{nombre_ticker}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #162035;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.54rem;color:#3d5a80;letter-spacing:0.25em;text-transform:uppercase;margin-bottom:6px;'>Rango histórico</div>", unsafe_allow_html=True)
    fecha_inicio = st.date_input("", value=datetime.date(2019,1,1), label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid #162035;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.54rem;color:#3d5a80;letter-spacing:0.25em;text-transform:uppercase;margin-bottom:10px;'>Configuración de modelos</div>", unsafe_allow_html=True)
    usar_rf  = st.checkbox("Random Forest",  value=True)
    usar_xgb = st.checkbox("XGBoost",        value=True)
    usar_lgb = st.checkbox("LightGBM",       value=True)
    usar_gb  = st.checkbox("Gradient Boost", value=True)

    st.markdown("<hr style='border:none;border-top:1px solid #162035;margin:16px 0;'>", unsafe_allow_html=True)
    enviar_telegram = st.checkbox("Enviar alerta por Telegram", value=True)
    st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
    if st.button("◈  EJECUTAR ANÁLISIS"):
        st.session_state.correr = True
    correr = st.session_state.get("correr", False)
    st.markdown("""</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.5rem;color:#1a2a3a;letter-spacing:0.08em;margin-top:28px;line-height:2;'>
    SOLO USO INTERNO<br>NO ES CONSEJO FINANCIERO<br>© DJT CAPITAL MANAGEMENT
    </div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
ahora_str = datetime.datetime.now().strftime("%d %b %Y  ·  %H:%M")
dolar_venta, dolar_compra = obtener_dolar_blue()
mercado_str = "MERCADO ARGENTINO  ·  ADR NYSE" if es_argentino(ticker) else "MERCADO EEUU  ·  NYSE / NASDAQ"

st.markdown(f"""
<div class='header-bar'>
    <div>
        <div class='header-firma'>DJT Capital Management</div>
        <div class='header-division'>Mesa de Renta Variable · Investigación Cuantitativa · {mercado_str}</div>
    </div>
    <div class='header-derecha'>
        <span class='punto-vivo'></span>DATOS EN VIVO &nbsp;·&nbsp; {ahora_str}
    </div>
</div>
""", unsafe_allow_html=True)

# ── TIRA MACRO ────────────────────────────────────────────────────────────────
st.markdown("<div class='etiqueta-seccion'>FX  ·  MACRO  ·  TIEMPO REAL</div>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
dv = f"ARS {dolar_venta:,.0f}"  if dolar_venta  else "N/D"
dc = f"ARS {dolar_compra:,.0f}" if dolar_compra else "N/D"
spread = f"{((dolar_venta-dolar_compra)/dolar_compra*100):.2f}%" if dolar_venta and dolar_compra else "N/D"
with c1: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>USD/ARS Blue · Venta</div><div class='valor-celda-grande'>{dv}</div><div class='sub-celda'>Mercado paralelo</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>USD/ARS Blue · Compra</div><div class='valor-celda-grande'>{dc}</div><div class='sub-celda'>Mercado paralelo</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Diferencial Compra/Venta</div><div class='valor-celda-med azul'>{spread}</div><div class='sub-celda'>Spread mercado blue</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Última actualización</div><div class='valor-celda-med'>{ahora_str}</div><div class='sub-celda'>Caché automático · 5 min</div></div>", unsafe_allow_html=True)
st.markdown("<hr class='separador'>", unsafe_allow_html=True)

# ── ESTADO INACTIVO ───────────────────────────────────────────────────────────
if not correr:
    nombre_display = nombre_ticker if nombre_ticker != ticker else ticker
    st.markdown(f"""
    <div class='estado-inactivo'>
        <span class='icono-inactivo'>◈</span>
        <div class='titulo-inactivo'>{ticker} &nbsp;·&nbsp; {nombre_display}</div>
        <div class='sub-inactivo'>
            Configure los parámetros en el panel lateral<br>
            y presione EJECUTAR ANÁLISIS para iniciar<br><br>
            <span style='color:#1e3050;'>Tiempo estimado de cómputo: 45 — 90 segundos</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='aviso-legal'>ESTE MATERIAL ES PRODUCIDO POR DJT CAPITAL MANAGEMENT PARA FINES INFORMATIVOS ÚNICAMENTE<br>NO CONSTITUYE ASESORAMIENTO DE INVERSIÓN NI UNA OFERTA DE COMPRA O VENTA DE VALORES</div>", unsafe_allow_html=True)
    st.stop()

# ── DESCARGA DE DATOS ─────────────────────────────────────────────────────────
modo_arg = es_argentino(ticker)

with st.spinner(f"🌐  Descargando datos de mercado · {ticker}..."):
    fs = fecha_inicio.strftime("%Y-%m-%d")
    stock = yf.download(ticker,  start=fs, progress=False)
    wti   = yf.download("CL=F", start=fs, progress=False)
    spy   = yf.download("SPY",  start=fs, progress=False)
    wti_id = yf.download("CL=F", period="5d", interval="1h", progress=False)

    if modo_arg:
        ggal = yf.download("GGAL", start=fs, progress=False)
        try:
            usdars = yf.download("ARS=X", start=fs, progress=False)
            usdars.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in usdars.columns]
            tiene_usdars = True
        except: tiene_usdars = False
    else:
        sector_etfs = {
            "AAPL":"XLK","MSFT":"XLK","GOOGL":"XLK","AMZN":"XLK","NVDA":"XLK","AMD":"XLK","INTC":"XLK",
            "META":"XLC","NFLX":"XLC","DIS":"XLC",
            "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","V":"XLF","MA":"XLF","PYPL":"XLF","SQ":"XLF","COIN":"XLF",
            "XOM":"XLE","CVX":"XLE","COP":"XLE","OXY":"XLE",
            "NKE":"XLY","TSLA":"XLY","UBER":"XLY","WMT":"XLP","KO":"XLP","PEP":"XLP","COST":"XLP",
            "LLY":"XLV","JNJ":"XLV","PFE":"XLV",
        }
        etf_sector = sector_etfs.get(ticker, "QQQ")
        sector_data = yf.download(etf_sector, start=fs, progress=False)
        qqq = yf.download("QQQ", start=fs, progress=False)
        tiene_usdars = False

if len(stock) < 100:
    st.error(f"❌ Datos insuficientes para {ticker}. Verifique el símbolo."); st.stop()

for d in [stock, wti, spy]:
    d.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in d.columns]

df = stock[['open','high','low','close','volume']].copy()
df['wti'] = wti['close'].reindex(df.index).ffill()
df['spy'] = spy['close'].reindex(df.index).ffill()

if modo_arg:
    ggal.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in ggal.columns]
    df['ggal'] = ggal['close'].reindex(df.index).ffill()
else:
    sector_data.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in sector_data.columns]
    qqq.columns         = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in qqq.columns]
    df['sector'] = sector_data['close'].reindex(df.index).ffill()
    df['qqq']    = qqq['close'].reindex(df.index).ffill()

# ── INDICADORES TÉCNICOS ──────────────────────────────────────────────────────
with st.spinner("⚙  Calculando indicadores técnicos..."):
    df['MA5']  = df['close'].rolling(5).mean()
    df['MA7']  = df['close'].rolling(7).mean()
    df['MA21'] = df['close'].rolling(21).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA200']= df['close'].rolling(200).mean()
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA12']= df['close'].ewm(span=12).mean()
    df['EMA26']= df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/loss))
    g7=delta.clip(lower=0).rolling(7).mean(); l7=(-delta.clip(upper=0)).rolling(7).mean()
    df['RSI7'] = 100 - (100/(1+g7/l7))

    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    df['BB_mid']   = df['close'].rolling(20).mean()
    df['BB_std']   = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
    df['BB_width'] = (df['BB_upper']-df['BB_lower'])/df['BB_mid']
    df['BB_pos']   = (df['close']-df['BB_lower'])/(df['BB_upper']-df['BB_lower'])

    tr = pd.concat([df['high']-df['low'],
                    (df['high']-df['close'].shift()).abs(),
                    (df['low'] -df['close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR']     = tr.rolling(14).mean()
    df['ATR_pct'] = df['ATR']/df['close']

    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['volatility_10'] = df['ret_1d'].rolling(10).std()
    df['volatility_21'] = df['ret_1d'].rolling(21).std()
    df['momentum_10']   = df['close']/df['close'].shift(10) - 1
    df['vol_ratio']     = df['volume']/df['volume'].rolling(20).mean()

    df['wti_ret']    = df['wti'].pct_change()
    df['wti_mom_3d'] = df['wti'].pct_change(3)
    df['wti_mom_5d'] = df['wti'].pct_change(5)
    df['wti_vol_10'] = df['wti_ret'].rolling(10).std()

    df['spy_ret'] = df['spy'].pct_change()

    if modo_arg:
        df['ggal_ret']     = df['ggal'].pct_change()
        df['ypf_wti_ratio']= df['close']/df['wti']
        df['ratio_zscore'] = (df['ypf_wti_ratio']-df['ypf_wti_ratio'].rolling(20).mean())/df['ypf_wti_ratio'].rolling(20).std()
        if tiene_usdars:
            df['usdars']       = usdars['close'].reindex(df.index).ffill()
            df['usdars_ret']   = df['usdars'].pct_change()
            df['usdars_mom_5d']= df['usdars'].pct_change(5)
            df['ars_zscore']   = (df['usdars']-df['usdars'].rolling(20).mean())/df['usdars'].rolling(20).std()
        else:
            df['usdars_ret']=df['usdars_mom_5d']=df['ars_zscore']=0.0
    else:
        df['sector_ret']    = df['sector'].pct_change()
        df['sector_mom_3d'] = df['sector'].pct_change(3)
        df['qqq_ret']       = df['qqq'].pct_change()
        df['beta_ratio']    = df['close']/df['spy']
        df['beta_zscore']   = (df['beta_ratio']-df['beta_ratio'].rolling(20).mean())/df['beta_ratio'].rolling(20).std()
        df['rel_spy']       = df['close'].pct_change(5) - df['spy'].pct_change(5)

    df.dropna(inplace=True)

# ── FEATURES ──────────────────────────────────────────────────────────────────
features_base = ['MA5','EMA9','EMA12','MA21','MA7','RSI','RSI7','MACD','MACD_signal','MACD_hist',
                 'BB_pos','BB_width','ATR_pct','ret_1d','ret_3d','ret_5d',
                 'volatility_10','volatility_21','vol_ratio','momentum_10',
                 'spy_ret','wti_ret','wti_mom_3d','wti_mom_5d','wti_vol_10']

if modo_arg:
    features = features_base + ['ggal_ret','ratio_zscore','usdars_ret','usdars_mom_5d','ars_zscore']
else:
    features = features_base + ['sector_ret','sector_mom_3d','qqq_ret','beta_zscore','rel_spy']

df['target'] = df['close'].pct_change(-1)*-1
df.dropna(inplace=True)

_MIN_FILAS = 60
if len(df) < _MIN_FILAS:
    st.error(
        f"⚠️ **Datos insuficientes para {ticker}**: solo se obtuvieron **{len(df)} registros** "
        f"válidos tras limpiar NaN (mínimo requerido: {_MIN_FILAS}).\n\n"
        "Probá cambiando la **fecha de inicio** a una más antigua (ej: hace 2 años)."
    )
    st.stop()

X = df[features]; y = df['target']
split = int(len(df)*0.8)
if split < 1:
    st.error(f"⚠️ Split de entrenamiento vacío con {len(df)} filas. Ampliá el rango de fechas.")
    st.stop()
X_train,X_test = X[:split],X[split:]
y_train,y_test = y[:split],y[split:]
if len(X_test) < 1:
    X_train,X_test = X[:-10],X[-10:]
    y_train,y_test = y[:-10],y[-10:]
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── ENTRENAMIENTO ─────────────────────────────────────────────────────────────
modelos,nombres,preds=[],[],[]
with st.spinner("🧠  Entrenando ensamble de modelos de IA..."):
    if usar_rf:
        m=RandomForestRegressor(n_estimators=300,max_depth=10,random_state=42)
        m.fit(X_train_sc,y_train); preds.append(m.predict(X_test_sc)); modelos.append(m); nombres.append("Random Forest")
    if usar_xgb:
        m=xgb.XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
        m.fit(X_train_sc,y_train); preds.append(m.predict(X_test_sc)); modelos.append(m); nombres.append("XGBoost")
    if usar_lgb:
        m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,random_state=42,verbose=-1)
        m.fit(X_train_sc,y_train); preds.append(m.predict(X_test_sc)); modelos.append(m); nombres.append("LightGBM")
    if usar_gb:
        m=GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=5,random_state=42)
        m.fit(X_train_sc,y_train); preds.append(m.predict(X_test_sc)); modelos.append(m); nombres.append("Grad. Boost")

if not preds: st.error("❌ Activá al menos un modelo."); st.stop()

maes  = [mean_absolute_error(y_test,p) for p in preds]
pesos = [1/m for m in maes]; pesos=[p/sum(pesos) for p in pesos]
p_ens = sum(w*p for w,p in zip(pesos,preds))
mae_r = mean_absolute_error(y_test,p_ens)
precio_hoy = float(df['close'].iloc[-1])
mae_usd    = mae_r*precio_hoy
ultimo_sc  = scaler.transform(X.iloc[[-1]])
ret_pred   = sum(w*m.predict(ultimo_sc)[0] for w,m in zip(pesos,modelos))

if len(wti_id)>0:
    wti_id.columns=[c[0].lower() if isinstance(c,tuple) else c.lower() for c in wti_id.columns]
    wti_intraday_ret=(float(wti_id['close'].iloc[-1])-float(df['wti'].iloc[-2]))/float(df['wti'].iloc[-2])
else: wti_intraday_ret=0.0

ajuste_wti    = wti_intraday_ret*0.6 if modo_arg else wti_intraday_ret*0.15
ajuste_dolar  = 0.0
if modo_arg and dolar_venta and tiene_usdars:
    ajuste_dolar = -(dolar_venta-float(df['usdars'].iloc[-1]))/float(df['usdars'].iloc[-1])*0.3

ret_final  = ret_pred + ajuste_wti + ajuste_dolar
pred_final = precio_hoy*(1+ret_final)
variacion  = ret_final*100

proximo_dia = (pd.Timestamp.today()+offsets.BDay(1)).strftime('%A %d %b %Y')
dia_es = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles',
          'Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
proximo_es = dia_es.get(proximo_dia.split()[0], proximo_dia.split()[0]) + " " + " ".join(proximo_dia.split()[1:])
wti_s  = f"{wti_intraday_ret*100:+.2f}%" if wti_intraday_ret else "N/D"
dolar_s= f"ARS {dolar_venta:,.0f}" if dolar_venta else "N/D"

# ── SEÑAL ─────────────────────────────────────────────────────────────────────
st.markdown(f"<div class='etiqueta-seccion'>{ticker}  ·  PRONÓSTICO DE PRECIO  ·  {proximo_es.upper()}</div>", unsafe_allow_html=True)

if variacion>3:    senal,clase,rec="SOBREPONDERAR","compra","Compra Fuerte"
elif variacion>0:  senal,clase,rec="LEVE ALZA","compra","Compra"
elif variacion<-3: senal,clase,rec="SUBPONDERAR","venta","Venta Fuerte"
else:              senal,clase,rec="LEVE BAJA","venta","Venta"

t_clase  = "tarjeta-compra" if clase=="compra" else "tarjeta-venta"
v_clase  = "valor-senal-compra" if clase=="compra" else "valor-senal-venta"

sc1,sc2=st.columns([1,2])
with sc1:
    mercado_badge = "🇦🇷 ARG" if modo_arg else "🇺🇸 USA"
    st.markdown(f"""
    <div class='{t_clase}'>
        <div class='tipo-senal'>Señal del Modelo · {ticker} · {mercado_badge}</div>
        <div class='{v_clase}'>{senal}</div>
        <div class='sub-senal'>Recomendación: <b style='color:#f0f6ff'>{rec}</b></div>
        <div class='sub-senal' style='margin-top:10px;'>
            Actual: <b style='color:#f0f6ff'>${precio_hoy:.2f}</b>
            &nbsp;·&nbsp;
            Objetivo: <b style='color:#f0f6ff'>${pred_final:.2f}</b>
        </div>
    </div>""", unsafe_allow_html=True)

with sc2:
    color_var="verde" if variacion>0 else "rojo"
    m1c,m2c,m3c,m4c=st.columns(4)
    with m1c: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Retorno esperado</div><div class='valor-celda-grande {color_var}'>{variacion:+.2f}%</div><div class='sub-celda'>Pronóstico 1 día</div></div>", unsafe_allow_html=True)
    with m2c: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Precio objetivo</div><div class='valor-celda-grande'>${pred_final:.2f}</div><div class='sub-celda'>{proximo_es[:3]}</div></div>", unsafe_allow_html=True)
    with m3c: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Rango probable</div><div class='valor-celda-med'>${pred_final-mae_usd:.2f} — ${pred_final+mae_usd:.2f}</div><div class='sub-celda'>Banda ±1 MAE</div></div>", unsafe_allow_html=True)
    with m4c: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Error del modelo (MAE)</div><div class='valor-celda-med azul'>±${mae_usd:.2f}</div><div class='sub-celda'>{len(df)} observaciones</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MÉTRICAS TÉCNICAS ─────────────────────────────────────────────────────────
a1,a2,a3,a4=st.columns(4)
wti_col="verde" if wti_intraday_ret>0 else "rojo"
rsi_val=float(df['RSI'].iloc[-1])
rsi_col="rojo" if rsi_val>70 else "verde" if rsi_val<30 else "azul"
rsi_etq="Sobrecomprado" if rsi_val>70 else "Sobrevendido" if rsi_val<30 else "Neutral"
macd_val=float(df['MACD_hist'].iloc[-1])
macd_col="verde" if macd_val>0 else "rojo"

with a1: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>WTI Crudo · Intradía</div><div class='valor-celda-med {wti_col}'>{wti_s}</div><div class='sub-celda'>Ajuste aplicado: {ajuste_wti*100:+.3f}%</div></div>", unsafe_allow_html=True)
with a2:
    if modo_arg:
        st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>USD/ARS Blue</div><div class='valor-celda-med'>{dolar_s}</div><div class='sub-celda'>Ajuste: {ajuste_dolar*100:+.3f}%</div></div>", unsafe_allow_html=True)
    else:
        beta_lbl = f"{(df['close'].pct_change().rolling(60).corr(df['spy'].pct_change())).iloc[-1]:.2f}"
        st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Correlación vs S&P500</div><div class='valor-celda-med'>{beta_lbl}</div><div class='sub-celda'>60 períodos · rolling</div></div>", unsafe_allow_html=True)
with a3: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>RSI (14 períodos)</div><div class='valor-celda-med {rsi_col}'>{rsi_val:.1f}</div><div class='sub-celda'>{rsi_etq}</div></div>", unsafe_allow_html=True)
with a4: st.markdown(f"<div class='celda-dato'><div class='etiqueta-celda'>Histograma MACD</div><div class='valor-celda-med {macd_col}'>{macd_val:+.4f}</div><div class='sub-celda'>{'Momentum alcista' if macd_val>0 else 'Momentum bajista'}</div></div>", unsafe_allow_html=True)

st.markdown("<hr class='separador'>", unsafe_allow_html=True)

# ── GRÁFICO ───────────────────────────────────────────────────────────────────
st.markdown("<div class='etiqueta-seccion'>GRÁFICO DE PRECIO  ·  SUPERPOSICIÓN TÉCNICA</div>", unsafe_allow_html=True)

test_index   = df.index[split:]
close_test   = df['close'].iloc[split:].values
pred_precios = close_test*(1+p_ens)

fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.52,0.18,0.18,0.12],
    vertical_spacing=0.02,
    subplot_titles=(f'Precio + Pronóstico IA + Bollinger · {ticker}','RSI','MACD','Volumen'))

fig.add_trace(go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],
    name=ticker,increasing_line_color='#05d890',increasing_fillcolor='#032d17',
    decreasing_line_color='#f43f5e',decreasing_fillcolor='#2d0310'),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['BB_upper'],name='BB Superior',line=dict(color='rgba(59,130,246,0.3)',dash='dot',width=1)),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['BB_lower'],name='BB Inferior',line=dict(color='rgba(59,130,246,0.3)',dash='dot',width=1),fill='tonexty',fillcolor='rgba(59,130,246,0.03)'),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['MA21'], name='MA21', line=dict(color='#f59e0b',width=1.3)),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['MA50'], name='MA50', line=dict(color='#60a5fa',width=1.3)),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['MA200'],name='MA200',line=dict(color='#7b93b8',width=1)),row=1,col=1)
fig.add_trace(go.Scatter(x=test_index,y=pred_precios,name='Pronóstico IA',line=dict(color='#3b82f6',width=2,dash='dash')),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['RSI'], name='RSI', line=dict(color='#a78bfa',width=1.3)),row=2,col=1)
fig.add_hline(y=70,line_dash="dot",line_color="#f43f5e",line_width=1,row=2,col=1)
fig.add_hline(y=30,line_dash="dot",line_color="#05d890",line_width=1,row=2,col=1)
fig.add_hrect(y0=70,y1=100,fillcolor="rgba(244,63,94,0.04)",  line_width=0,row=2,col=1)
fig.add_hrect(y0=0, y1=30, fillcolor="rgba(5,216,144,0.04)",  line_width=0,row=2,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],        name='MACD',  line=dict(color='#60a5fa',width=1.3)),row=3,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['MACD_signal'], name='Señal', line=dict(color='#f59e0b',width=1)),row=3,col=1)
ch=['#05d890' if v>=0 else '#f43f5e' for v in df['MACD_hist']]
fig.add_trace(go.Bar(x=df.index,y=df['MACD_hist'],name='Histograma',marker_color=ch,opacity=0.7),row=3,col=1)
fig.add_trace(go.Bar(x=df.index,y=df['volume'],name='Volumen',marker_color='rgba(59,130,246,0.18)'),row=4,col=1)

fig.update_layout(template='plotly_dark',paper_bgcolor='#060d1a',plot_bgcolor='#090f1e',
    height=800,showlegend=True,xaxis_rangeslider_visible=False,
    font=dict(family='JetBrains Mono, monospace',color='#3d5a80',size=10),
    legend=dict(bgcolor='rgba(6,13,26,0.9)',bordercolor='#162035',borderwidth=1,font=dict(size=10)),
    margin=dict(l=0,r=0,t=30,b=0))
for i in range(1,5):
    fig.update_xaxes(gridcolor='#0d1626',showgrid=True,row=i,col=1,zeroline=False)
    fig.update_yaxes(gridcolor='#0d1626',showgrid=True,row=i,col=1,zeroline=False)

st.plotly_chart(fig,use_container_width=True)

# ── TABLA MODELOS ─────────────────────────────────────────────────────────────
st.markdown("<hr class='separador'>", unsafe_allow_html=True)
st.markdown("<div class='etiqueta-seccion'>RENDIMIENTO DEL ENSAMBLE DE MODELOS</div>", unsafe_allow_html=True)

filas  = "<div class='tabla-modelos'>"
filas += "<div class='fila-modelo fila-encabezado'><span>MODELO</span><span>ERROR MAE</span><span>PESO</span><span>ESTADO</span></div>"
for n,mae,peso in zip(nombres,maes,pesos):
    ancho = int(peso*100)
    filas += f"""<div class='fila-modelo'>
        <span class='nombre-modelo'>{n}</span>
        <span class='mae-modelo'>±${mae*precio_hoy:.4f}</span>
        <span style='color:#7b93b8;'>{peso*100:.1f}%<div class='barra-peso' style='width:{ancho}px;'></div></span>
        <span class='badge-activo'>ACTIVO</span>
    </div>"""
filas += f"""<div class='fila-modelo fila-ensamble'>
    <span style='color:#f0f6ff;font-weight:600;'>◈ ENSAMBLE</span>
    <span class='mae-ensamble'>±${mae_usd:.4f}</span>
    <span style='color:#3b82f6;'>100.0%</span>
    <span class='badge-activo'>ACTIVO</span>
</div></div>"""
st.markdown(filas, unsafe_allow_html=True)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
if enviar_telegram:
    mercado_t = "🇦🇷 Mercado ARG" if modo_arg else "🇺🇸 Mercado EEUU"
    if variacion>3:    s="🚀 SOBREPONDERAR · Compra Fuerte"
    elif variacion>0:  s="📈 LEVE ALZA · Compra"
    elif variacion<-3: s="🔴 SUBPONDERAR · Venta Fuerte"
    else:              s="📉 LEVE BAJA · Venta"
    msg = (f"<b>DJT Capital Research · {ticker}</b>  {mercado_t}\n{'─'*32}\n"
           f"<b>Señal:</b> {s}\n"
           f"<b>Precio actual:</b> ${precio_hoy:.2f}\n"
           f"<b>Objetivo ({proximo_es[:3]}):</b> ${pred_final:.2f}\n"
           f"<b>Variación:</b> {variacion:+.2f}%\n"
           f"<b>Rango:</b> ${pred_final-mae_usd:.2f} — ${pred_final+mae_usd:.2f}\n"
           f"<b>MAE:</b> ±${mae_usd:.2f}\n{'─'*32}\n")
    if modo_arg:
        msg += f"USD/ARS Blue: {dolar_s}  |  WTI: {wti_s}\n{'─'*32}\n"
    else:
        msg += f"WTI: {wti_s}  |  S&P500 correlation aplicada\n{'─'*32}\n"
    msg += "<i>Solo con fines informativos. No es consejo financiero.</i>"
    status = enviar_alerta(msg)
    if status==200: st.success("✅ Alerta enviada correctamente por Telegram")
    else:           st.warning(f"⚠ Error de Telegram: código {status}")

st.markdown("<div class='aviso-legal'>ESTE MATERIAL ES PRODUCIDO POR DJT CAPITAL MANAGEMENT PARA FINES INFORMATIVOS ÚNICAMENTE<br>NO CONSTITUYE ASESORAMIENTO DE INVERSIÓN NI UNA OFERTA DE COMPRA O VENTA DE VALORES · EL RENDIMIENTO PASADO NO GARANTIZA RESULTADOS FUTUROS</div>", unsafe_allow_html=True)

# ── GUARDAR PREDICCIÓN ────────────────────────────────────────────────────────
guardar_prediccion({
    "fecha":                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "ticker":               ticker,
    "mercado":              "ARG" if modo_arg else "USA",
    "precio_actual":        round(precio_hoy, 4),
    "precio_objetivo":      round(pred_final, 4),
    "variacion_pct":        round(variacion, 4),
    "rango_min":            round(pred_final - mae_usd, 4),
    "rango_max":            round(pred_final + mae_usd, 4),
    "mae_usd":              round(mae_usd, 4),
    "senal":                rec,
    "precio_real_siguiente":"",
    "error_real_usd":       "",
    "acierto":              "",
})

st.markdown("<hr class='separador'>", unsafe_allow_html=True)

with st.expander("◈  INTELIGENCIA AVANZADA  ·  VOLUME PROFILE · SMART MONEY · CORRELACIONES MACRO", expanded=False):

    st.markdown("""<style>
    /* ── EXPANDER PERSONALIZADO ── */
    details > summary {
        font-family:'JetBrains Mono',monospace !important;
        font-size:0.68rem !important;
        letter-spacing:0.18em !important;
        text-transform:uppercase !important;
        color:var(--acento2) !important;
        cursor:pointer !important;
        padding:14px 0 !important;
    }
    details[open] > summary { color:var(--verde) !important; }
    [data-testid="stExpander"] {
        background: var(--navy2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 3px !important;
    }
    .adv-titulo {
        font-family:'JetBrains Mono',monospace;
        font-size:0.54rem; letter-spacing:0.3em; text-transform:uppercase;
        color:var(--texto-xs); border-left:2px solid var(--verde);
        padding-left:10px; margin:20px 0 14px 0;
    }
    .vp-barra-cont {
        display:flex; align-items:center; gap:8px;
        margin-bottom:3px; font-family:'JetBrains Mono',monospace;
    }
    .vp-precio { font-size:0.62rem; color:var(--texto-dim); min-width:62px; text-align:right; }
    .vp-barra-wrap { flex:1; height:12px; background:var(--navy3); border-radius:2px; overflow:hidden; }
    .vp-barra-fill { height:100%; border-radius:2px; transition:width 1s ease; }
    .vp-vol { font-size:0.58rem; color:var(--texto-xs); min-width:55px; }
    .vp-poc { border:1px solid var(--dorado) !important; background:rgba(245,158,11,0.08) !important; }
    .vp-vah { border:1px solid rgba(5,216,144,0.3) !important; }
    .vp-val { border:1px solid rgba(244,63,94,0.3) !important; }
    .sm-fila {
        display:grid; grid-template-columns:90px 70px 80px 80px 80px 1fr;
        align-items:center; padding:8px 14px;
        border-bottom:1px solid var(--border);
        font-family:'JetBrains Mono',monospace; font-size:0.7rem;
        gap:8px; transition:background 0.2s;
    }
    .sm-fila:hover { background:var(--navy3); }
    .sm-encab { font-size:0.5rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--texto-xs); background:var(--navy3); }
    .sm-whale { color:var(--dorado); font-weight:700; }
    .sm-bull  { color:var(--verde); }
    .sm-bear  { color:var(--rojo); }
    .corr-celda {
        background:var(--navy2); border:1px solid var(--border);
        border-radius:3px; padding:14px 16px; text-align:center;
        transition:all 0.3s ease;
    }
    .corr-celda:hover { border-color:var(--border2); transform:translateY(-1px); }
    .corr-val { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:600; }
    .corr-lbl { font-family:'JetBrains Mono',monospace; font-size:0.52rem; color:var(--texto-xs); letter-spacing:0.2em; text-transform:uppercase; margin-top:5px; }
    .corr-desc { font-family:'JetBrains Mono',monospace; font-size:0.58rem; margin-top:4px; }
    .leyenda-vp {
        display:flex; gap:18px; margin-bottom:12px;
        font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:var(--texto-xs);
    }
    .ley-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; vertical-align:middle; }
    </style>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 1. VOLUME PROFILE
    # ─────────────────────────────────────────────────────────────
    st.markdown("<div class='adv-titulo'>① VOLUME PROFILE  ·  NIVELES DE PRECIO CON MAYOR ACTIVIDAD INSTITUCIONAL</div>", unsafe_allow_html=True)

    # Calcular Volume Profile sobre los últimos 252 días (1 año aprox)
    df_vp = df.tail(252).copy()
    precio_min = df_vp['low'].min()
    precio_max = df_vp['high'].max()
    N_BINS = 30
    bins = np.linspace(precio_min, precio_max, N_BINS + 1)
    vol_por_bin = np.zeros(N_BINS)

    for _, row_vp in df_vp.iterrows():
        for b in range(N_BINS):
            low_b, high_b = bins[b], bins[b+1]
            overlap = max(0, min(row_vp['high'], high_b) - max(row_vp['low'], low_b))
            rango_vela = row_vp['high'] - row_vp['low']
            if rango_vela > 0:
                frac = overlap / rango_vela
                vol_por_bin[b] += row_vp['volume'] * frac

    # POC, VAH, VAL
    poc_idx = np.argmax(vol_por_bin)
    poc_precio = (bins[poc_idx] + bins[poc_idx+1]) / 2

    total_vol = vol_por_bin.sum()
    vol_acum = 0
    va_indices = sorted(range(N_BINS), key=lambda i: vol_por_bin[i], reverse=True)
    va_set = set()
    for i in va_indices:
        va_set.add(i)
        vol_acum += vol_por_bin[i]
        if vol_acum >= total_vol * 0.70:
            break
    vah_idx = max(va_set)
    val_idx  = min(va_set)
    vah_precio = (bins[vah_idx] + bins[vah_idx+1]) / 2
    val_precio  = (bins[val_idx]  + bins[val_idx+1]) / 2

    vp_col1, vp_col2 = st.columns([2, 1])

    with vp_col1:
        # HTML del Volume Profile
        max_vol = vol_por_bin.max()
        html_vp = ""
        for b in range(N_BINS - 1, -1, -1):
            precio_centro = (bins[b] + bins[b+1]) / 2
            pct_ancho = (vol_por_bin[b] / max_vol * 100)
            vol_fmt = f"{vol_por_bin[b]/1e6:.1f}M" if vol_por_bin[b] > 1e6 else f"{vol_por_bin[b]/1e3:.0f}K"

            # Color según zona
            if b == poc_idx:
                color = "#f59e0b"
                clase_extra = "vp-poc"
            elif b in va_set and b >= val_idx and b <= vah_idx:
                # Dentro del Value Area
                if vol_por_bin[b] > total_vol * 0.03:
                    color = "rgba(59,130,246,0.7)"
                else:
                    color = "rgba(59,130,246,0.3)"
                clase_extra = ""
            else:
                color = "rgba(59,130,246,0.18)"
                clase_extra = ""

            # Marcar VAH y VAL
            if b == vah_idx: clase_extra = "vp-vah"
            if b == val_idx:  clase_extra = "vp-val"

            es_actual = abs(precio_centro - precio_hoy) < (precio_max - precio_min) / N_BINS
            precio_fmt = f"${precio_centro:.2f}"
            marker = " ◄ actual" if es_actual else ""

            html_vp += f"""
            <div class='vp-barra-cont {clase_extra}' style='{"background:rgba(255,255,255,0.03);border-radius:2px;" if es_actual else ""}'>
                <span class='vp-precio'>{precio_fmt}<span style='color:#3d5a80;font-size:0.5rem;'>{marker}</span></span>
                <div class='vp-barra-wrap'>
                    <div class='vp-barra-fill' style='width:{pct_ancho:.1f}%;background:{color};'></div>
                </div>
                <span class='vp-vol'>{vol_fmt}</span>
            </div>"""

        st.markdown(f"""
        <div class='leyenda-vp'>
            <span><span class='ley-dot' style='background:#f59e0b;'></span>POC ${poc_precio:.2f}</span>
            <span><span class='ley-dot' style='background:rgba(5,216,144,0.7);'></span>VAH ${vah_precio:.2f}</span>
            <span><span class='ley-dot' style='background:rgba(244,63,94,0.7);'></span>VAL ${val_precio:.2f}</span>
            <span><span class='ley-dot' style='background:rgba(59,130,246,0.5);'></span>Value Area (70% vol)</span>
        </div>
        <div style='background:var(--navy3);border:1px solid var(--border);border-radius:3px;padding:12px 16px;max-height:400px;overflow-y:auto;'>
            {html_vp}
        </div>
        """, unsafe_allow_html=True)

    with vp_col2:
        dist_poc = (precio_hoy - poc_precio) / poc_precio * 100
        dist_vah = (precio_hoy - vah_precio) / vah_precio * 100
        dist_val = (precio_hoy - val_precio) / val_precio * 100

        if precio_hoy > vah_precio:
            zona_actual = "SOBRE VALUE AREA"
            zona_col = "#f59e0b"
            zona_desc = "Precio extendido arriba · posible retorno al VA"
        elif precio_hoy < val_precio:
            zona_actual = "BAJO VALUE AREA"
            zona_col = "#f43f5e"
            zona_desc = "Precio extendido abajo · soporte en VAL"
        else:
            zona_actual = "DENTRO VALUE AREA"
            zona_col = "#05d890"
            zona_desc = "Zona de equilibrio · alta liquidez"

        st.markdown(f"""
        <div style='display:flex;flex-direction:column;gap:10px;'>
            <div class='celda-dato'>
                <div class='etiqueta-celda'>POC — Point of Control</div>
                <div class='valor-celda-med dorado'>${poc_precio:.2f}</div>
                <div class='sub-celda'>Precio con mayor volumen histórico</div>
                <div class='sub-celda' style='color:{"var(--verde)" if dist_poc < 0 else "var(--rojo)"};'>{dist_poc:+.2f}% vs actual</div>
            </div>
            <div class='celda-dato'>
                <div class='etiqueta-celda'>VAH — Value Area High</div>
                <div class='valor-celda-med verde'>${vah_precio:.2f}</div>
                <div class='sub-celda'>Techo del 70% del volumen</div>
                <div class='sub-celda' style='color:var(--texto-xs);'>{dist_vah:+.2f}% vs actual</div>
            </div>
            <div class='celda-dato'>
                <div class='etiqueta-celda'>VAL — Value Area Low</div>
                <div class='valor-celda-med rojo'>${val_precio:.2f}</div>
                <div class='sub-celda'>Piso del 70% del volumen</div>
                <div class='sub-celda' style='color:var(--texto-xs);'>{dist_val:+.2f}% vs actual</div>
            </div>
            <div class='celda-dato' style='border-left:3px solid {zona_col};'>
                <div class='etiqueta-celda'>Zona actual</div>
                <div class='valor-celda-med' style='color:{zona_col};font-size:0.82rem;'>{zona_actual}</div>
                <div class='sub-celda'>{zona_desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='separador'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 2. SMART MONEY DETECTION
    # ─────────────────────────────────────────────────────────────
    st.markdown("<div class='adv-titulo'>② DETECCIÓN DE SMART MONEY  ·  VELAS CON ACTIVIDAD INSTITUCIONAL ANORMAL</div>", unsafe_allow_html=True)

    df_sm = df.copy()
    vol_media     = df_sm['volume'].rolling(20).mean()
    vol_std       = df_sm['volume'].rolling(20).std()
    df_sm['vol_zscore'] = (df_sm['volume'] - vol_media) / vol_std
    df_sm['body_size']  = abs(df_sm['close'] - df_sm['open']) / df_sm['open'] * 100
    df_sm['es_alcista'] = df_sm['close'] > df_sm['open']

    # Whale: volumen > 2.5 sigma Y movimiento > 0.8%
    whales = df_sm[
        (df_sm['vol_zscore'] > 2.5) &
        (df_sm['body_size'] > 0.8)
    ].tail(20).iloc[::-1]

    sm_filas = f"""
    <div style='background:var(--navy2);border:1px solid var(--border);border-radius:3px;overflow:hidden;'>
        <div class='sm-fila sm-encab'>
            <span>FECHA</span><span>PRECIO</span><span>VOLUMEN</span>
            <span>Z-SCORE</span><span>MOVIMIENTO</span><span>SEÑAL</span>
        </div>
    """

    if len(whales) == 0:
        sm_filas += "<div style='padding:20px;text-align:center;font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#3d5a80;'>Sin eventos de Smart Money detectados en el período reciente</div>"
    else:
        for _, r in whales.iterrows():
            fecha_fmt = r.name.strftime("%d %b %Y") if hasattr(r.name, 'strftime') else str(r.name)[:10]
            dir_clase = "sm-bull" if r['es_alcista'] else "sm-bear"
            dir_flecha = "▲" if r['es_alcista'] else "▼"
            dir_txt = "COMPRA INSTITUCIONAL" if r['es_alcista'] else "VENTA INSTITUCIONAL"
            vol_fmt = f"{r['volume']/1e6:.1f}M" if r['volume'] > 1e6 else f"{r['volume']/1e3:.0f}K"
            intensidad = "🐳 WHALE" if r['vol_zscore'] > 4 else "🔥 FUERTE" if r['vol_zscore'] > 3 else "⚡ NOTABLE"

            sm_filas += f"""
            <div class='sm-fila'>
                <span style='color:#4a6080;'>{fecha_fmt}</span>
                <span style='color:#c8d8e8;'>${r['close']:.2f}</span>
                <span class='sm-whale'>{vol_fmt}</span>
                <span style='color:#f59e0b;'>+{r['vol_zscore']:.1f}σ</span>
                <span class='{dir_clase}'>{dir_flecha} {r['body_size']:.2f}%</span>
                <span class='{dir_clase}'>{intensidad} {dir_txt}</span>
            </div>"""

    sm_filas += "</div>"
    st.markdown(sm_filas, unsafe_allow_html=True)

    # Gráfico de volumen con marcas de whale
    fig_sm = go.Figure()
    colores_vol = ['rgba(5,216,144,0.5)' if c > o else 'rgba(244,63,94,0.5)'
                   for c, o in zip(df_sm['close'].tail(120), df_sm['open'].tail(120))]
    fig_sm.add_trace(go.Bar(
        x=df_sm.index[-120:], y=df_sm['volume'].tail(120),
        marker_color=colores_vol, name='Volumen', opacity=0.7
    ))
    # Marcar whales en el gráfico
    whale_idx_graf = df_sm[df_sm['vol_zscore'] > 2.5].tail(120)
    if len(whale_idx_graf) > 0:
        fig_sm.add_trace(go.Scatter(
            x=whale_idx_graf.index, y=whale_idx_graf['volume'],
            mode='markers', name='Smart Money',
            marker=dict(color='#f59e0b', size=10, symbol='diamond',
                       line=dict(color='#f59e0b', width=1))
        ))
    fig_sm.add_trace(go.Scatter(
        x=df_sm.index[-120:], y=vol_media.tail(120)*2,
        name='2× Media', line=dict(color='rgba(244,63,94,0.5)', dash='dot', width=1)
    ))
    fig_sm.update_layout(
        template='plotly_dark', paper_bgcolor='#060d1a', plot_bgcolor='#090f1e',
        height=220, showlegend=True, margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family='JetBrains Mono, monospace', color='#3d5a80', size=10),
        legend=dict(bgcolor='rgba(6,13,26,0.9)', bordercolor='#162035', borderwidth=1, font=dict(size=9)),
        xaxis=dict(gridcolor='#0d1626'), yaxis=dict(gridcolor='#0d1626')
    )
    st.plotly_chart(fig_sm, use_container_width=True)

    st.markdown("<hr class='separador'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 3. CORRELACIONES MACRO
    # ─────────────────────────────────────────────────────────────
    st.markdown("<div class='adv-titulo'>③ CORRELACIONES MACRO  ·  RELACIÓN CON ACTIVOS GLOBALES (ROLLING 60 DÍAS)</div>", unsafe_allow_html=True)

    ret_ticker = df['close'].pct_change().dropna()
    ret_wti_m  = df['wti'].pct_change().dropna()
    ret_spy_m  = df['spy'].pct_change().dropna()

    # Correlaciones rolling 60d
    idx_comun = ret_ticker.index.intersection(ret_wti_m.index).intersection(ret_spy_m.index)
    df_corr = pd.DataFrame({
        'ticker': ret_ticker[idx_comun],
        'wti':    ret_wti_m[idx_comun],
        'spy':    ret_spy_m[idx_comun],
    })

    corr_wti_roll = df_corr['ticker'].rolling(60).corr(df_corr['wti'])
    corr_spy_roll = df_corr['ticker'].rolling(60).corr(df_corr['spy'])

    corr_wti_actual = float(corr_wti_roll.iloc[-1])
    corr_spy_actual = float(corr_spy_roll.iloc[-1])

    def corr_color(v):
        if v > 0.6:   return "#05d890"
        elif v > 0.3: return "#60a5fa"
        elif v > 0:   return "#7b93b8"
        elif v > -0.3:return "#fb923c"
        else:          return "#f43f5e"

    def corr_desc(v, activo):
        if abs(v) < 0.2:   return f"Baja correlación con {activo}"
        elif abs(v) < 0.5: return f"Correlación moderada con {activo}"
        else:               return f"Alta correlación con {activo}"

    cc1, cc2, cc3, cc4 = st.columns(4)

    with cc1:
        cv = corr_wti_actual
        st.markdown(f"""<div class='corr-celda'>
            <div class='corr-val' style='color:{corr_color(cv)};'>{cv:+.2f}</div>
            <div class='corr-lbl'>vs WTI Crudo</div>
            <div class='corr-desc' style='color:{corr_color(cv)};'>{corr_desc(cv, "petróleo")}</div>
        </div>""", unsafe_allow_html=True)

    with cc2:
        cv = corr_spy_actual
        st.markdown(f"""<div class='corr-celda'>
            <div class='corr-val' style='color:{corr_color(cv)};'>{cv:+.2f}</div>
            <div class='corr-lbl'>vs S&P 500</div>
            <div class='corr-desc' style='color:{corr_color(cv)};'>{corr_desc(cv, "mercado USA")}</div>
        </div>""", unsafe_allow_html=True)

    if modo_arg and tiene_usdars:
        ret_usdars = df['usdars'].pct_change().reindex(idx_comun).dropna()
        idx_arg = idx_comun.intersection(ret_usdars.index)
        if len(idx_arg) > 60:
            corr_dolar = float(df_corr['ticker'].reindex(idx_arg).rolling(60).corr(ret_usdars.reindex(idx_arg)).iloc[-1])
        else:
            corr_dolar = float(df_corr['ticker'].corr(ret_usdars.reindex(idx_comun)))
        with cc3:
            cv = corr_dolar
            st.markdown(f"""<div class='corr-celda'>
                <div class='corr-val' style='color:{corr_color(cv)};'>{cv:+.2f}</div>
                <div class='corr-lbl'>vs USD/ARS</div>
                <div class='corr-desc' style='color:{corr_color(cv)};'>{corr_desc(cv, "tipo de cambio")}</div>
            </div>""", unsafe_allow_html=True)
    else:
        # Para USA: correlación vs QQQ
        if 'qqq' in df.columns:
            ret_qqq = df['qqq'].pct_change().reindex(idx_comun)
            corr_qqq = float(df_corr['ticker'].rolling(60).corr(ret_qqq).iloc[-1])
            with cc3:
                cv = corr_qqq
                st.markdown(f"""<div class='corr-celda'>
                    <div class='corr-val' style='color:{corr_color(cv)};'>{cv:+.2f}</div>
                    <div class='corr-lbl'>vs NASDAQ QQQ</div>
                    <div class='corr-desc' style='color:{corr_color(cv)};'>{corr_desc(cv, "Nasdaq")}</div>
                </div>""", unsafe_allow_html=True)

    # Beta dinámico (volatilidad relativa)
    vol_ticker = float(df['volatility_21'].iloc[-1]) * np.sqrt(252) * 100
    vol_spy_m  = float(df['spy'].pct_change().rolling(21).std().iloc[-1]) * np.sqrt(252) * 100
    beta_din   = vol_ticker / vol_spy_m if vol_spy_m > 0 else 1.0
    beta_color = "#f43f5e" if beta_din > 1.5 else "#f59e0b" if beta_din > 1 else "#05d890"
    with cc4:
        st.markdown(f"""<div class='corr-celda'>
            <div class='corr-val' style='color:{beta_color};'>{beta_din:.2f}x</div>
            <div class='corr-lbl'>Beta dinámico</div>
            <div class='corr-desc' style='color:{beta_color};'>{"Alta volatilidad relativa" if beta_din > 1.5 else "Vol. moderada" if beta_din > 1 else "Baja volatilidad relativa"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico de correlaciones rolling
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=df_corr.index, y=corr_wti_roll,
        name='vs WTI', line=dict(color='#f59e0b', width=1.5)
    ))
    fig_corr.add_trace(go.Scatter(
        x=df_corr.index, y=corr_spy_roll,
        name='vs S&P500', line=dict(color='#60a5fa', width=1.5)
    ))
    fig_corr.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.08)", line_width=1)
    fig_corr.add_hline(y=0.5, line_dash="dot", line_color="rgba(5,216,144,0.3)", line_width=1)
    fig_corr.add_hline(y=-0.5, line_dash="dot", line_color="rgba(244,63,94,0.3)", line_width=1)
    fig_corr.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(5,216,144,0.03)", line_width=0)
    fig_corr.add_hrect(y0=-1.0, y1=-0.5, fillcolor="rgba(244,63,94,0.03)", line_width=0)
    fig_corr.update_layout(
        template='plotly_dark', paper_bgcolor='#060d1a', plot_bgcolor='#090f1e',
        height=240, showlegend=True, margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(range=[-1,1], gridcolor='#0d1626', tickformat='.1f'),
        xaxis=dict(gridcolor='#0d1626'),
        font=dict(family='JetBrains Mono, monospace', color='#3d5a80', size=10),
        legend=dict(bgcolor='rgba(6,13,26,0.9)', bordercolor='#162035', borderwidth=1, font=dict(size=9)),
        title=dict(text=f'Correlación Rolling 60d — {ticker}', font=dict(size=11, color='#3d5a80'), x=0)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#1a2a3a;
        letter-spacing:0.08em;text-align:center;padding:10px 0;'>
        VOLUME PROFILE: ÚLTIMOS 252 DÍAS · SMART MONEY: σ > 2.5 SOBRE MEDIA 20D · CORRELACIONES: ROLLING 60 DÍAS
    </div>""", unsafe_allow_html=True)
# ── PANEL DE HISTORIAL ────────────────────────────────────────────────────────
st.markdown("<hr class='separador'>", unsafe_allow_html=True)

@st.fragment
def panel_historial():
    st.markdown("<div class='etiqueta-seccion'>HISTORIAL DE PREDICCIONES  ·  SEGUIMIENTO DE ACIERTOS</div>", unsafe_allow_html=True)

    st.markdown("""<style>
    @keyframes fadeSlideOut {
        0%   { opacity:1; transform:translateX(0)   scaleY(1);   max-height:60px; }
        40%  { opacity:0; transform:translateX(40px) scaleY(0.5); background:rgba(240,75,106,0.18); }
        100% { opacity:0; transform:translateX(60px) scaleY(0);   max-height:0; margin:0; padding:0; }
    }
    @keyframes fadeIn {
        from { opacity:0; transform:translateY(-6px); }
        to   { opacity:1; transform:translateY(0); }
    }
    .fila-hist { animation: fadeIn 0.25s ease forwards; transition: background 0.2s; }
    .fila-hist:hover { background: rgba(58,82,112,0.18) !important; }
    .fila-borrando { animation: fadeSlideOut 0.45s ease forwards !important; pointer-events: none; }
    div[data-testid="stHorizontalBlock"] .stButton button { min-height: 32px !important; height: 32px !important; }
    .hdr { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#2a4060;
           letter-spacing:0.12em; text-transform:uppercase; padding:4px 0 8px 0; }
    .hcel { font-family:'JetBrains Mono',monospace; font-size:0.7rem; padding:7px 0; line-height:1.4; }
    .btn-del button {
        background: rgba(240,75,106,0.08) !important;
        border: 1px solid rgba(240,75,106,0.25) !important;
        color: #f04b6a !important; font-size: 0.75rem !important; transition: all 0.15s !important;
    }
    .btn-del button:hover { background: rgba(240,75,106,0.25) !important; border-color: #f04b6a !important; transform: scale(1.15) !important; }
    </style>""", unsafe_allow_html=True)

    if "hist_pagina"  not in st.session_state: st.session_state.hist_pagina  = 1
    if "eliminar_idx" not in st.session_state: st.session_state.eliminar_idx = None

    _df_hist = cargar_historial()
    _df_hist = actualizar_resultados_reales(_df_hist)

    if st.session_state.eliminar_idx is not None:
        _n = len(_df_hist)
        _real_idx = _n - 1 - st.session_state.eliminar_idx
        if 0 <= _real_idx < _n:
            _df_hist = _df_hist.drop(index=_real_idx).reset_index(drop=True)
            _df_hist.to_csv(HISTORIAL_CSV, index=False)
        st.session_state.eliminar_idx = None
        _POR_P_tmp = 10
        _nt = max(1, (len(_df_hist) + _POR_P_tmp - 1) // _POR_P_tmp)
        if st.session_state.hist_pagina > _nt: st.session_state.hist_pagina = _nt
        st.rerun(scope="fragment")

    if _df_hist.empty:
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#3d5a80;padding:20px;text-align:center;'>No hay predicciones guardadas aún.</div>", unsafe_allow_html=True)
        return

    _df_con_resultado = _df_hist[_df_hist["acierto"].isin(["✅","❌"])]
    _total_pred   = len(_df_hist)
    _con_res      = len(_df_con_resultado)
    _aciertos     = (_df_con_resultado["acierto"] == "✅").sum()
    _tasa_acierto = f"{_aciertos/_con_res*100:.1f}%" if _con_res > 0 else "—"
    _error_prom   = _df_con_resultado["error_real_usd"].apply(pd.to_numeric, errors="coerce").mean()
    _error_prom_s = f"${_error_prom:.4f}" if pd.notna(_error_prom) else "—"

    hc1,hc2,hc3,hc4 = st.columns(4)
    with hc1: st.markdown(f"<div class='hist-stat'><div class='hist-stat-val'>{_total_pred}</div><div class='hist-stat-lbl'>Total predicciones</div></div>", unsafe_allow_html=True)
    with hc2: st.markdown(f"<div class='hist-stat'><div class='hist-stat-val'>{_con_res}</div><div class='hist-stat-lbl'>Con resultado real</div></div>", unsafe_allow_html=True)
    _color_tasa = 'var(--verde)' if _con_res > 0 and _aciertos / _con_res >= 0.5 else 'var(--rojo)'
    with hc3: st.markdown(f"<div class='hist-stat'><div class='hist-stat-val' style='color:{_color_tasa};'>{_tasa_acierto}</div><div class='hist-stat-lbl'>Tasa de acierto</div></div>", unsafe_allow_html=True)
    with hc4: st.markdown(f"<div class='hist-stat'><div class='hist-stat-val'>{_error_prom_s}</div><div class='hist-stat-lbl'>Error promedio real</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    _POR_PAGINA = 10
    _df_inv     = _df_hist.iloc[::-1].reset_index(drop=True)
    _total_pags = max(1, (len(_df_inv) + _POR_PAGINA - 1) // _POR_PAGINA)
    _pag        = min(st.session_state.hist_pagina, _total_pags)
    _df_page    = _df_inv.iloc[(_pag-1)*_POR_PAGINA : _pag*_POR_PAGINA]

    C = [1.9, 0.85, 0.7, 1.05, 1.05, 0.95, 1.55, 1.05, 1.0, 0.45]
    H = ["FECHA","TICKER","MKT","ACTUAL","OBJETIVO","VAR%","RANGO","PRECIO REAL","ACIERTO",""]
    cols = st.columns(C)
    for col, h in zip(cols, H):
        col.markdown(f"<div class='hdr'>{h}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#0d1626;margin-bottom:4px;'></div>", unsafe_allow_html=True)

    for i, (_, row) in enumerate(_df_page.iterrows()):
        _inv_pos = (_pag-1)*_POR_PAGINA + i
        _var_pct = float(row["variacion_pct"]) if str(row["variacion_pct"]) not in ["","nan"] else 0
        _var_col = "#05d890" if _var_pct > 0 else "#f04b6a"
        _mkt_col = "#c9a84c" if str(row["mercado"]) == "ARG" else "#3b82f6"
        _pr_s    = f"${float(row['precio_real_siguiente']):.2f}" if str(row['precio_real_siguiente']) not in ["","nan","None"] else "—"
        _rg_s    = f"${float(row['rango_min']):.2f}–{float(row['rango_max']):.2f}" if str(row['rango_min']) not in ["","nan","None"] else "—"
        _fe_s    = str(row['fecha'])[:16]
        if   row["acierto"] == "✅": _ac_s, _ac_c = "✅", "#05d890"
        elif row["acierto"] == "❌": _ac_s, _ac_c = "❌", "#f04b6a"
        else:                         _ac_s, _ac_c = "pendiente", "#3d5a80"

        _bg = "rgba(240,75,106,0.10)" if st.session_state.get("eliminar_idx")==_inv_pos else ("rgba(12,22,42,0.6)" if i%2==0 else "rgba(8,14,28,0.4)")
        _clase = "fila-borrando" if st.session_state.get("eliminar_idx")==_inv_pos else "fila-hist"

        st.markdown(f"<div class='{_clase}' style='background:{_bg};border-radius:2px;margin-bottom:1px;'>", unsafe_allow_html=True)
        cols = st.columns(C)
        cols[0].markdown(f"<div class='hcel' style='color:#4a6080;'>{_fe_s}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div class='hcel' style='color:#7cb9e8;font-weight:700;'>{row['ticker']}</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div class='hcel' style='color:{_mkt_col};font-weight:600;'>{row['mercado']}</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div class='hcel' style='color:#c8d8e8;'>${float(row['precio_actual']):.2f}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div class='hcel' style='color:#c8d8e8;'>${float(row['precio_objetivo']):.2f}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div class='hcel' style='color:{_var_col};font-weight:700;'>{_var_pct:+.2f}%</div>", unsafe_allow_html=True)
        cols[6].markdown(f"<div class='hcel' style='color:#3a5270;font-size:0.64rem;'>{_rg_s}</div>", unsafe_allow_html=True)
        cols[7].markdown(f"<div class='hcel' style='color:#c8d8e8;'>{_pr_s}</div>", unsafe_allow_html=True)
        cols[8].markdown(f"<div class='hcel' style='color:{_ac_c};'>{_ac_s}</div>", unsafe_allow_html=True)
        with cols[9]:
            st.markdown("<div class='btn-del'>", unsafe_allow_html=True)
            if st.button("✕", key=f"d_{_inv_pos}", help="Eliminar esta predicción"):
                st.session_state.eliminar_idx = _inv_pos
                st.rerun(scope="fragment")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px;background:#080e1a;'></div>", unsafe_allow_html=True)

    if _total_pags > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#2a4060;margin-bottom:6px;'>PÁGINA {_pag} DE {_total_pags}</div>", unsafe_allow_html=True)
        pag_cols = st.columns([1] * _total_pags)
        for p in range(1, _total_pags+1):
            with pag_cols[p-1]:
                if st.button(f"**{p}**" if p==_pag else str(p), key=f"pag_{p}"):
                    st.session_state.hist_pagina = p
                    st.rerun(scope="fragment")

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇  Descargar historial completo (CSV)",
        data=_df_hist.to_csv(index=False).encode("utf-8"),
        file_name=f"historial_djt_capital_{datetime.date.today()}.csv",
        mime="text/csv",
    )

panel_historial()

