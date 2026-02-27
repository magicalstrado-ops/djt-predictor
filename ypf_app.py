
# ── MÓDULO DE INTELIGENCIA AVANZADA ──────────────────────────
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
