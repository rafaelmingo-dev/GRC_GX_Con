# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# O GEX é carregado primeiro. GARCH/ARCH e Plotly entram somente depois do pico de memória do pipeline B3.
import gex_core as gex

st.set_page_config(
    page_title="GARCH × GEX — Confluência V3",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1700px;}
    header[data-testid="stHeader"] {height: 0rem;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    .v3-sub {font-size:.94rem; opacity:.78; margin-top:-.35rem; margin-bottom:.8rem;}
    .v3-note {padding:.65rem .8rem; border:1px solid rgba(128,128,128,.25); border-radius:.55rem;}
    .metric-label {font-size:.82rem; opacity:.75;}
</style>
""",
    unsafe_allow_html=True,
)

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if "ativo_detalhe" not in st.session_state:
    st.session_state.ativo_detalhe = None


@st.cache_resource(show_spinner=False)
def preparar_runtime_gex(refresh_token: int):
    """Carrega o GEX uma única vez sem manter uma segunda cópia grande no cache.

    O load_complete_bundle devolve DataFrames grandes e initialize_runtime instala
    cópias no módulo GEX. Se o bundle inteiro também ficar no cache, o Cloud pode
    manter as duas cópias simultaneamente. Aqui o cache guarda somente metadata.
    """
    # Em atualização forçada, libera o runtime anterior antes do novo pipeline.
    if refresh_token > 0:
        try:
            gex.gex_series = pd.DataFrame()
            gex.historical_prices = pd.DataFrame()
            gex.metadata = {}
        except Exception:
            pass
        gc.collect()

    series, metadata, history = gex.load_complete_bundle(force=refresh_token > 0)
    gex.initialize_runtime(series, metadata, history)

    # O runtime já possui os dados necessários; as variáveis locais não precisam
    # permanecer retidas pelo cache.
    del series
    del history
    gc.collect()

    return dict(metadata)


@st.cache_data(show_spinner=False, ttl=1800)
def calcular_garch_cacheado(ativo: str, data_ref_iso: str, refresh_token: int):
    data_ref = pd.Timestamp(data_ref_iso)
    try:
        return core.calcular_garch_ativo(ativo, data_ref)
    except Exception as exc:
        return {
            "ok": False,
            "preco": np.nan,
            "intervalo": None,
            "momento": pd.NaT,
            "bandas": {"MENSAL": None, "SEMESTRAL": None, "ANUAL": None},
            "erros": {"GERAL": f"{type(exc).__name__}: {exc}"},
        }


def fmt_num(v, casas=2, vazio="—"):
    try:
        x = float(v)
        if not np.isfinite(x):
            return vazio
        return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return vazio


def fmt_pct(v, casas=2, vazio="—"):
    t = fmt_num(v, casas, vazio)
    return t if t == vazio else f"{t}%"


def fmt_gamma(v):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
        a = abs(x)
        if a >= 1_000_000_000:
            return f"{x/1_000_000_000:.2f} bi"
        if a >= 1_000_000:
            return f"{x/1_000_000:.2f} mi"
        if a >= 1_000:
            return f"{x/1_000:.2f} mil"
        return f"{x:.2f}"
    except Exception:
        return "—"


def dados_item(item):
    if not item:
        return {
            "par": "SEM DADOS",
            "dif": np.nan,
            "preco_zona": np.nan,
            "qualidade": "N/D",
            "wall": "—",
            "banda": "—",
        }
    return {
        "par": f"{item['Banda GARCH']} × {item['Wall/Região GEX']}",
        "dif": item["Diferença Wall↔Banda %"],
        "preco_zona": item["Dist Spot→Zona %"],
        "qualidade": item["Classe qualidade"],
        "wall": item["Wall/Região GEX"],
        "banda": item["Banda GARCH"],
    }


def tabela_principal(resultados):
    df = core.dataframe_radar(resultados)
    if df.empty:
        return df
    cols = [
        "Ativo", "Empresa", "Preço", "Spot GEX",
        "30D · Principal", "30D · Confluência %", "30D · Preço→Confluência %", "30D · Qualidade",
        "90D · Principal", "90D · Confluência %", "90D · Preço→Confluência %", "90D · Qualidade",
        "180D · Principal", "180D · Confluência %", "180D · Preço→Confluência %", "180D · Qualidade",
        "Anual · Banda", "Anual · Dist %", "Anual · Status",
    ]
    return df[[c for c in cols if c in df.columns]]


def tabela_secundaria(resultados):
    df = core.dataframe_radar(resultados)
    if df.empty:
        return df
    cols = [
        "Ativo", "Empresa", "Preço", "Spot GEX",
        "30D · Secundária", "30D · Sec %",
        "90D · Secundária", "90D · Sec %",
        "180D · Secundária", "180D · Sec %",
    ]
    return df[[c for c in cols if c in df.columns]]


def dataframe_display(df):
    formatos = {}
    for col in df.columns:
        if "Preço" in col or col == "Spot GEX":
            formatos[col] = st.column_config.NumberColumn(format="%.2f")
        elif "%" in col or "Dist" in col:
            formatos[col] = st.column_config.NumberColumn(format="%.2f%%")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=formatos,
        height=min(820, 38 * (len(df) + 1)),
    )


def grafico_niveis(ativo_res, bloco_nome):
    bloco = ativo_res["blocos"][bloco_nome]
    spot = core.numero_seguro(ativo_res.get("Spot GEX"))
    preco_garch = core.numero_seguro(ativo_res.get("Preço GARCH"))
    bandas = bloco.get("bandas") or {}
    p = bloco.get("principal")
    s = bloco.get("secundaria")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=["Preço"],
            y=[spot],
            mode="markers+text",
            text=[f"Spot GEX {fmt_num(spot)}"],
            textposition="middle right",
            name="Spot GEX",
            marker=dict(size=13),
        )
    )
    if np.isfinite(preco_garch):
        fig.add_trace(
            go.Scatter(
                x=["Preço GARCH"],
                y=[preco_garch],
                mode="markers+text",
                text=[f"Preço GARCH {fmt_num(preco_garch)}"],
                textposition="middle right",
                name="Preço GARCH",
                marker=dict(size=11, symbol="diamond"),
            )
        )

    for rotulo, chave in core.BANDAS_COMPARADAS:
        nivel = core.numero_seguro(bandas.get(chave))
        if np.isfinite(nivel):
            fig.add_hline(y=nivel, line_dash="dot", annotation_text=f"GARCH {rotulo}: {fmt_num(nivel)}")

    if p:
        nivel = core.numero_seguro(p.get("Nível Wall/Região"))
        if np.isfinite(nivel):
            fig.add_hline(y=nivel, line_width=3, annotation_text=f"Principal {p['Wall/Região GEX']}: {fmt_num(nivel)}")
    if s:
        nivel = core.numero_seguro(s.get("Nível Wall/Região"))
        if np.isfinite(nivel):
            fig.add_hline(y=nivel, line_dash="dash", annotation_text=f"Secundária {s['Wall/Região GEX']}: {fmt_num(nivel)}")

    fig.update_layout(
        title=f"{ativo_res['Ativo']} — {bloco_nome}",
        xaxis_title="Referências",
        yaxis_title="Preço",
        height=560,
        margin=dict(l=40, r=40, t=70, b=40),
        showlegend=True,
    )
    return fig


def render_item(titulo, item):
    st.markdown(f"#### {titulo}")
    if not item:
        st.info("SEM DADOS para este bloco.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confluência GARCH × GEX", fmt_pct(item["Diferença Wall↔Banda %"], 3))
    c2.metric("Distância do preço à confluência", fmt_pct(item["Dist Spot→Zona %"], 3))
    c3.metric("Qualidade GEX", f"{fmt_num(item['Qualidade GEX'],1)} · {item['Classe qualidade']}")
    c4.metric("Wall / Região", item["Wall/Região GEX"])

    st.markdown(
        f"""
<div class="v3-note">
<b>{item['Banda GARCH']}</b> em <b>{fmt_num(item['Nível banda'])}</b>
&nbsp; × &nbsp;
<b>{item['Wall/Região GEX']}</b> em <b>{fmt_num(item['Nível Wall/Região'])}</b><br>
Zona: <b>{fmt_num(item['Zona inferior'])}</b> a <b>{fmt_num(item['Zona superior'])}</b>
&nbsp; | &nbsp; Centro: <b>{fmt_num(item['Centro da zona'])}</b>
&nbsp; | &nbsp; {item['Posição da zona vs Spot']}
</div>
""",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Participação Call", fmt_pct(item.get("Participação Call %")))
    d2.metric("Participação Put", fmt_pct(item.get("Participação Put %")))
    d3.metric("Gross Gamma Call", fmt_gamma(item.get("Gross Gamma Call")))
    d4.metric("Gross Gamma Put", fmt_gamma(item.get("Gross Gamma Put")))


def gerar_zip_csv(resultados):
    principal = core.dataframe_detalhes(resultados, "PRINCIPAL W1")
    secundaria = core.dataframe_detalhes(resultados, "SECUNDÁRIA W2/W3")
    todas = core.dataframe_todas_comparacoes(resultados)
    radar = core.dataframe_radar(resultados)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("radar_v3.csv", radar.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("principal_w1.csv", principal.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("secundaria_w2_w3.csv", secundaria.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("todas_comparacoes.csv", todas.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


# ======================================================================================
# CABEÇALHO E ATUALIZAÇÃO
# ======================================================================================

col_title, col_btn = st.columns([7, 1])
with col_title:
    st.title("GARCH × GEX — CONFLUÊNCIA V3")
    st.markdown(
        '<div class="v3-sub">Principal W1 • Secundária W2/W3 • Mensal×30D • Semestral×90D • Semestral×180D • GARCH Anual</div>',
        unsafe_allow_html=True,
    )
with col_btn:
    if st.button("↻ Atualizar", use_container_width=True):
        st.session_state.refresh_token += 1
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

token = int(st.session_state.refresh_token)

try:
    with st.spinner("Buscando a última sessão completa da B3 e preparando o motor GEX..."):
        metadata = preparar_runtime_gex(token)
        gex_reference_date = pd.Timestamp(metadata["reference_date"])
except Exception as exc:
    st.error(f"Falha ao carregar o GEX: {type(exc).__name__}: {exc}")
    st.stop()

# Imports pesados somente DEPOIS do pipeline GEX/B3.
# Isso reduz o pico de memória no Streamlit Community Cloud sem alterar a matemática.
import garch_core as garch
import confluence_core as core
import plotly.graph_objects as go

gc.collect()

ativos_comuns = [
    codigo for codigo in garch.ATIVOS.keys()
    if codigo != "BTC-USD" and codigo in set(gex.ASSETS)
]

data_ref = garch.agora_local().normalize()
garch_resultados = {}
resultados = {}

progress = st.progress(0, text="Calculando GARCH e cruzando com GEX...")
for i, ativo in enumerate(ativos_comuns, start=1):
    gd = calcular_garch_cacheado(ativo, data_ref.isoformat(), token)
    garch_resultados[ativo] = gd
    if gd.get("ok"):
        try:
            resultados[ativo] = core.calcular_confluencia_ativo(
                ativo, gd, gex_reference_date
            )
        except Exception as exc:
            st.warning(f"{ativo}: falha no cruzamento — {type(exc).__name__}: {exc}")
    progress.progress(i / len(ativos_comuns), text=f"{ativo} — {i}/{len(ativos_comuns)}")
    gc.collect()
progress.empty()

# BTC: somente contexto anual do GARCH
btc = None
if "BTC-USD" in garch.ATIVOS:
    btc_gd = calcular_garch_cacheado("BTC-USD", data_ref.isoformat(), token)
    if btc_gd.get("ok"):
        btc = {
            "preco": btc_gd.get("preco"),
            "anual": core.resultado_anual_garch(
                btc_gd.get("preco"), btc_gd.get("bandas", {}).get("ANUAL")
            ),
        }

st.caption(
    f"Base GEX: fechamento B3 {gex_reference_date.strftime('%d/%m/%Y')} • "
    f"GARCH: preço mais recente disponível ao executar o painel. "
    f"Confluência = distância entre Banda GARCH e Wall/Região GEX."
)

if not resultados:
    st.error("Nenhum ativo pôde ser calculado.")
    st.stop()

# ======================================================================================
# ABAS
# ======================================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Radar Principal W1", "Secundária W2/W3", "Detalhes por ativo", "Metodologia"]
)

with tab1:
    st.subheader("Confluência Principal — W1")
    st.caption(
        "Ordenação relativa: menor confluência principal disponível primeiro. "
        "Não há classificação Forte/Moderada nesta versão."
    )
    dataframe_display(tabela_principal(resultados))

    st.download_button(
        "Baixar CSVs do painel V3",
        data=gerar_zip_csv(resultados),
        file_name="garch_gex_painel_v3_dados.zip",
        mime="application/zip",
    )

with tab2:
    st.subheader("Confluência Secundária — W2/W3")
    st.caption("W2/W3 permanecem separadas da W1 e não substituem a confluência principal.")
    dataframe_display(tabela_secundaria(resultados))

    sec = core.dataframe_detalhes(resultados, "SECUNDÁRIA W2/W3")
    if not sec.empty:
        cols = [
            "Ativo", "Bloco", "Banda GARCH", "Nível banda", "Wall/Região GEX",
            "Rank Wall", "Diferença Wall↔Banda %", "Dist Spot→Zona %",
            "Participação Wall %", "Participação Call %", "Participação Put %",
            "Gross Gamma Wall", "Gross Gamma Call", "Gross Gamma Put",
            "Qualidade GEX", "Classe qualidade", "Séries GEX", "Vencimentos GEX",
        ]
        dataframe_display(sec[[c for c in cols if c in sec.columns]])

with tab3:
    st.subheader("Detalhes por ativo")
    ativos_lista = list(resultados.keys())
    default_idx = 0
    if st.session_state.ativo_detalhe in ativos_lista:
        default_idx = ativos_lista.index(st.session_state.ativo_detalhe)

    ativo = st.selectbox("Ativo", ativos_lista, index=default_idx)
    st.session_state.ativo_detalhe = ativo
    ativo_res = resultados[ativo]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço GARCH", fmt_num(ativo_res["Preço GARCH"]))
    c2.metric("Spot GEX", fmt_num(ativo_res["Spot GEX"]))
    c3.metric("Dif. preço GARCH × Spot", fmt_pct(ativo_res["Preço GARCH × Spot GEX · Dif %"]))
    c4.metric("Base GEX", pd.Timestamp(ativo_res["Data efetiva GEX"]).strftime("%d/%m/%Y"))

    bloco_nome = st.radio(
        "Bloco",
        ["Mensal × 30D", "Semestral × 90D", "Semestral × 180D"],
        horizontal=True,
    )
    bloco = ativo_res["blocos"][bloco_nome]

    render_item("Principal W1", bloco.get("principal"))
    st.markdown("---")
    render_item("Secundária W2/W3", bloco.get("secundaria"))

    st.plotly_chart(grafico_niveis(ativo_res, bloco_nome), use_container_width=True)

    todas = pd.DataFrame(
        bloco.get("comparacoes_principal", []) + bloco.get("comparacoes_secundaria", [])
    )
    if not todas.empty:
        st.markdown("#### Todas as combinações deste bloco")
        cols = [
            "Camada", "Banda GARCH", "Nível banda", "Wall/Região GEX", "Rank Wall",
            "Nível Wall/Região", "Diferença Wall↔Banda %", "Dist Spot→Zona %",
            "Dist Spot→Centro %", "Posição da zona vs Spot",
            "Participação Wall %", "Participação Call %", "Participação Put %",
            "Gross Gamma Wall", "Gross Gamma Call", "Gross Gamma Put",
            "Qualidade GEX", "Classe qualidade", "Séries GEX", "Vencimentos GEX",
        ]
        dataframe_display(todas[[c for c in cols if c in todas.columns]])

    anual = ativo_res.get("anual")
    st.markdown("#### GARCH Anual — sem contraparte GEX")
    if anual:
        a1, a2, a3 = st.columns(3)
        a1.metric("Banda", anual["rotulo"])
        a2.metric("Nível", fmt_num(anual["nivel"]))
        a3.metric("Distância", fmt_pct(anual["dist_pct"]))
        st.caption(anual["status"])
    else:
        st.info("SEM DADOS no GARCH Anual.")

with tab4:
    st.subheader("Metodologia preservada da V3")
    st.markdown(
        """
- **Principal:** somente Call/Put W1.
- **Secundária:** somente W2/W3.
- **Mensal GARCH × GEX 30D**.
- **Semestral GARCH × GEX 90D**.
- **Semestral GARCH × GEX 180D**.
- **GARCH Anual:** permanece sozinho.
- **GEX 60D:** não participa deste painel conjunto.
- **Confluência GARCH × GEX (%):** `|Wall/Região GEX − Banda GARCH| / Spot GEX × 100`.
- **Distância do preço à confluência:** distância do Spot GEX até o intervalo entre a Banda e a Wall/Região. Se o spot estiver dentro do intervalo, é zero.
- Call/Put do mesmo rank só são agrupadas usando a tolerância original do GEX.
- Participações e Gross Gamma de Call/Put compartilhadas **não são somados artificialmente**.
- Não há score composto, nem limiar Forte/Moderada/Fraca, nem sinal de compra/venda.
"""
    )

    st.info(
        "A primeira métrica mede quão próximos GARCH e GEX estão. "
        "A distância do preço à confluência mede apenas onde o mercado está em relação à região."
    )

    if btc and btc.get("anual"):
        st.markdown("#### Bitcoin — somente GARCH Anual")
        st.write(
            f"Preço: {fmt_num(btc['preco'])} • "
            f"Banda: {btc['anual']['rotulo']} • "
            f"Distância: {fmt_pct(btc['anual']['dist_pct'])} • "
            f"{btc['anual']['status']}"
