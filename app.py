# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import hashlib
import io
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CACHE_SCHEMA = 3
MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = MODULE_DIR / ".panel_cache"
CACHE_FILE = CACHE_DIR / "painel_v3.pkl"
WORKER_FILE = MODULE_DIR / "panel_worker.py"

BANDAS_COMPARADAS = [
    ("-2σ", "menos_2"),
    ("-1,5σ", "menos_15"),
    ("+1,5σ", "mais_15"),
    ("+2σ", "mais_2"),
]

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
    .v3-status {padding:.70rem .85rem; border:1px solid rgba(128,128,128,.25); border-radius:.55rem; margin:.25rem 0 .8rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

if "ativo_detalhe" not in st.session_state:
    st.session_state.ativo_detalhe = None


# ======================================================================================
# CACHE / WORKER
# ======================================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_core_hashes() -> dict[str, str]:
    return {
        nome: sha256_file(MODULE_DIR / nome)
        for nome in ("gex_core.py", "garch_core.py", "confluence_core.py")
    }


def carregar_cache():
    if not CACHE_FILE.exists():
        return None, "CACHE_AUSENTE"

    try:
        with CACHE_FILE.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        return None, f"CACHE_ILEGÍVEL: {type(exc).__name__}: {exc}"

    if not isinstance(payload, dict):
        return None, "CACHE_INVÁLIDO"

    if payload.get("cache_schema") != CACHE_SCHEMA:
        return None, "CACHE_INCOMPATÍVEL_COM_ESTA_VERSÃO"

    try:
        hashes_atuais = current_core_hashes()
    except Exception as exc:
        return None, f"ERRO_AO_VALIDAR_CORES: {type(exc).__name__}: {exc}"

    if payload.get("core_hashes") != hashes_atuais:
        return None, "CACHE_DE_OUTRA_VERSÃO_DOS_MOTORES"

    resultados = payload.get("resultados")
    if not isinstance(resultados, dict) or not resultados:
        return None, "CACHE_SEM_RESULTADOS"

    return payload, None


def ultimas_linhas(linhas, n=16):
    return "\n".join(linhas[-n:])


def executar_worker(force_gex: bool):
    """Executa B3/GEX/GARCH em processo separado e preserva o cache anterior em falha."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(WORKER_FILE),
        "--output",
        str(CACHE_FILE),
    ]

    if force_gex:
        cmd.append("--force-gex")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
        }
    )

    status = st.status(
        "Atualizando B3, GEX, GARCH e confluências...",
        expanded=True,
    )

    log_box = st.empty()
    linhas = []

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(MODULE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        status.update(label="Falha ao iniciar atualização.", state="error")
        return False, f"{type(exc).__name__}: {exc}"

    assert process.stdout is not None

    for raw in process.stdout:
        line = raw.rstrip()
        if not line:
            continue
        linhas.append(line)
        log_box.code(
            ultimas_linhas(linhas),
            language="text",
        )

    return_code = process.wait()

    if return_code != 0:
        status.update(
            label=f"Atualização falhou (worker retornou {return_code}).",
            state="error",
        )
        return False, ultimas_linhas(linhas, 30)

    status.update(
        label="Atualização concluída. Carregando o painel...",
        state="complete",
    )
    return True, ultimas_linhas(linhas, 30)


# ======================================================================================
# FORMATAÇÃO
# ======================================================================================

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


# ======================================================================================
# CABEÇALHO
# ======================================================================================

col_title, col_btn = st.columns([7, 1])

with col_title:
    st.title("GARCH × GEX — CONFLUÊNCIA V3")
    st.markdown(
        '<div class="v3-sub">Principal W1 • Secundária W2/W3 • Mensal×30D • Semestral×90D • Semestral×180D • GARCH Anual</div>',
        unsafe_allow_html=True,
    )

payload, cache_error = carregar_cache()

with col_btn:
    atualizar = (
        st.button("↻ Atualizar", use_container_width=True)
        if payload is not None
        else False
    )

# O app abre sem disparar o pipeline pesado.
if payload is None:
    st.warning(
        "O painel ainda não possui um cache calculado compatível nesta instância. "
        "A página abriu sem processar a B3 para evitar o travamento do Streamlit Cloud."
    )

    if cache_error not in (None, "CACHE_AUSENTE"):
        st.caption(f"Estado do cache: {cache_error}")

    st.markdown(
        """
<div class="v3-status">
<b>Primeira execução:</b> clique em <b>Preparar painel agora</b>.
O cálculo pesado será feito em um processo separado. O resultado final só substitui
o cache depois que todas as etapas terminarem.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("▶ Preparar painel agora", type="primary"):
        ok, mensagem = executar_worker(force_gex=False)

        if ok:
            st.rerun()
        else:
            st.error(
                "A preparação não terminou. O processo principal do Streamlit permaneceu ativo."
            )
            if mensagem:
                st.code(mensagem, language="text")

    st.stop()

if atualizar:
    ok, mensagem = executar_worker(force_gex=True)

    if ok:
        st.rerun()
    else:
        st.error(
            "A atualização falhou, mas o último cache válido foi preservado. "
            "O painel abaixo continua utilizando a base anterior."
        )
        if mensagem:
            st.code(mensagem, language="text")


# ======================================================================================
# RESULTADOS PRONTOS
# ======================================================================================

resultados = payload["resultados"]
btc = payload.get("btc")
erros_worker = payload.get("erros", {})

# Importados somente depois que o cache final existe.
import confluence_core as core
import plotly.graph_objects as go

gc.collect()


# ======================================================================================
# TABELAS / GRÁFICOS / DETALHES
# ======================================================================================

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
            x=["Spot GEX"],
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

    for rotulo, chave in BANDAS_COMPARADAS:
        nivel = core.numero_seguro(bandas.get(chave))
        if np.isfinite(nivel):
            fig.add_hline(
                y=nivel,
                line_dash="dot",
                annotation_text=f"GARCH {rotulo}: {fmt_num(nivel)}",
            )

    if p:
        nivel = core.numero_seguro(p.get("Nível Wall/Região"))
        if np.isfinite(nivel):
            fig.add_hline(
                y=nivel,
                line_width=3,
                annotation_text=f"Principal {p['Wall/Região GEX']}: {fmt_num(nivel)}",
            )

    if s:
        nivel = core.numero_seguro(s.get("Nível Wall/Região"))
        if np.isfinite(nivel):
            fig.add_hline(
                y=nivel,
                line_dash="dash",
                annotation_text=f"Secundária {s['Wall/Região GEX']}: {fmt_num(nivel)}",
            )

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

    c1.metric(
        "Confluência GARCH × GEX",
        fmt_pct(item["Diferença Wall↔Banda %"], 3),
    )

    c2.metric(
        "Distância do preço à confluência",
        fmt_pct(item["Dist Spot→Zona %"], 3),
    )

    c3.metric(
        "Qualidade GEX",
        f"{fmt_num(item['Qualidade GEX'],1)} · {item['Classe qualidade']}",
    )

    c4.metric(
        "Wall / Região",
        item["Wall/Região GEX"],
    )

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

    d1.metric(
        "Participação Call",
        fmt_pct(item.get("Participação Call %")),
    )

    d2.metric(
        "Participação Put",
        fmt_pct(item.get("Participação Put %")),
    )

    d3.metric(
        "Gross Gamma Call",
        fmt_gamma(item.get("Gross Gamma Call")),
    )

    d4.metric(
        "Gross Gamma Put",
        fmt_gamma(item.get("Gross Gamma Put")),
    )


def gerar_zip_csv(resultados):
    principal = core.dataframe_detalhes(
        resultados,
        "PRINCIPAL W1",
    )

    secundaria = core.dataframe_detalhes(
        resultados,
        "SECUNDÁRIA W2/W3",
    )

    todas = core.dataframe_todas_comparacoes(
        resultados,
    )

    radar = core.dataframe_radar(
        resultados,
    )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr(
            "radar_v3.csv",
            radar.to_csv(index=False).encode("utf-8-sig"),
        )

        z.writestr(
            "principal_w1.csv",
            principal.to_csv(index=False).encode("utf-8-sig"),
        )

        z.writestr(
            "secundaria_w2_w3.csv",
            secundaria.to_csv(index=False).encode("utf-8-sig"),
        )

        z.writestr(
            "todas_comparacoes.csv",
            todas.to_csv(index=False).encode("utf-8-sig"),
        )

    return buffer.getvalue()


# ======================================================================================
# STATUS DA BASE
# ======================================================================================

gex_date = pd.Timestamp(payload["gex_reference_date"])
generated_at = pd.Timestamp(payload["generated_at"])

st.caption(
    f"Base GEX: fechamento B3 {gex_date.strftime('%d/%m/%Y')} • "
    f"Painel calculado em {generated_at.strftime('%d/%m/%Y %H:%M')} • "
    f"Confluência = distância entre Banda GARCH e Wall/Região GEX."
)

if erros_worker:
    st.warning(
        f"{len(erros_worker)} ativo(s) tiveram erro parcial na última atualização. "
        "Os demais resultados válidos foram preservados."
    )


# ======================================================================================
# ABAS
# ======================================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Radar Principal W1",
        "Secundária W2/W3",
        "Detalhes por ativo",
        "Metodologia",
    ]
)


with tab1:
    st.subheader("Confluência Principal — W1")

    st.caption(
        "Ordenação relativa: menor confluência principal disponível primeiro. "
        "Ainda não há classificação Forte/Moderada/Fraca."
    )

    dataframe_display(
        tabela_principal(resultados)
    )

    st.download_button(
        "Baixar CSVs do painel V3",
        data=gerar_zip_csv(resultados),
        file_name="garch_gex_painel_v3_dados.zip",
        mime="application/zip",
    )


with tab2:
    st.subheader("Confluência Secundária — W2/W3")

    st.caption(
        "W2/W3 permanecem separadas da W1 e não substituem a confluência principal."
    )

    dataframe_display(
        tabela_secundaria(resultados)
    )

    sec = core.dataframe_detalhes(
        resultados,
        "SECUNDÁRIA W2/W3",
    )

    if not sec.empty:
        cols = [
            "Ativo",
            "Bloco",
            "Banda GARCH",
            "Nível banda",
            "Wall/Região GEX",
            "Rank Wall",
            "Diferença Wall↔Banda %",
            "Dist Spot→Zona %",
            "Participação Wall %",
            "Participação Call %",
            "Participação Put %",
            "Gross Gamma Wall",
            "Gross Gamma Call",
            "Gross Gamma Put",
            "Qualidade GEX",
            "Classe qualidade",
            "Séries GEX",
            "Vencimentos GEX",
        ]

        dataframe_display(
            sec[[c for c in cols if c in sec.columns]]
        )


with tab3:
    st.subheader("Detalhes por ativo")

    ativos_lista = list(resultados.keys())
    default_idx = 0

    if st.session_state.ativo_detalhe in ativos_lista:
        default_idx = ativos_lista.index(
            st.session_state.ativo_detalhe
        )

    ativo = st.selectbox(
        "Ativo",
        ativos_lista,
        index=default_idx,
    )

    st.session_state.ativo_detalhe = ativo
    ativo_res = resultados[ativo]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Preço GARCH",
        fmt_num(ativo_res["Preço GARCH"]),
    )

    c2.metric(
        "Spot GEX",
        fmt_num(ativo_res["Spot GEX"]),
    )

    c3.metric(
        "Dif. preço GARCH × Spot",
        fmt_pct(
            ativo_res["Preço GARCH × Spot GEX · Dif %"]
        ),
    )

    c4.metric(
        "Base GEX",
        pd.Timestamp(
            ativo_res["Data efetiva GEX"]
        ).strftime("%d/%m/%Y"),
    )

    bloco_nome = st.radio(
        "Bloco",
        [
            "Mensal × 30D",
            "Semestral × 90D",
            "Semestral × 180D",
        ],
        horizontal=True,
    )

    bloco = ativo_res["blocos"][bloco_nome]

    render_item(
        "Principal W1",
        bloco.get("principal"),
    )

    st.markdown("---")

    render_item(
        "Secundária W2/W3",
        bloco.get("secundaria"),
    )

    st.plotly_chart(
        grafico_niveis(
            ativo_res,
            bloco_nome,
        ),
        use_container_width=True,
    )

    todas = pd.DataFrame(
        bloco.get(
            "comparacoes_principal",
            [],
        )
        + bloco.get(
            "comparacoes_secundaria",
            [],
        )
    )

    if not todas.empty:
        st.markdown(
            "#### Todas as combinações deste bloco"
        )

        cols = [
            "Camada",
            "Banda GARCH",
            "Nível banda",
            "Wall/Região GEX",
            "Rank Wall",
            "Nível Wall/Região",
            "Diferença Wall↔Banda %",
            "Dist Spot→Zona %",
            "Dist Spot→Centro %",
            "Posição da zona vs Spot",
            "Participação Wall %",
            "Participação Call %",
            "Participação Put %",
            "Gross Gamma Wall",
            "Gross Gamma Call",
            "Gross Gamma Put",
            "Qualidade GEX",
            "Classe qualidade",
            "Séries GEX",
            "Vencimentos GEX",
        ]

        dataframe_display(
            todas[
                [c for c in cols if c in todas.columns]
            ]
        )

    anual = ativo_res.get("anual")

    st.markdown(
        "#### GARCH Anual — sem contraparte GEX"
    )

    if anual:
        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Banda",
            anual["rotulo"],
        )

        a2.metric(
            "Nível",
            fmt_num(anual["nivel"]),
        )

        a3.metric(
            "Distância",
            fmt_pct(anual["dist_pct"]),
        )

        st.caption(
            anual["status"]
        )

    else:
        st.info(
            "SEM DADOS no GARCH Anual."
        )


with tab4:
    st.subheader(
        "Metodologia preservada da V3"
    )

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
        "A Confluência GARCH × GEX mede quão próximos os dois níveis estão. "
        "A Distância do preço à confluência mede apenas onde o mercado está "
        "em relação àquela região."
    )

    st.markdown(
        "#### Arquitetura Cloud"
    )

    st.caption(
        "O cálculo pesado roda somente ao preparar/atualizar o painel e acontece "
        "em um processo separado. O COTAHIST do GEX não é carregado porque não "
        "participa da matemática deste painel conjunto."
    )

    if btc and btc.get("anual"):
        st.markdown(
            "#### Bitcoin — somente GARCH Anual"
        )

        st.write(
            f"Preço: {fmt_num(btc['preco'])} • "
            f"Banda: {btc['anual']['rotulo']} • "
            f"Distância: {fmt_pct(btc['anual']['dist_pct'])} • "
            f"{btc['anual']['status']}"
        )

    worker_info = payload.get(
        "worker_info",
        {},
    )

    if worker_info:
        st.markdown(
            "#### Diagnóstico da última atualização"
        )

        st.json(
            worker_info
        )
