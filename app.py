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
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CACHE_SCHEMA = 4
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
    # O worker também participa da construção do payload/cache. Incluí-lo aqui
    # evita aceitar um cache produzido por uma versão incompatível do worker.
    return {
        nome: sha256_file(MODULE_DIR / nome)
        for nome in (
            "gex_core.py",
            "garch_core.py",
            "confluence_core.py",
            "panel_worker.py",
        )
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
    seq = list(linhas)
    return "\n".join(seq[-n:])


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
    # Mantém somente uma janela recente na memória. O histórico completo continua
    # disponível nos Cloud Logs porque cada linha também é reimpressa no stdout.
    linhas = deque(maxlen=400)

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
        # O subprocesso é capturado para a interface; sem este print, as linhas
        # do worker desaparecem do Cloud Log justamente se o app for encerrado.
        print(line, flush=True)
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


def fmt_momento(v, vazio="N/D"):
    try:
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            return vazio
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/Sao_Paulo").tz_localize(None)
        return ts.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return vazio


def dataframe_display(df):
    formatos = {}

    for col in df.columns:
        # Percentuais têm prioridade: nomes como "Dist Preço→Zona %" também
        # contêm a palavra Preço e não podem ser formatados como valor em R$.
        if "%" in col:
            formatos[col] = st.column_config.NumberColumn(format="%.2f%%")
        elif "Preço" in col or col == "Spot GEX":
            formatos[col] = st.column_config.NumberColumn(format="%.2f")

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

st.title("GARCH × GEX — CONFLUÊNCIA V3")
st.markdown(
    '<div class="v3-sub">Radar W1 • Walls W2/W3 • Mensal×30D • Semestral×90D • Semestral×180D • GARCH Mensal/Semestral/Anual</div>',
    unsafe_allow_html=True,
)

payload, cache_error = carregar_cache()

# Quando já existe cache válido, a atualização fica em uma linha própria,
# com botão primário e largura suficiente para permanecer visível no tablet.
# A lógica continua igual: o clique chama o worker com force_gex=True.
atualizar = False
if payload is not None:
    col_atualizar, col_ultima_atualizacao = st.columns([2, 5])

    with col_atualizar:
        atualizar = st.button(
            "🔄 ATUALIZAR PAINEL",
            type="primary",
            use_container_width=True,
            key="atualizar_painel",
        )

    with col_ultima_atualizacao:
        st.caption(
            f"Última atualização do painel: "
            f"{fmt_momento(payload.get('generated_at'))}"
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
O cálculo pesado será feito em um processo separado, com leitura B3 filtrada para os ativos monitorados. O resultado final só substitui
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
import garch_core as garch
import plotly.graph_objects as go

gc.collect()


# ======================================================================================
# TABELAS / GRÁFICOS / DETALHES
# ======================================================================================

def bandas_garch_do_periodo(ativo_res, periodo):
    """Obtém as bandas GARCH já calculadas e armazenadas no payload, sem recalcular o modelo."""
    periodo = str(periodo).upper()

    if periodo == "MENSAL":
        bloco = ativo_res.get("blocos", {}).get("Mensal × 30D", {})
        return bloco.get("bandas")

    if periodo == "SEMESTRAL":
        # 90D e 180D usam o mesmo GARCH Semestral. Preferimos 90D e usamos 180D como fallback.
        for nome in ("Semestral × 90D", "Semestral × 180D"):
            bloco = ativo_res.get("blocos", {}).get(nome, {})
            bandas = bloco.get("bandas")
            if bandas:
                return bandas
        return None

    return None


def leitura_garch_puro(ativo_res, periodo):
    """Replica apenas a leitura de Banda/Distância/Status do GARCH original sobre bandas já prontas."""
    periodo = str(periodo).upper()

    if periodo == "ANUAL":
        anual = ativo_res.get("anual")
        if not anual:
            return {
                "banda": "SEM DADOS",
                "dist_pct": np.nan,
                "status": "SEM DADOS",
            }
        return {
            "banda": anual.get("rotulo", "SEM DADOS"),
            "dist_pct": core.numero_seguro(anual.get("dist_pct")),
            "status": anual.get("status", "SEM DADOS"),
        }

    preco = core.numero_seguro(ativo_res.get("Preço GARCH"))
    bandas = bandas_garch_do_periodo(ativo_res, periodo)

    if not bandas or not np.isfinite(preco) or preco <= 0:
        return {
            "banda": "SEM DADOS",
            "dist_pct": np.nan,
            "status": "SEM DADOS",
        }

    try:
        status, proxima, distancia, _prioridade = garch.analisar_status(
            preco,
            bandas,
        )
    except Exception as exc:
        print(
            f"[V3 APP] Falha ao ler GARCH puro {periodo}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "banda": "SEM DADOS",
            "dist_pct": np.nan,
            "status": "SEM DADOS",
        }

    return {
        "banda": proxima,
        "dist_pct": core.numero_seguro(distancia),
        "status": status,
    }


def adicionar_leituras_garch(df, resultados):
    """Adiciona GARCH puro Mensal/Semestral ao radar sem alterar confluências nem o cache."""
    if df.empty or "Ativo" not in df.columns:
        return df

    saida = df.copy()
    mapa = {}

    for ativo, ativo_res in resultados.items():
        mapa[ativo] = {
            "MENSAL": leitura_garch_puro(ativo_res, "MENSAL"),
            "SEMESTRAL": leitura_garch_puro(ativo_res, "SEMESTRAL"),
        }

    for periodo, rotulo in (
        ("MENSAL", "Mensal"),
        ("SEMESTRAL", "Semestral"),
    ):
        saida[f"{rotulo} · Banda"] = saida["Ativo"].map(
            lambda ativo: mapa.get(ativo, {}).get(periodo, {}).get("banda", "SEM DADOS")
        )
        saida[f"{rotulo} · Dist %"] = saida["Ativo"].map(
            lambda ativo: mapa.get(ativo, {}).get(periodo, {}).get("dist_pct", np.nan)
        )
        saida[f"{rotulo} · Status"] = saida["Ativo"].map(
            lambda ativo: mapa.get(ativo, {}).get(periodo, {}).get("status", "SEM DADOS")
        )

    return saida


def tabela_principal(resultados):
    df = adicionar_leituras_garch(
        core.dataframe_radar(resultados),
        resultados,
    )

    if df.empty:
        return df

    cols = [
        "Ativo", "Empresa", "Preço atual", "Preço GARCH", "Spot GEX",
        "Mensal · Banda", "Mensal · Dist %", "Mensal · Status",
        "30D · Principal", "30D · Confluência %", "30D · Dist Preço→Zona %", "30D · Qualidade",
        "Semestral · Banda", "Semestral · Dist %", "Semestral · Status",
        "90D · Principal", "90D · Confluência %", "90D · Dist Preço→Zona %", "90D · Qualidade",
        "180D · Principal", "180D · Confluência %", "180D · Dist Preço→Zona %", "180D · Qualidade",
        "Anual · Banda", "Anual · Dist %", "Anual · Status",
    ]

    return df[[c for c in cols if c in df.columns]]


def _texto_garch_resumido(leitura):
    """Resume a leitura do GARCH puro para uma célula compacta, sem mudar a regra original."""
    if not leitura:
        return "N/D"

    status = str(leitura.get("status", "SEM DADOS") or "SEM DADOS")
    banda = str(leitura.get("banda", "SEM DADOS") or "SEM DADOS")
    dist = core.numero_seguro(leitura.get("dist_pct"))

    if status == "SEM DADOS" or banda == "SEM DADOS":
        return "N/D"

    dist_txt = fmt_pct(dist) if np.isfinite(dist) else "—"

    # O status NORMAL não traz a banda no próprio texto; nos demais estados
    # (PRÓXIMO/ACIMA/ABAIXO), o status original já identifica a região.
    if "NORMAL" in status.upper():
        return f"{status} · banda {banda} · {dist_txt}"

    return f"{status} · {dist_txt}"


def tabela_garch_puro(resultados):
    """Visão separada do GARCH puro Mensal/Semestral/Anual.

    Essa tabela não participa da confluência. Ela apenas reapresenta a leitura
    original Banda/Distância/Status já calculada para cada ativo.
    """
    rows = []
    ordem = core.dataframe_radar(resultados)
    ativos_ordenados = (
        ordem["Ativo"].tolist()
        if not ordem.empty and "Ativo" in ordem.columns
        else list(resultados.keys())
    )

    for ativo in ativos_ordenados:
        ativo_res = resultados[ativo]
        rows.append(
            {
                "Ativo": ativo,
                "Preço GARCH": core.numero_seguro(ativo_res.get("Preço GARCH")),
                "Mensal": _texto_garch_resumido(
                    leitura_garch_puro(ativo_res, "MENSAL")
                ),
                "Semestral": _texto_garch_resumido(
                    leitura_garch_puro(ativo_res, "SEMESTRAL")
                ),
                "Anual": _texto_garch_resumido(
                    leitura_garch_puro(ativo_res, "ANUAL")
                ),
            }
        )

    return pd.DataFrame(rows)


def _texto_confluencia_radar(conf_pct, dist_preco_pct):
    """Texto operacional de uma célula do Radar W1.

    Conf = distância Banda GARCH ↔ W1, normalizada pelo Spot GEX.
    Dist. zona = distância do Preço atual independente até a zona Banda↔W1.
    O alvo aparece somente quando o Preço atual está dentro da zona.
    """
    conf = core.numero_seguro(conf_pct)
    dist_preco = core.numero_seguro(dist_preco_pct)

    if not np.isfinite(conf):
        return "N/D"

    conf_txt = fmt_pct(conf)

    if np.isfinite(dist_preco):
        if np.isclose(dist_preco, 0.0, atol=1e-12, rtol=0.0):
            return f"Conf {conf_txt} · 🎯 PREÇO DENTRO DA ZONA"
        return f"Conf {conf_txt} · Dist. zona {fmt_pct(dist_preco)}"

    return f"Conf {conf_txt} · Dist. zona N/D"


def tabela_radar_w1(resultados):
    """Radar operacional enxuto: Ativo, Preço atual e os três horizontes W1.

    A tabela técnica completa continua preservada em tabela_principal().
    """
    base = core.dataframe_radar(resultados)

    if base.empty:
        return base, base

    visual = pd.DataFrame(index=base.index)
    visual["Ativo"] = base["Ativo"]
    visual["Preço atual"] = pd.to_numeric(base["Preço atual"], errors="coerce")

    mapa = (
        ("30D — Mensal", "30D"),
        ("90D — Semestral", "90D"),
        ("180D — Semestral", "180D"),
    )

    metricas = pd.DataFrame(index=base.index)

    for coluna_visual, prefixo in mapa:
        conf_col = f"{prefixo} · Confluência %"
        preco_col = f"{prefixo} · Dist Preço→Zona %"

        conf = (
            pd.to_numeric(base[conf_col], errors="coerce")
            if conf_col in base.columns
            else pd.Series(np.nan, index=base.index, dtype=float)
        )
        dist_preco = (
            pd.to_numeric(base[preco_col], errors="coerce")
            if preco_col in base.columns
            else pd.Series(np.nan, index=base.index, dtype=float)
        )

        visual[coluna_visual] = [
            _texto_confluencia_radar(c, d)
            for c, d in zip(conf, dist_preco)
        ]

        metricas[f"{coluna_visual} · conf"] = conf
        metricas[f"{coluna_visual} · preco"] = dist_preco

    return visual, metricas


def _css_confluencia_radar(conf, dist_preco):
    """Destaque exclusivamente visual e contínuo, sem criar faixas de classificação."""
    conf = core.numero_seguro(conf)
    dist_preco = core.numero_seguro(dist_preco)

    if not np.isfinite(conf):
        return "color: rgba(245,247,250,0.45);"

    # Transformação monotônica apenas visual. Valores muito distantes perdem
    # rapidamente o fundo verde; não há cortes, classes ou novo score.
    distancia = max(float(conf), 0.0)
    intensidade = 1.0 / (1.0 + distancia)
    alpha = 0.02 + 0.58 * intensidade
    peso = int(round(600 + 200 * intensidade))

    css = (
        "background-color: rgba(46, 204, 113, "
        f"{alpha:.3f}); color: #F5F7FA; font-weight: {peso};"
    )

    # Contorno somente quando a métrica já existente Dist Preço→Zona é zero.
    if np.isfinite(dist_preco) and np.isclose(
        dist_preco,
        0.0,
        atol=1e-12,
        rtol=0.0,
    ):
        css += " box-shadow: inset 0 0 0 2px rgba(247,201,72,0.95);"

    return css


def dataframe_radar_w1(resultados):
    """Renderiza a visão rápida W1 sem poluir a tela com colunas técnicas."""
    visual, metricas = tabela_radar_w1(resultados)

    if visual.empty:
        st.dataframe(
            visual,
            use_container_width=True,
            hide_index=True,
        )
        return

    estilos = pd.DataFrame(
        "",
        index=visual.index,
        columns=visual.columns,
    )

    for coluna in ("30D — Mensal", "90D — Semestral", "180D — Semestral"):
        conf = metricas[f"{coluna} · conf"]
        dist_preco = metricas[f"{coluna} · preco"]

        estilos[coluna] = [
            _css_confluencia_radar(c, d)
            for c, d in zip(conf, dist_preco)
        ]

    styler = visual.style.apply(
        lambda _df: estilos,
        axis=None,
    )

    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ativo": st.column_config.TextColumn(
                "Ativo",
                width="small",
            ),
            "Preço atual": st.column_config.NumberColumn(
                "Preço atual",
                format="%.2f",
                width="small",
            ),
            "30D — Mensal": st.column_config.TextColumn(
                "30D — Mensal",
                width="large",
                help="GARCH Mensal × W1 do GEX 30D.",
            ),
            "90D — Semestral": st.column_config.TextColumn(
                "90D — Semestral",
                width="large",
                help="GARCH Semestral × W1 do GEX 90D.",
            ),
            "180D — Semestral": st.column_config.TextColumn(
                "180D — Semestral",
                width="large",
                help="GARCH Semestral × W1 do GEX 180D.",
            ),
        },
        height=min(820, 38 * (len(visual) + 1)),
    )


def tabela_secundaria(resultados):
    df = core.dataframe_radar(resultados)

    if df.empty:
        return df

    cols = [
        "Ativo", "Empresa", "Preço atual", "Preço GARCH", "Spot GEX",
        "30D · Secundária", "30D · Sec %",
        "90D · Secundária", "90D · Sec %",
        "180D · Secundária", "180D · Sec %",
    ]

    return df[[c for c in cols if c in df.columns]]


def grafico_niveis(ativo_res, bloco_nome):
    """Mapa vertical dos níveis, incluindo Preço atual, Spot GEX e Preço GARCH."""
    bloco = ativo_res["blocos"][bloco_nome]
    preco_atual = core.numero_seguro(ativo_res.get("Preço atual"))
    spot = core.numero_seguro(ativo_res.get("Spot GEX"))
    preco_garch = core.numero_seguro(ativo_res.get("Preço GARCH"))
    momento_preco_atual = ativo_res.get("Momento preço atual", pd.NaT)
    fonte_preco_atual = str(ativo_res.get("Fonte preço atual", "N/D") or "N/D")
    bandas = bloco.get("bandas") or {}
    p = bloco.get("principal")
    s = bloco.get("secundaria")

    # Paleta explícita para não depender do tema automático do Plotly/Streamlit.
    cor_fundo = "#0E1117"
    cor_texto = "#F5F7FA"
    cor_grid = "rgba(255,255,255,0.10)"
    cor_garch = "#D0D7DE"
    cor_principal = "#FFB000"
    cor_secundaria = "#B388FF"
    cor_spot = "#4CC9F0"
    cor_preco_atual = "#FFFFFF"
    cor_preco_garch = "#2DE2A6"

    fig = go.Figure()
    niveis_validos = []

    # Quatro bandas do GARCH do período selecionado.
    for rotulo, chave in BANDAS_COMPARADAS:
        nivel = core.numero_seguro(bandas.get(chave))
        if not np.isfinite(nivel):
            continue

        niveis_validos.append(nivel)

        fig.add_shape(
            type="line",
            xref="paper",
            yref="y",
            x0=0.08,
            x1=0.92,
            y0=nivel,
            y1=nivel,
            line=dict(
                color=cor_garch,
                width=2,
                dash="dot",
            ),
            opacity=0.82,
            layer="below",
        )

        fig.add_annotation(
            x=0.98,
            xref="paper",
            y=nivel,
            yref="y",
            text=f"GARCH {rotulo} · {fmt_num(nivel)}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=cor_garch, size=12),
            bgcolor="rgba(14,17,23,0.88)",
            borderpad=2,
        )

    principal_nivel = np.nan
    secundaria_nivel = np.nan

    if p:
        principal_nivel = core.numero_seguro(p.get("Nível Wall/Região"))
        if np.isfinite(principal_nivel):
            niveis_validos.append(principal_nivel)
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0.06,
                x1=0.94,
                y0=principal_nivel,
                y1=principal_nivel,
                line=dict(color=cor_principal, width=4),
                layer="above",
            )

    if s:
        secundaria_nivel = core.numero_seguro(s.get("Nível Wall/Região"))
        if np.isfinite(secundaria_nivel):
            niveis_validos.append(secundaria_nivel)
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0.06,
                x1=0.94,
                y0=secundaria_nivel,
                y1=secundaria_nivel,
                line=dict(color=cor_secundaria, width=3, dash="dash"),
                layer="above",
            )

    if np.isfinite(spot):
        niveis_validos.append(spot)
        fig.add_trace(
            go.Scatter(
                x=[0.30],
                y=[spot],
                mode="markers",
                name="Spot GEX",
                marker=dict(
                    size=14,
                    color=cor_spot,
                    line=dict(color=cor_fundo, width=2),
                ),
                hovertemplate=f"Spot GEX: {fmt_num(spot)}<extra></extra>",
            )
        )

    if np.isfinite(preco_atual):
        niveis_validos.append(preco_atual)
        hover_atual = (
            f"Preço atual: {fmt_num(preco_atual)}"
            f"<br>Fonte: {fonte_preco_atual}"
            f"<br>Momento: {fmt_momento(momento_preco_atual)}"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[0.50],
                y=[preco_atual],
                mode="markers",
                name="Preço atual",
                marker=dict(
                    size=17,
                    symbol="star",
                    color=cor_preco_atual,
                    line=dict(color=cor_fundo, width=2),
                ),
                hovertemplate=hover_atual,
            )
        )

    if np.isfinite(preco_garch):
        niveis_validos.append(preco_garch)
        fig.add_trace(
            go.Scatter(
                x=[0.70],
                y=[preco_garch],
                mode="markers",
                name="Preço GARCH",
                marker=dict(
                    size=13,
                    symbol="diamond",
                    color=cor_preco_garch,
                    line=dict(color=cor_fundo, width=2),
                ),
                hovertemplate=f"Preço GARCH: {fmt_num(preco_garch)}<extra></extra>",
            )
        )

    # Enquadramento vertical somente com os níveis realmente existentes.
    if niveis_validos:
        minimo = float(min(niveis_validos))
        maximo = float(max(niveis_validos))
        amplitude = maximo - minimo
        referencia = max(abs(minimo), abs(maximo), 1.0)
        margem = max(amplitude * 0.08, referencia * 0.015)
        if amplitude <= 0:
            margem = max(referencia * 0.03, 0.50)
        faixa_y = [minimo - margem, maximo + margem]
    else:
        faixa_y = None

    amplitude_rotulos = (
        float(max(niveis_validos) - min(niveis_validos))
        if len(niveis_validos) >= 2
        else 0.0
    )
    limiar_proximidade = max(amplitude_rotulos * 0.035, 0.05)

    principal_shift = 0
    secundaria_shift = 0
    if (
        np.isfinite(principal_nivel)
        and np.isfinite(secundaria_nivel)
        and abs(principal_nivel - secundaria_nivel) <= limiar_proximidade
    ):
        principal_shift = 14
        secundaria_shift = -14

    if np.isfinite(principal_nivel):
        fig.add_annotation(
            x=0.02,
            xref="paper",
            y=principal_nivel,
            yref="y",
            text=f"Principal {p['Wall/Região GEX']} · {fmt_num(principal_nivel)}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            yshift=principal_shift,
            font=dict(color=cor_principal, size=12),
            bgcolor="rgba(14,17,23,0.92)",
            bordercolor=cor_principal,
            borderwidth=1,
            borderpad=3,
        )

    if np.isfinite(secundaria_nivel):
        fig.add_annotation(
            x=0.02,
            xref="paper",
            y=secundaria_nivel,
            yref="y",
            text=f"Secundária {s['Wall/Região GEX']} · {fmt_num(secundaria_nivel)}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            yshift=secundaria_shift,
            font=dict(color=cor_secundaria, size=12),
            bgcolor="rgba(14,17,23,0.92)",
            bordercolor=cor_secundaria,
            borderwidth=1,
            borderpad=3,
        )

    # X distintos reduzem sobreposição mesmo quando os três preços são muito próximos.
    if np.isfinite(spot):
        fig.add_annotation(
            x=0.30,
            y=spot,
            text=f"Spot GEX · {fmt_num(spot)}",
            showarrow=False,
            yshift=-20,
            font=dict(color=cor_spot, size=12),
            bgcolor="rgba(14,17,23,0.88)",
            borderpad=2,
        )

    if np.isfinite(preco_atual):
        fig.add_annotation(
            x=0.50,
            y=preco_atual,
            text=f"Preço atual · {fmt_num(preco_atual)}",
            showarrow=False,
            yshift=24,
            font=dict(color=cor_preco_atual, size=12),
            bgcolor="rgba(14,17,23,0.88)",
            borderpad=2,
        )

    if np.isfinite(preco_garch):
        fig.add_annotation(
            x=0.70,
            y=preco_garch,
            text=f"Preço GARCH · {fmt_num(preco_garch)}",
            showarrow=False,
            yshift=-20,
            font=dict(color=cor_preco_garch, size=12),
            bgcolor="rgba(14,17,23,0.88)",
            borderpad=2,
        )

    fig.update_xaxes(
        visible=False,
        range=[0.0, 1.0],
        fixedrange=True,
    )

    fig.update_yaxes(
        title="Preço",
        range=faixa_y,
        gridcolor=cor_grid,
        gridwidth=1,
        zeroline=False,
        tickfont=dict(color=cor_texto, size=12),
        title_font=dict(color=cor_texto, size=13),
        automargin=True,
    )

    fig.update_layout(
        title=dict(
            text=f"{ativo_res['Ativo']} — {bloco_nome}",
            font=dict(color=cor_texto, size=18),
            x=0.01,
            xanchor="left",
        ),
        height=540,
        margin=dict(l=60, r=28, t=65, b=25),
        paper_bgcolor=cor_fundo,
        plot_bgcolor=cor_fundo,
        font=dict(color=cor_texto),
        showlegend=False,
        hovermode="closest",
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
        "Distância do preço atual à zona",
        fmt_pct(item.get("Dist Preço→Zona %"), 3),
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
Preço atual: <b>{fmt_num(item.get('Preço atual'))}</b>
&nbsp; | &nbsp; Zona: <b>{fmt_num(item['Zona inferior'])}</b> a <b>{fmt_num(item['Zona superior'])}</b>
&nbsp; | &nbsp; Centro: <b>{fmt_num(item['Centro da zona'])}</b>
&nbsp; | &nbsp; {item.get('Posição da zona vs Preço', 'SEM DADOS')}
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

    radar = adicionar_leituras_garch(
        core.dataframe_radar(resultados),
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
    with st.expander("Ver erro(s) parcial(is) da última atualização", expanded=False):
        for ativo_erro, mensagem_erro in erros_worker.items():
            st.code(f"{ativo_erro}: {mensagem_erro}", language="text")


# ======================================================================================
# ABAS
# ======================================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Radar W1",
        "Walls W2/W3",
        "Detalhar ativo",
        "Como funciona",
    ]
)


with tab1:
    st.subheader("Radar W1 — visão rápida")

    st.caption(
        "Cada horizonte mostra duas métricas: "
        "Conf = distância entre Banda GARCH e W1, mantendo o Spot GEX como denominador da regra V3; "
        "Dist. zona = distância do Preço atual independente até a zona formada por Banda e W1. "
        "🎯 PREÇO DENTRO DA ZONA aparece somente quando o Preço atual está dentro desse intervalo. "
        "O fundo verde é apenas intensidade visual contínua da própria Conf %, sem faixas Forte/Moderada/Fraca."
    )

    dataframe_radar_w1(resultados)

    with st.expander(
        "Ver GARCH puro — Mensal / Semestral / Anual",
        expanded=False,
    ):
        st.caption(
            "Esta leitura é somente do GARCH em relação ao Preço GARCH. "
            "Ela não mede confluência com GEX e pode apontar uma banda diferente da banda usada na confluência."
        )
        dataframe_display(
            tabela_garch_puro(resultados)
        )

    with st.expander(
        "Ver tabela técnica completa do Radar W1",
        expanded=False,
    ):
        st.caption(
            "Preserva Banda/Dist/Status do GARCH puro, W1 escolhida, Confluência %, "
            "Distância Preço atual→Zona, Qualidade e todos os horizontes 30D/90D/180D."
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
    st.subheader("Walls secundárias — W2/W3")

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
            "Dist Preço→Zona %",
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

        with st.expander(
            "Ver detalhes completos da Secundária W2/W3",
            expanded=False,
        ):
            dataframe_display(
                sec[[c for c in cols if c in sec.columns]]
            )


with tab3:
    st.subheader("Detalhar ativo")

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
        "Preço atual",
        fmt_num(ativo_res.get("Preço atual")),
    )

    c2.metric(
        "Spot GEX",
        fmt_num(ativo_res.get("Spot GEX")),
    )

    c3.metric(
        "Preço GARCH",
        fmt_num(ativo_res.get("Preço GARCH")),
    )

    c4.metric(
        "Base GEX",
        pd.Timestamp(
            ativo_res["Data efetiva GEX"]
        ).strftime("%d/%m/%Y"),
    )

    st.caption(
        f"Preço atual: {ativo_res.get('Fonte preço atual', 'N/D')} • "
        f"momento {fmt_momento(ativo_res.get('Momento preço atual'))} • "
        f"Dif. Preço GARCH × Spot GEX: "
        f"{fmt_pct(ativo_res.get('Preço GARCH × Spot GEX · Dif %'))}. "
        "Preço atual, Spot GEX e Preço GARCH são referências separadas."
    )

    st.markdown("#### Período exibido no gráfico e nos detalhes")
    st.caption(
        "Troque aqui entre Mensal × 30D, Semestral × 90D e Semestral × 180D. "
        "O mapa, Principal W1, Secundária W2/W3 e a tabela de combinações abaixo mudam juntos."
    )

    bloco_nome = st.radio(
        "Período / bloco",
        [
            "Mensal × 30D",
            "Semestral × 90D",
            "Semestral × 180D",
        ],
        horizontal=True,
        key="bloco_detalhe_periodo",
    )

    bloco = ativo_res["blocos"][bloco_nome]

    st.markdown("#### Mapa de níveis")
    st.caption(
        "O gráfico mostra as quatro bandas GARCH do período escolhido, Principal W1, "
        "Secundária W2/W3, Preço atual, Spot GEX e Preço GARCH. "
        "O Preço atual é a referência usada para medir a distância até a zona. "
        "A banda usada na confluência continua sendo a que ficou mais próxima da Wall/Região."
    )

    st.plotly_chart(
        grafico_niveis(
            ativo_res,
            bloco_nome,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    render_item(
        "Principal W1",
        bloco.get("principal"),
    )

    st.markdown("---")

    render_item(
        "Secundária W2/W3",
        bloco.get("secundaria"),
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
            "Preço atual",
            "Preço GARCH",
            "Spot GEX",
            "Banda GARCH",
            "Nível banda",
            "Wall/Região GEX",
            "Rank Wall",
            "Nível Wall/Região",
            "Diferença Wall↔Banda %",
            "Dist Preço→Zona %",
            "Dist Preço→Centro %",
            "Posição da zona vs Preço",
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
        "Como funciona — metodologia preservada da V3"
    )

    st.markdown(
        """
### O que cada aba mostra

- **Radar W1:** triagem principal. Mostra Ativo, Preço atual e os três horizontes de confluência com a Wall principal W1.
- **Walls W2/W3:** contexto secundário. Mostra as Walls de rank 2 e 3, que não substituem a W1.
- **Detalhar ativo:** investigação de um ativo, com Principal W1, Secundária W2/W3, mapa de níveis e todas as combinações do bloco.
- **Como funciona:** regras, metodologia e diagnóstico da atualização.

### Regras preservadas

- **Principal:** somente Call/Put W1.
- **Secundária:** somente W2/W3.
- **Mensal GARCH × GEX 30D**.
- **Semestral GARCH × GEX 90D**.
- **Semestral GARCH × GEX 180D**.
- **GARCH puro Mensal/Semestral/Anual:** Banda, Distância e Status usam a leitura original do GARCH em relação ao Preço GARCH.
- **Banda da confluência:** é a banda GARCH que ficou mais próxima da Wall/Região GEX do bloco e pode ser diferente da banda GARCH mais próxima do preço.
- **GARCH Anual:** permanece sem contraparte GEX.
- **GEX 60D:** não participa deste painel conjunto.
- **Confluência GARCH × GEX (%):** `|Wall/Região GEX − Banda GARCH| / Spot GEX × 100`.
- **Preço atual:** cotação independente do mercado, usada somente para posição/distância até a zona e para exibição.
- **Spot GEX:** permanece referência interna do GEX e denominador da Confluência %.
- **Preço GARCH:** permanece referência da leitura Banda/Distância/Status do GARCH.
- **Distância do preço atual à zona:** usa uma cotação de mercado independente, capturada na atualização do painel. É a distância desse Preço atual até o intervalo entre Banda e Wall/Região; se o Preço atual estiver dentro do intervalo, é zero. Spot GEX e Preço GARCH não são usados como substitutos.
- Call/Put do mesmo rank só são agrupadas usando a tolerância original do GEX.
- Participações e Gross Gamma de Call/Put compartilhadas **não são somados artificialmente**.
- Não há score composto, nem limiar Forte/Moderada/Fraca, nem sinal de compra/venda.
"""
    )

    st.info(
        "A Confluência GARCH × GEX mede quão próximos os dois níveis estão. "
        "A Distância do preço atual à zona mede onde a cotação independente do mercado está "
        "em relação àquela região. Spot GEX e Preço GARCH permanecem referências próprias dos seus motores."
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
