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
    ("-2Ïƒ", "menos_2"),
    ("-1,5Ïƒ", "menos_15"),
    ("+1,5Ïƒ", "mais_15"),
    ("+2Ïƒ", "mais_2"),
]

st.set_page_config(
    page_title="GARCH Ã— GEX â€” ConfluÃªncia V3",
    page_icon="ðŸ“¡",
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
    # O worker tambÃ©m participa da construÃ§Ã£o do payload/cache. IncluÃ­-lo aqui
    # evita aceitar um cache produzido por uma versÃ£o incompatÃ­vel do worker.
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
        return None, f"CACHE_ILEGÃVEL: {type(exc).__name__}: {exc}"

    if not isinstance(payload, dict):
        return None, "CACHE_INVÃLIDO"

    if payload.get("cache_schema") != CACHE_SCHEMA:
        return None, "CACHE_INCOMPATÃVEL_COM_ESTA_VERSÃƒO"

    try:
        hashes_atuais = current_core_hashes()
    except Exception as exc:
        return None, f"ERRO_AO_VALIDAR_CORES: {type(exc).__name__}: {exc}"

    if payload.get("core_hashes") != hashes_atuais:
        return None, "CACHE_DE_OUTRA_VERSÃƒO_DOS_MOTORES"

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
        "Atualizando B3, GEX, GARCH e confluÃªncias...",
        expanded=True,
    )

    log_box = st.empty()
    # MantÃ©m somente uma janela recente na memÃ³ria. O histÃ³rico completo continua
    # disponÃ­vel nos Cloud Logs porque cada linha tambÃ©m Ã© reimpressa no stdout.
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
        status.update(label="Falha ao iniciar atualizaÃ§Ã£o.", state="error")
        return False, f"{type(exc).__name__}: {exc}"

    assert process.stdout is not None

    for raw in process.stdout:
        line = raw.rstrip()
        if not line:
            continue
        linhas.append(line)
        # O subprocesso Ã© capturado para a interface; sem este print, as linhas
        # do worker desaparecem do Cloud Log justamente se o app for encerrado.
        print(line, flush=True)
        log_box.code(
            ultimas_linhas(linhas),
            language="text",
        )

    return_code = process.wait()

    if return_code != 0:
        status.update(
            label=f"AtualizaÃ§Ã£o falhou (worker retornou {return_code}).",
            state="error",
        )
        return False, ultimas_linhas(linhas, 30)

    status.update(
        label="AtualizaÃ§Ã£o concluÃ­da. Carregando o painel...",
        state="complete",
    )
    return True, ultimas_linhas(linhas, 30)


# ======================================================================================
# FORMATAÃ‡ÃƒO
# ======================================================================================

def fmt_num(v, casas=2, vazio="â€”"):
    try:
        x = float(v)
        if not np.isfinite(x):
            return vazio
        return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return vazio


def fmt_pct(v, casas=2, vazio="â€”"):
    t = fmt_num(v, casas, vazio)
    return t if t == vazio else f"{t}%"


def fmt_gamma(v):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "â€”"
        a = abs(x)
        if a >= 1_000_000_000:
            return f"{x/1_000_000_000:.2f} bi"
        if a >= 1_000_000:
            return f"{x/1_000_000:.2f} mi"
        if a >= 1_000:
            return f"{x/1_000:.2f} mil"
        return f"{x:.2f}"
    except Exception:
        return "â€”"


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
        # Percentuais tÃªm prioridade: nomes como "Dist PreÃ§oâ†’Zona %" tambÃ©m
        # contÃªm a palavra PreÃ§o e nÃ£o podem ser formatados como valor em R$.
        if "%" in col:
            formatos[col] = st.column_config.NumberColumn(format="%.2f%%")
        elif "PreÃ§o" in col or col == "Spot GEX":
            formatos[col] = st.column_config.NumberColumn(format="%.2f")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=formatos,
        height=min(820, 38 * (len(df) + 1)),
    )


# ======================================================================================
# CABEÃ‡ALHO
# ======================================================================================

st.title("GARCH Ã— GEX â€” CONFLUÃŠNCIA V3")
st.markdown(
    '<div class="v3-sub">Radar W1 â€¢ Walls W2/W3 â€¢ MensalÃ—30D â€¢ SemestralÃ—90D â€¢ SemestralÃ—180D â€¢ GARCH Mensal/Semestral/Anual</div>',
    unsafe_allow_html=True,
)

payload, cache_error = carregar_cache()

# Quando jÃ¡ existe cache vÃ¡lido, a atualizaÃ§Ã£o fica em uma linha prÃ³pria,
# com botÃ£o primÃ¡rio e largura suficiente para permanecer visÃ­vel no tablet.
# A lÃ³gica continua igual: o clique chama o worker com force_gex=True.
atualizar = False
if payload is not None:
    col_atualizar, col_ultima_atualizacao = st.columns([2, 5])

    with col_atualizar:
        atualizar = st.button(
            "ðŸ”„ ATUALIZAR PAINEL",
            type="primary",
            use_container_width=True,
            key="atualizar_painel",
        )

    with col_ultima_atualizacao:
        st.caption(
            f"Ãšltima atualizaÃ§Ã£o do painel: "
            f"{fmt_momento(payload.get('generated_at'))}"
        )

# O app abre sem disparar o pipeline pesado.
if payload is None:
    st.warning(
        "O painel ainda nÃ£o possui um cache calculado compatÃ­vel nesta instÃ¢ncia. "
        "A pÃ¡gina abriu sem processar a B3 para evitar o travamento do Streamlit Cloud."
    )

    if cache_error not in (None, "CACHE_AUSENTE"):
        st.caption(f"Estado do cache: {cache_error}")

    st.markdown(
        """
<div class="v3-status">
<b>Primeira execuÃ§Ã£o:</b> clique em <b>Preparar painel agora</b>.
O cÃ¡lculo pesado serÃ¡ feito em um processo separado, com leitura B3 filtrada para os ativos monitorados. O resultado final sÃ³ substitui
o cache depois que todas as etapas terminarem.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("â–¶ Preparar painel agora", type="primary"):
        ok, mensagem = executar_worker(force_gex=False)

        if ok:
            st.rerun()
        else:
            st.error(
                "A preparaÃ§Ã£o nÃ£o terminou. O processo principal do Streamlit permaneceu ativo."
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
            "A atualizaÃ§Ã£o falhou, mas o Ãºltimo cache vÃ¡lido foi preservado. "
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
# TABELAS / GRÃFICOS / DETALHES
# ======================================================================================

def bandas_garch_do_periodo(ativo_res, periodo):
    """ObtÃ©m as bandas GARCH jÃ¡ calculadas e armazenadas no payload, sem recalcular o modelo."""
    periodo = str(periodo).upper()

    if periodo == "MENSAL":
        bloco = ativo_res.get("blocos", {}).get("Mensal Ã— 30D", {})
        return bloco.get("bandas")

    if periodo == "SEMESTRAL":
        # 90D e 180D usam o mesmo GARCH Semestral. Preferimos 90D e usamos 180D como fallback.
        for nome in ("Semestral Ã— 90D", "Semestral Ã— 180D"):
            bloco = ativo_res.get("blocos", {}).get(nome, {})
            bandas = bloco.get("bandas")
            if bandas:
                return bandas
        return None

    return None


def leitura_garch_puro(ativo_res, periodo):
    """Replica apenas a leitura de Banda/DistÃ¢ncia/Status do GARCH original sobre bandas jÃ¡ prontas."""
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

    preco = core.numero_seguro(ativo_res.get("PreÃ§o GARCH"))
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
    """Adiciona GARCH puro Mensal/Semestral ao radar sem alterar confluÃªncias nem o cache."""
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
        saida[f"{rotulo} Â· Banda"] = saida["Ativo"].map(
            lambda ativo: mapa.get(ativo, {}).get(periodo, {}).get("banda", "SEM DADOS")
        )
        saida[f"{rotulo} Â· Dist %"] = saida["Ativo"].map(
            lambda ativo: mapa.get(ativo, {}).get(periodo, {}).get("dist_pct", np.nan)
        )
        saida[f"{rotulo} Â· Status"] = saida["Ativo"].map(
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
        "Ativo", "Empresa", "PreÃ§o atual", "PreÃ§o GARCH", "Spot GEX",
        "Mensal Â· Banda", "Mensal Â· Dist %", "Mensal Â· Status",
        "30D Â· Principal", "30D Â· ConfluÃªncia %", "30D Â· Dist PreÃ§oâ†’Zona %", "30D Â· Qualidade",
        "Semestral Â· Banda", "Semestral Â· Dist %", "Semestral Â· Status",
        "90D Â· Principal", "90D Â· ConfluÃªncia %", "90D Â· Dist PreÃ§oâ†’Zona %", "90D Â· Qualidade",
        "180D Â· Principal", "180D Â· ConfluÃªncia %", "180D Â· Dist PreÃ§oâ†’Zona %", "180D Â· Qualidade",
        "Anual Â· Banda", "Anual Â· Dist %", "Anual Â· Status",
    ]

    return df[[c for c in cols if c in df.columns]]


def _texto_garch_resumido(leitura):
    """Resume a leitura do GARCH puro para uma cÃ©lula compacta, sem mudar a regra original."""
    if not leitura:
        return "N/D"

    status = str(leitura.get("status", "SEM DADOS") or "SEM DADOS")
    banda = str(leitura.get("banda", "SEM DADOS") or "SEM DADOS")
    dist = core.numero_seguro(leitura.get("dist_pct"))

    if status == "SEM DADOS" or banda == "SEM DADOS":
        return "N/D"

    dist_txt = fmt_pct(dist) if np.isfinite(dist) else "â€”"

    # O status NORMAL nÃ£o traz a banda no prÃ³prio texto; nos demais estados
    # (PRÃ“XIMO/ACIMA/ABAIXO), o status original jÃ¡ identifica a regiÃ£o.
    if "NORMAL" in status.upper():
        return f"{status} Â· banda {banda} Â· {dist_txt}"

    return f"{status} Â· {dist_txt}"


def tabela_garch_puro(resultados):
    """VisÃ£o separada do GARCH puro Mensal/Semestral/Anual.

    Essa tabela nÃ£o participa da confluÃªncia. Ela apenas reapresenta a leitura
    original Banda/DistÃ¢ncia/Status jÃ¡ calculada para cada ativo.
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
                "PreÃ§o GARCH": core.numero_seguro(ativo_res.get("PreÃ§o GARCH")),
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


def _nivel_w1_por_lado(bloco, lado):
    """ObtÃ©m o nÃ­vel da W1 de Call ou Put a partir dos dados jÃ¡ existentes no bloco.

    NÃ£o recalcula Walls e nÃ£o altera o ranking do GEX. Apenas lÃª Principal W1 e
    as comparaÃ§Ãµes W1 jÃ¡ armazenadas no cache para identificar o nÃ­vel do lado
    solicitado.
    """
    lado = str(lado).upper().strip()
    if lado not in {"CALL", "PUT"}:
        return np.nan

    candidatos = []

    principal = bloco.get("principal")
    if isinstance(principal, dict):
        candidatos.append(principal)

    comparacoes = bloco.get("comparacoes_principal", [])
    if isinstance(comparacoes, list):
        candidatos.extend(item for item in comparacoes if isinstance(item, dict))

    for item in candidatos:
        nome = str(item.get("Wall/RegiÃ£o GEX", "") or "").upper()
        if "W1" not in nome or lado not in nome:
            continue

        nivel = core.numero_seguro(item.get("NÃ­vel Wall/RegiÃ£o"))
        if np.isfinite(nivel):
            return float(nivel)

    return np.nan


def _alerta_posicao_estrutura_w1(ativo_res, bloco_nome):
    """Descreve apenas a posiÃ§Ã£o atual do preÃ§o frente Ã  W1 e Ã s bandas extremas.

    Regras objetivas, sem score e sem sinal de compra/venda:
    - inferior: PreÃ§o atual abaixo da Put W1 e tambÃ©m abaixo de -1,5Ïƒ ou -2Ïƒ;
    - superior: PreÃ§o atual acima da Call W1 e tambÃ©m acima de +1,5Ïƒ ou +2Ïƒ.

    O texto usa "ABAIXO"/"ACIMA", e nÃ£o "ROMPEU", porque o cache contÃ©m o
    retrato atual e nÃ£o uma sÃ©rie intradiÃ¡ria que permita provar o instante do
    cruzamento.
    """
    if not isinstance(ativo_res, dict):
        return ""

    bloco = ativo_res.get("blocos", {}).get(bloco_nome)
    if not isinstance(bloco, dict):
        return ""

    preco_atual = core.numero_seguro(ativo_res.get("PreÃ§o atual"))
    if not np.isfinite(preco_atual) or preco_atual <= 0:
        return ""

    bandas = bloco.get("bandas") or {}
    menos_15 = core.numero_seguro(bandas.get("menos_15"))
    menos_2 = core.numero_seguro(bandas.get("menos_2"))
    mais_15 = core.numero_seguro(bandas.get("mais_15"))
    mais_2 = core.numero_seguro(bandas.get("mais_2"))

    put_w1 = _nivel_w1_por_lado(bloco, "PUT")
    call_w1 = _nivel_w1_por_lado(bloco, "CALL")

    alertas = []

    # Lado inferior: exige simultaneamente estar abaixo da Put W1 e de pelo
    # menos uma das duas bandas inferiores. Se estiver abaixo de -2Ïƒ,
    # mostramos o nÃ­vel mais extremo jÃ¡ ultrapassado.
    if np.isfinite(put_w1) and preco_atual < put_w1:
        if np.isfinite(menos_2) and preco_atual < menos_2:
            alertas.append("ðŸ”» ABAIXO PUT W1 E -2Ïƒ")
        elif np.isfinite(menos_15) and preco_atual < menos_15:
            alertas.append("ðŸ”» ABAIXO PUT W1 E -1,5Ïƒ")

    # Lado superior: regra espelhada da inferior.
    if np.isfinite(call_w1) and preco_atual > call_w1:
        if np.isfinite(mais_2) and preco_atual > mais_2:
            alertas.append("ðŸ”º ACIMA CALL W1 E +2Ïƒ")
        elif np.isfinite(mais_15) and preco_atual > mais_15:
            alertas.append("ðŸ”º ACIMA CALL W1 E +1,5Ïƒ")

    return " Â· ".join(alertas)


def _texto_confluencia_radar(conf_pct, dist_preco_pct, alerta_estrutura=""):
    """Texto operacional de uma cÃ©lula do Radar W1.

    Conf = distÃ¢ncia Banda GARCH â†” W1, normalizada pelo Spot GEX.
    Dist. zona = distÃ¢ncia do PreÃ§o atual independente atÃ© a zona Bandaâ†”W1.
    O alvo aparece somente quando o PreÃ§o atual estÃ¡ dentro da zona.
    """
    conf = core.numero_seguro(conf_pct)
    dist_preco = core.numero_seguro(dist_preco_pct)
    alerta_estrutura = str(alerta_estrutura or "").strip()

    if not np.isfinite(conf):
        return "N/D"

    conf_txt = fmt_pct(conf)
    partes = [f"Conf {conf_txt}"]

    # Limiar solicitado apenas como aviso visual. A fÃ³rmula e a ordenaÃ§Ã£o
    # da ConfluÃªncia % permanecem exatamente as mesmas.
    if float(conf) < 1.0:
        partes.append("â­ CONF <1%")

    if np.isfinite(dist_preco):
        if np.isclose(dist_preco, 0.0, atol=1e-12, rtol=0.0):
            partes.append("ðŸŽ¯ PREÃ‡O DENTRO DA ZONA")
        else:
            partes.append(f"Dist. zona {fmt_pct(dist_preco)}")
    else:
        partes.append("Dist. zona N/D")

    if alerta_estrutura:
        partes.append(alerta_estrutura)

    return " Â· ".join(partes)


def tabela_radar_w1(resultados):
    """Radar operacional enxuto: Ativo, PreÃ§o atual e os trÃªs horizontes W1.

    A tabela tÃ©cnica completa continua preservada em tabela_principal().
    """
    base = core.dataframe_radar(resultados)

    if base.empty:
        return base, base

    visual = pd.DataFrame(index=base.index)
    visual["Ativo"] = base["Ativo"]
    visual["PreÃ§o atual"] = pd.to_numeric(base["PreÃ§o atual"], errors="coerce")

    mapa = (
        ("30D â€” Mensal", "30D", "Mensal Ã— 30D"),
        ("90D â€” Semestral", "90D", "Semestral Ã— 90D"),
        ("180D â€” Semestral", "180D", "Semestral Ã— 180D"),
    )

    metricas = pd.DataFrame(index=base.index)

    for coluna_visual, prefixo, bloco_nome in mapa:
        conf_col = f"{prefixo} Â· ConfluÃªncia %"
        preco_col = f"{prefixo} Â· Dist PreÃ§oâ†’Zona %"

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

        alertas_estrutura = [
            _alerta_posicao_estrutura_w1(
                resultados.get(str(ativo), {}),
                bloco_nome,
            )
            for ativo in base["Ativo"]
        ]

        visual[coluna_visual] = [
            _texto_confluencia_radar(c, d, alerta)
            for c, d, alerta in zip(conf, dist_preco, alertas_estrutura)
        ]

        metricas[f"{coluna_visual} Â· conf"] = conf
        metricas[f"{coluna_visual} Â· preco"] = dist_preco

    return visual, metricas


def _css_confluencia_radar(conf, dist_preco):
    """Destaque exclusivamente visual e contÃ­nuo, sem criar faixas de classificaÃ§Ã£o."""
    conf = core.numero_seguro(conf)
    dist_preco = core.numero_seguro(dist_preco)

    if not np.isfinite(conf):
        return "color: rgba(245,247,250,0.45);"

    # TransformaÃ§Ã£o monotÃ´nica apenas visual. Valores muito distantes perdem
    # rapidamente o fundo verde; nÃ£o hÃ¡ cortes, classes ou novo score.
    distancia = max(float(conf), 0.0)
    intensidade = 1.0 / (1.0 + distancia)
    alpha = 0.02 + 0.58 * intensidade
    peso = int(round(600 + 200 * intensidade))

    css = (
        "background-color: rgba(46, 204, 113, "
        f"{alpha:.3f}); color: #F5F7FA; font-weight: {peso};"
    )

    # Contorno somente quando a mÃ©trica jÃ¡ existente Dist PreÃ§oâ†’Zona Ã© zero.
    if np.isfinite(dist_preco) and np.isclose(
        dist_preco,
        0.0,
        atol=1e-12,
        rtol=0.0,
    ):
        css += " box-shadow: inset 0 0 0 2px rgba(247,201,72,0.95);"

    return css


def dataframe_radar_w1(resultados):
    """Renderiza a visÃ£o rÃ¡pida W1 sem poluir a tela com colunas tÃ©cnicas."""
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

    for coluna in ("30D â€” Mensal", "90D â€” Semestral", "180D â€” Semestral"):
        conf = metricas[f"{coluna} Â· conf"]
        dist_preco = metricas[f"{coluna} Â· preco"]

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
            "PreÃ§o atual": st.column_config.NumberColumn(
                "PreÃ§o atual",
                format="%.2f",
                width="small",
            ),
            "30D â€” Mensal": st.column_config.TextColumn(
                "30D â€” Mensal",
                width="large",
                help="GARCH Mensal Ã— W1 do GEX 30D.",
            ),
            "90D â€” Semestral": st.column_config.TextColumn(
                "90D â€” Semestral",
                width="large",
                help="GARCH Semestral Ã— W1 do GEX 90D.",
            ),
            "180D â€” Semestral": st.column_config.TextColumn(
                "180D â€” Semestral",
                width="large",
                help="GARCH Semestral Ã— W1 do GEX 180D.",
            ),
        },
        height=min(820, 38 * (len(visual) + 1)),
    )


def tabela_secundaria(resultados):
    df = core.dataframe_radar(resultados)

    if df.empty:
        return df

    cols = [
        "Ativo", "Empresa", "PreÃ§o atual", "PreÃ§o GARCH", "Spot GEX",
        "30D Â· SecundÃ¡ria", "30D Â· Sec %",
        "90D Â· SecundÃ¡ria", "90D Â· Sec %",
        "180D Â· SecundÃ¡ria", "180D Â· Sec %",
    ]

    return df[[c for c in cols if c in df.columns]]


def grafico_niveis(ativo_res, bloco_nome):
    """Mapa vertical dos nÃ­veis, incluindo PreÃ§o atual, Spot GEX e PreÃ§o GARCH."""
    bloco = ativo_res["blocos"][bloco_nome]
    preco_atual = core.numero_seguro(ativo_res.get("PreÃ§o atual"))
    spot = core.numero_seguro(ativo_res.get("Spot GEX"))
    preco_garch = core.numero_seguro(ativo_res.get("PreÃ§o GARCH"))
    momento_preco_atual = ativo_res.get("Momento preÃ§o atual", pd.NaT)
    fonte_preco_atual = str(ativo_res.get("Fonte preÃ§o atual", "N/D") or "N/D")
    bandas = bloco.get("bandas") or {}
    p = bloco.get("principal")
    s = bloco.get("secundaria")

    # Paleta explÃ­cita para nÃ£o depender do tema automÃ¡tico do Plotly/Streamlit.
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

    # Quatro bandas do GARCH do perÃ­odo selecionado.
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
            text=f"GARCH {rotulo} Â· {fmt_num(nivel)}",
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
        principal_nivel = core.numero_seguro(p.get("NÃ­vel Wall/RegiÃ£o"))
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
        secundaria_nivel = core.numero_seguro(s.get("NÃ­vel Wall/RegiÃ£o"))
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
            f"PreÃ§o atual: {fmt_num(preco_atual)}"
            f"<br>Fonte: {fonte_preco_atual}"
            f"<br>Momento: {fmt_momento(momento_preco_atual)}"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[0.50],
                y=[preco_atual],
                mode="markers",
                name="PreÃ§o atual",
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
                name="PreÃ§o GARCH",
                marker=dict(
                    size=13,
                    symbol="diamond",
                    color=cor_preco_garch,
                    line=dict(color=cor_fundo, width=2),
                ),
                hovertemplate=f"PreÃ§o GARCH: {fmt_num(preco_garch)}<extra></extra>",
            )
        )

    # Enquadramento vertical somente com os nÃ­veis realmente existentes.
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
            text=f"Principal {p['Wall/RegiÃ£o GEX']} Â· {fmt_num(principal_nivel)}",
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
            text=f"SecundÃ¡ria {s['Wall/RegiÃ£o GEX']} Â· {fmt_num(secundaria_nivel)}",
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

    # X distintos reduzem sobreposiÃ§Ã£o mesmo quando os trÃªs preÃ§os sÃ£o muito prÃ³ximos.
    if np.isfinite(spot):
        fig.add_annotation(
            x=0.30,
            y=spot,
            text=f"Spot GEX Â· {fmt_num(spot)}",
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
            text=f"PreÃ§o atual Â· {fmt_num(preco_atual)}",
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
            text=f"PreÃ§o GARCH Â· {fmt_num(preco_garch)}",
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
        title="PreÃ§o",
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
            text=f"{ativo_res['Ativo']} â€” {bloco_nome}",
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
        "ConfluÃªncia GARCH Ã— GEX",
        fmt_pct(item["DiferenÃ§a Wallâ†”Banda %"], 3),
    )

    c2.metric(
        "DistÃ¢ncia do preÃ§o atual Ã  zona",
        fmt_pct(item.get("Dist PreÃ§oâ†’Zona %"), 3),
    )

    c3.metric(
        "Qualidade GEX",
        f"{fmt_num(item['Qualidade GEX'],1)} Â· {item['Classe qualidade']}",
    )

    c4.metric(
        "Wall / RegiÃ£o",
        item["Wall/RegiÃ£o GEX"],
    )

    st.markdown(
        f"""
<div class="v3-note">
<b>{item['Banda GARCH']}</b> em <b>{fmt_num(item['NÃ­vel banda'])}</b>
&nbsp; Ã— &nbsp;
<b>{item['Wall/RegiÃ£o GEX']}</b> em <b>{fmt_num(item['NÃ­vel Wall/RegiÃ£o'])}</b><br>
PreÃ§o atual: <b>{fmt_num(item.get('PreÃ§o atual'))}</b>
&nbsp; | &nbsp; Zona: <b>{fmt_num(item['Zona inferior'])}</b> a <b>{fmt_num(item['Zona superior'])}</b>
&nbsp; | &nbsp; Centro: <b>{fmt_num(item['Centro da zona'])}</b>
&nbsp; | &nbsp; {item.get('PosiÃ§Ã£o da zona vs PreÃ§o', 'SEM DADOS')}
</div>
""",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "ParticipaÃ§Ã£o Call",
        fmt_pct(item.get("ParticipaÃ§Ã£o Call %")),
    )

    d2.metric(
        "ParticipaÃ§Ã£o Put",
        fmt_pct(item.get("ParticipaÃ§Ã£o Put %")),
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
        "SECUNDÃRIA W2/W3",
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
    f"Base GEX: fechamento B3 {gex_date.strftime('%d/%m/%Y')} â€¢ "
    f"Painel calculado em {generated_at.strftime('%d/%m/%Y %H:%M')} â€¢ "
    f"ConfluÃªncia = distÃ¢ncia entre Banda GARCH e Wall/RegiÃ£o GEX."
)

if erros_worker:
    st.warning(
        f"{len(erros_worker)} ativo(s) tiveram erro parcial na Ãºltima atualizaÃ§Ã£o. "
        "Os demais resultados vÃ¡lidos foram preservados."
    )
    with st.expander("Ver erro(s) parcial(is) da Ãºltima atualizaÃ§Ã£o", expanded=False):
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
    st.subheader("Radar W1 â€” visÃ£o rÃ¡pida")

    st.caption(
        "Cada horizonte mantÃ©m as mesmas mÃ©tricas da V3: "
        "Conf = distÃ¢ncia entre Banda GARCH e W1, com Spot GEX como denominador; "
        "Dist. zona = distÃ¢ncia do PreÃ§o atual atÃ© a zona Bandaâ†”W1. "
        "ðŸŽ¯ PREÃ‡O DENTRO DA ZONA = preÃ§o dentro do intervalo; "
        "ðŸ”» = preÃ§o abaixo da Put W1 e tambÃ©m de -1,5Ïƒ ou -2Ïƒ; "
        "ðŸ”º = preÃ§o acima da Call W1 e tambÃ©m de +1,5Ïƒ ou +2Ïƒ; "
        "â­ CONF <1% = a ConfluÃªncia jÃ¡ calculada Ã© menor que 1%. "
        "SÃ£o avisos factuais de posiÃ§Ã£o, sem score e sem sinal de compra/venda."
    )

    dataframe_radar_w1(resultados)

    with st.expander(
        "Ver GARCH puro â€” Mensal / Semestral / Anual",
        expanded=False,
    ):
        st.caption(
            "Esta leitura Ã© somente do GARCH em relaÃ§Ã£o ao PreÃ§o GARCH. "
            "Ela nÃ£o mede confluÃªncia com GEX e pode apontar uma banda diferente da banda usada na confluÃªncia."
        )
        dataframe_display(
            tabela_garch_puro(resultados)
        )

    with st.expander(
        "Ver tabela tÃ©cnica completa do Radar W1",
        expanded=False,
    ):
        st.caption(
            "Preserva Banda/Dist/Status do GARCH puro, W1 escolhida, ConfluÃªncia %, "
            "DistÃ¢ncia PreÃ§o atualâ†’Zona, Qualidade e todos os horizontes 30D/90D/180D."
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
    st.subheader("Walls secundÃ¡rias â€” W2/W3")

    st.caption(
        "W2/W3 permanecem separadas da W1 e nÃ£o substituem a confluÃªncia principal."
    )

    dataframe_display(
        tabela_secundaria(resultados)
    )

    sec = core.dataframe_detalhes(
        resultados,
        "SECUNDÃRIA W2/W3",
    )

    if not sec.empty:
        cols = [
            "Ativo",
            "Bloco",
            "Banda GARCH",
            "NÃ­vel banda",
            "Wall/RegiÃ£o GEX",
            "Rank Wall",
            "DiferenÃ§a Wallâ†”Banda %",
            "Dist PreÃ§oâ†’Zona %",
            "ParticipaÃ§Ã£o Wall %",
            "ParticipaÃ§Ã£o Call %",
            "ParticipaÃ§Ã£o Put %",
            "Gross Gamma Wall",
            "Gross Gamma Call",
            "Gross Gamma Put",
            "Qualidade GEX",
            "Classe qualidade",
            "SÃ©ries GEX",
            "Vencimentos GEX",
        ]

        with st.expander(
            "Ver detalhes completos da SecundÃ¡ria W2/W3",
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
        "PreÃ§o atual",
        fmt_num(ativo_res.get("PreÃ§o atual")),
    )

    c2.metric(
        "Spot GEX",
        fmt_num(ativo_res.get("Spot GEX")),
    )

    c3.metric(
        "PreÃ§o GARCH",
        fmt_num(ativo_res.get("PreÃ§o GARCH")),
    )

    c4.metric(
        "Base GEX",
        pd.Timestamp(
            ativo_res["Data efetiva GEX"]
        ).strftime("%d/%m/%Y"),
    )

    st.caption(
        f"PreÃ§o atual: {ativo_res.get('Fonte preÃ§o atual', 'N/D')} â€¢ "
        f"momento {fmt_momento(ativo_res.get('Momento preÃ§o atual'))} â€¢ "
        f"Dif. PreÃ§o GARCH Ã— Spot GEX: "
        f"{fmt_pct(ativo_res.get('PreÃ§o GARCH Ã— Spot GEX Â· Dif %'))}. "
        "PreÃ§o atual, Spot GEX e PreÃ§o GARCH sÃ£o referÃªncias separadas."
    )

    st.markdown("#### PerÃ­odo exibido no grÃ¡fico e nos detalhes")
    st.caption(
        "Troque aqui entre Mensal Ã— 30D, Semestral Ã— 90D e Semestral Ã— 180D. "
        "O mapa, Principal W1, SecundÃ¡ria W2/W3 e a tabela de combinaÃ§Ãµes abaixo mudam juntos."
    )

    bloco_nome = st.radio(
        "PerÃ­odo / bloco",
        [
            "Mensal Ã— 30D",
            "Semestral Ã— 90D",
            "Semestral Ã— 180D",
        ],
        horizontal=True,
        key="bloco_detalhe_periodo",
    )

    bloco = ativo_res["blocos"][bloco_nome]

    st.markdown("#### Mapa de nÃ­veis")
    st.caption(
        "O grÃ¡fico mostra as quatro bandas GARCH do perÃ­odo escolhido, Principal W1, "
        "SecundÃ¡ria W2/W3, PreÃ§o atual, Spot GEX e PreÃ§o GARCH. "
        "O PreÃ§o atual Ã© a referÃªncia usada para medir a distÃ¢ncia atÃ© a zona. "
        "A banda usada na confluÃªncia continua sendo a que ficou mais prÃ³xima da Wall/RegiÃ£o."
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
        "SecundÃ¡ria W2/W3",
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
            "#### Todas as combinaÃ§Ãµes deste bloco"
        )

        cols = [
            "Camada",
            "PreÃ§o atual",
            "PreÃ§o GARCH",
            "Spot GEX",
            "Banda GARCH",
            "NÃ­vel banda",
            "Wall/RegiÃ£o GEX",
            "Rank Wall",
            "NÃ­vel Wall/RegiÃ£o",
            "DiferenÃ§a Wallâ†”Banda %",
            "Dist PreÃ§oâ†’Zona %",
            "Dist PreÃ§oâ†’Centro %",
            "PosiÃ§Ã£o da zona vs PreÃ§o",
            "ParticipaÃ§Ã£o Wall %",
            "ParticipaÃ§Ã£o Call %",
            "ParticipaÃ§Ã£o Put %",
            "Gross Gamma Wall",
            "Gross Gamma Call",
            "Gross Gamma Put",
            "Qualidade GEX",
            "Classe qualidade",
            "SÃ©ries GEX",
            "Vencimentos GEX",
        ]

        dataframe_display(
            todas[
                [c for c in cols if c in todas.columns]
            ]
        )

    anual = ativo_res.get("anual")

    st.markdown(
        "#### GARCH Anual â€” sem contraparte GEX"
    )

    if anual:
        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Banda",
            anual["rotulo"],
        )

        a2.metric(
            "NÃ­vel",
            fmt_num(anual["nivel"]),
        )

        a3.metric(
            "DistÃ¢ncia",
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
        "Como funciona â€” metodologia preservada da V3"
    )

    st.markdown(
        """
### O que cada aba mostra

- **Radar W1:** triagem principal. Mostra Ativo, PreÃ§o atual e os trÃªs horizontes de confluÃªncia com a Wall principal W1.
- **Walls W2/W3:** contexto secundÃ¡rio. Mostra as Walls de rank 2 e 3, que nÃ£o substituem a W1.
- **Detalhar ativo:** investigaÃ§Ã£o de um ativo, com Principal W1, SecundÃ¡ria W2/W3, mapa de nÃ­veis e todas as combinaÃ§Ãµes do bloco.
- **Como funciona:** regras, metodologia e diagnÃ³stico da atualizaÃ§Ã£o.

### Regras preservadas

- **Principal:** somente Call/Put W1.
- **SecundÃ¡ria:** somente W2/W3.
- **Mensal GARCH Ã— GEX 30D**.
- **Semestral GARCH Ã— GEX 90D**.
- **Semestral GARCH Ã— GEX 180D**.
- **GARCH puro Mensal/Semestral/Anual:** Banda, DistÃ¢ncia e Status usam a leitura original do GARCH em relaÃ§Ã£o ao PreÃ§o GARCH.
- **Banda da confluÃªncia:** Ã© a banda GARCH que ficou mais prÃ³xima da Wall/RegiÃ£o GEX do bloco e pode ser diferente da banda GARCH mais prÃ³xima do preÃ§o.
- **GARCH Anual:** permanece sem contraparte GEX.
- **GEX 60D:** nÃ£o participa deste painel conjunto.
- **ConfluÃªncia GARCH Ã— GEX (%):** `|Wall/RegiÃ£o GEX âˆ’ Banda GARCH| / Spot GEX Ã— 100`.
- **PreÃ§o atual:** cotaÃ§Ã£o independente do mercado, usada somente para posiÃ§Ã£o/distÃ¢ncia atÃ© a zona e para exibiÃ§Ã£o.
- **Spot GEX:** permanece referÃªncia interna do GEX e denominador da ConfluÃªncia %.
- **PreÃ§o GARCH:** permanece referÃªncia da leitura Banda/DistÃ¢ncia/Status do GARCH.
- **DistÃ¢ncia do preÃ§o atual Ã  zona:** usa uma cotaÃ§Ã£o de mercado independente, capturada na atualizaÃ§Ã£o do painel. Ã‰ a distÃ¢ncia desse PreÃ§o atual atÃ© o intervalo entre Banda e Wall/RegiÃ£o; se o PreÃ§o atual estiver dentro do intervalo, Ã© zero. Spot GEX e PreÃ§o GARCH nÃ£o sÃ£o usados como substitutos.
- **Avisos objetivos do Radar W1:** ðŸŽ¯ indica PreÃ§o atual dentro da zona; ðŸ”» aparece somente quando o PreÃ§o atual estÃ¡ abaixo da Put W1 e tambÃ©m abaixo de -1,5Ïƒ ou -2Ïƒ; ðŸ”º Ã© a regra espelhada acima da Call W1 e de +1,5Ïƒ ou +2Ïƒ; â­ CONF <1% aparece quando a prÃ³pria ConfluÃªncia % jÃ¡ calculada Ã© estritamente menor que 1%. Esses avisos nÃ£o alteram cÃ¡lculos, ranking, score ou sinal.
- Call/Put do mesmo rank sÃ³ sÃ£o agrupadas usando a tolerÃ¢ncia original do GEX.
- ParticipaÃ§Ãµes e Gross Gamma de Call/Put compartilhadas **nÃ£o sÃ£o somados artificialmente**.
- NÃ£o hÃ¡ score composto, nem limiar Forte/Moderada/Fraca, nem sinal de compra/venda.
"""
    )

    st.info(
        "A ConfluÃªncia GARCH Ã— GEX mede quÃ£o prÃ³ximos os dois nÃ­veis estÃ£o. "
        "A DistÃ¢ncia do preÃ§o atual Ã  zona mede onde a cotaÃ§Ã£o independente do mercado estÃ¡ "
        "em relaÃ§Ã£o Ã quela regiÃ£o. Spot GEX e PreÃ§o GARCH permanecem referÃªncias prÃ³prias dos seus motores."
    )

    st.markdown(
        "#### Arquitetura Cloud"
    )

    st.caption(
        "O cÃ¡lculo pesado roda somente ao preparar/atualizar o painel e acontece "
        "em um processo separado. O COTAHIST do GEX nÃ£o Ã© carregado porque nÃ£o "
        "participa da matemÃ¡tica deste painel conjunto."
    )

    if btc and btc.get("anual"):
        st.markdown(
            "#### Bitcoin â€” somente GARCH Anual"
        )

        st.write(
            f"PreÃ§o: {fmt_num(btc['preco'])} â€¢ "
            f"Banda: {btc['anual']['rotulo']} â€¢ "
            f"DistÃ¢ncia: {fmt_pct(btc['anual']['dist_pct'])} â€¢ "
            f"{btc['anual']['status']}"
        )

    worker_info = payload.get(
        "worker_info",
        {},
    )

    if worker_info:
        st.markdown(
            "#### DiagnÃ³stico da Ãºltima atualizaÃ§Ã£o"
        )

        st.json(
            worker_info
        )
