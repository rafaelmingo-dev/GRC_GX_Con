# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import garch_core as garch
import gex_core as gex

# ======================================================================================
# GARCH × GEX — NÚCLEO DO PAINEL V3
# ======================================================================================
# Regras preservadas da V3 validada:
#   GARCH Mensal    × GEX 30D
#   GARCH Semestral × GEX 90D
#   GARCH Semestral × GEX 180D
#   GARCH Anual     = sozinho
#   GEX 60D         = fora do cruzamento
#
# Principal   = somente W1
# Secundária  = somente W2/W3
# Confluência inferior = Put W1/W2/W3 × bandas inferiores (-1,5σ / -2σ)
# Confluência superior = Call W1/W2/W3 × bandas superiores (+1,5σ / +2σ)
# Combinações cruzadas Call×banda inferior e Put×banda superior são inválidas.
# Sem score composto.
# Sem limiar forte/fraca.
# Sem sinal de compra/venda.
# ======================================================================================

BANDAS_COMPARADAS = [
    ("-2σ", "menos_2"),
    ("-1,5σ", "menos_15"),
    ("+1,5σ", "mais_15"),
    ("+2σ", "mais_2"),
]

BANDAS_INFERIORES = (
    ("-1,5σ", "menos_15"),
    ("-2σ", "menos_2"),
)

BANDAS_SUPERIORES = (
    ("+1,5σ", "mais_15"),
    ("+2σ", "mais_2"),
)

BLOCOS_CONFLUENCIA = [
    {"bloco": "Mensal × 30D", "garch_periodo": "MENSAL", "gex_horizonte": "30 dias"},
    {"bloco": "Semestral × 90D", "garch_periodo": "SEMESTRAL", "gex_horizonte": "90 dias"},
    {"bloco": "Semestral × 180D", "garch_periodo": "SEMESTRAL", "gex_horizonte": "180 dias"},
]

INCLUIR_BTC_ANUAL = True


def validar_configuracao() -> None:
    esperado = [
        ("Mensal × 30D", "MENSAL", "30 dias"),
        ("Semestral × 90D", "SEMESTRAL", "90 dias"),
        ("Semestral × 180D", "SEMESTRAL", "180 dias"),
    ]
    atual = [(b["bloco"], b["garch_periodo"], b["gex_horizonte"]) for b in BLOCOS_CONFLUENCIA]
    if atual != esperado:
        raise RuntimeError("Configuração GARCH × GEX foi alterada indevidamente.")
    if any(b["gex_horizonte"] == "60 dias" for b in BLOCOS_CONFLUENCIA):
        raise RuntimeError("GEX 60D não deve participar do cruzamento V3.")

    if BANDAS_INFERIORES != (
        ("-1,5σ", "menos_15"),
        ("-2σ", "menos_2"),
    ):
        raise RuntimeError("Bandas inferiores direcionais foram alteradas indevidamente.")

    if BANDAS_SUPERIORES != (
        ("+1,5σ", "mais_15"),
        ("+2σ", "mais_2"),
    ):
        raise RuntimeError("Bandas superiores direcionais foram alteradas indevidamente.")


validar_configuracao()


def numero_seguro(valor: Any) -> float:
    try:
        x = float(valor)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def lista_walls(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Lê Call/Put W1/W2/W3 sem reclassificar o ranking do GEX."""
    if metrics is None:
        return []
    saida: list[dict[str, Any]] = []
    for lado, chave in [("Call", "call_walls"), ("Put", "put_walls")]:
        for wall in metrics.get(chave, []) or []:
            strike = numero_seguro(wall.get("strike"))
            if not np.isfinite(strike):
                continue
            rank = int(wall.get("rank", 9))
            if rank not in (1, 2, 3):
                continue
            saida.append(
                {
                    "lado": lado,
                    "rank": rank,
                    "rotulo": f"{lado} W{rank}",
                    "strike": strike,
                    "gamma_1pct": numero_seguro(wall.get("gamma_1pct")),
                    "share_pct": numero_seguro(wall.get("share_pct")),
                    "dist_spot_pct": numero_seguro(wall.get("distance_pct")),
                }
            )
    return saida


def tolerancia_call_put_v3() -> float:
    valor = getattr(gex, "CONFLUENCIA_WALL_ATOL", None)
    if valor is None:
        raise RuntimeError("CONFLUENCIA_WALL_ATOL não está disponível no motor GEX.")
    return float(valor)


CONFLUENCIA_CALL_PUT_ATOL_V3 = tolerancia_call_put_v3()


def copiar_dados_wall_v3(wall):
    if wall is None:
        return None
    return {
        "lado": str(wall["lado"]),
        "rank": int(wall["rank"]),
        "rotulo": str(wall["rotulo"]),
        "strike": numero_seguro(wall["strike"]),
        "gamma_1pct": numero_seguro(wall["gamma_1pct"]),
        "share_pct": numero_seguro(wall["share_pct"]),
        "dist_spot_pct": numero_seguro(wall["dist_spot_pct"]),
    }


def construir_regiao_single_v3(wall, camada):
    w = copiar_dados_wall_v3(wall)
    if w is None or not np.isfinite(w["strike"]):
        return None
    lado = w["lado"]
    return {
        "Camada": camada,
        "Wall/Região GEX": w["rotulo"],
        "Rank Wall": int(w["rank"]),
        "Call/Put compartilhado": False,
        "Nível Wall/Região": w["strike"],
        "Call Wall": w["rotulo"] if lado == "Call" else "—",
        "Put Wall": w["rotulo"] if lado == "Put" else "—",
        "Nível Call": w["strike"] if lado == "Call" else np.nan,
        "Nível Put": w["strike"] if lado == "Put" else np.nan,
        "Gross Gamma Call": w["gamma_1pct"] if lado == "Call" else np.nan,
        "Gross Gamma Put": w["gamma_1pct"] if lado == "Put" else np.nan,
        "Participação Call %": w["share_pct"] if lado == "Call" else np.nan,
        "Participação Put %": w["share_pct"] if lado == "Put" else np.nan,
        "Gross Gamma Wall": w["gamma_1pct"],
        "Participação Wall %": w["share_pct"],
        "Distância Wall ao Spot %": w["dist_spot_pct"],
    }


def construir_regiao_call_put_v3(call_wall, put_wall, camada):
    call = copiar_dados_wall_v3(call_wall)
    put = copiar_dados_wall_v3(put_wall)
    if call is None or put is None:
        return None
    if call["lado"] != "Call" or put["lado"] != "Put":
        raise RuntimeError("Pareamento Call/Put inválido na V3.")
    if int(call["rank"]) != int(put["rank"]):
        raise RuntimeError("A V3 só agrupa Call/Put do mesmo rank.")
    if not (
        np.isfinite(call["strike"])
        and np.isfinite(put["strike"])
        and np.isclose(
            call["strike"],
            put["strike"],
            atol=CONFLUENCIA_CALL_PUT_ATOL_V3,
            rtol=0.0,
        )
    ):
        return None
    rank = int(call["rank"])
    nivel = float((call["strike"] + put["strike"]) / 2.0)
    return {
        "Camada": camada,
        "Wall/Região GEX": f"Call/Put W{rank}",
        "Rank Wall": rank,
        "Call/Put compartilhado": True,
        "Nível Wall/Região": nivel,
        "Call Wall": call["rotulo"],
        "Put Wall": put["rotulo"],
        "Nível Call": call["strike"],
        "Nível Put": put["strike"],
        "Gross Gamma Call": call["gamma_1pct"],
        "Gross Gamma Put": put["gamma_1pct"],
        "Participação Call %": call["share_pct"],
        "Participação Put %": put["share_pct"],
        "Gross Gamma Wall": np.nan,
        "Participação Wall %": np.nan,
        "Distância Wall ao Spot %": np.nan,
    }


def construir_regioes_gex_v3(metrics, ranks, camada):
    if metrics is None:
        return []
    ranks = {int(r) for r in ranks}
    walls = [
        copiar_dados_wall_v3(w)
        for w in lista_walls(metrics)
        if int(w["rank"]) in ranks
    ]
    walls = [w for w in walls if w is not None]
    regioes = []

    for rank in sorted(ranks):
        calls = [w for w in walls if w["lado"] == "Call" and int(w["rank"]) == rank]
        puts = [w for w in walls if w["lado"] == "Put" and int(w["rank"]) == rank]
        call = calls[0] if calls else None
        put = puts[0] if puts else None

        compartilhada = None
        if call is not None and put is not None:
            compartilhada = construir_regiao_call_put_v3(call, put, camada)

        if compartilhada is not None:
            regioes.append(compartilhada)
        else:
            if call is not None:
                reg = construir_regiao_single_v3(call, camada)
                if reg is not None:
                    regioes.append(reg)
            if put is not None:
                reg = construir_regiao_single_v3(put, camada)
                if reg is not None:
                    regioes.append(reg)

    return regioes


def construir_regioes_direcionais_v3(metrics, ranks, camada):
    """Constrói uma região por Wall, mantendo Call e Put separadas.

    A função original construir_regioes_gex_v3() continua intacta para
    preservar a lógica histórica de Call/Put compartilhada. Porém a nova
    regra de confluência é direcional, então a seleção precisa tratar cada
    lado separadamente:
    - Put -> bandas inferiores;
    - Call -> bandas superiores.
    """
    if metrics is None:
        return []

    ranks = {int(r) for r in ranks}
    regioes = []

    for wall in lista_walls(metrics):
        if int(wall["rank"]) not in ranks:
            continue

        regiao = construir_regiao_single_v3(wall, camada)
        if regiao is not None:
            regioes.append(regiao)

    def chave(regiao):
        rank = int(regiao.get("Rank Wall", 9))
        nome = str(regiao.get("Wall/Região GEX", ""))
        lado_ordem = 0 if nome.startswith("Put ") else 1
        return rank, lado_ordem, nome

    return sorted(regioes, key=chave)


def bandas_direcionais_para_regiao_v3(regiao):
    """Retorna somente as bandas permitidas para o lado da Wall."""
    nome = str(regiao.get("Wall/Região GEX", "") or "")

    if nome.startswith("Put "):
        return BANDAS_INFERIORES

    if nome.startswith("Call "):
        return BANDAS_SUPERIORES

    return ()


def metricas_zona_v3(preco_atual, banda, nivel_gex, spot_gex):
    """Mede a posição do preço atual em relação à zona Banda GARCH ↔ Wall GEX.

    Regras preservadas:
    - a zona continua sendo [min(banda, wall), max(banda, wall)];
    - a largura percentual da zona continua normalizada pelo Spot GEX, como antes;
    - somente as métricas de POSIÇÃO/DISTÂNCIA DO PREÇO passam a usar o Preço atual
      independente, conforme definido para esta versão.
    """
    preco_atual = numero_seguro(preco_atual)
    banda = numero_seguro(banda)
    nivel_gex = numero_seguro(nivel_gex)
    spot_gex = numero_seguro(spot_gex)

    vazio = {
        "Zona inferior": np.nan,
        "Zona superior": np.nan,
        "Centro da zona": np.nan,
        "Largura zona R$": np.nan,
        "Largura zona %": np.nan,
        "Preço atual dentro da zona": False,
        "Dist Preço→Zona R$": np.nan,
        "Dist Preço→Zona %": np.nan,
        "Dist Preço→Centro R$": np.nan,
        "Dist Preço→Centro %": np.nan,
        "Dist Preço→Centro % assinada": np.nan,
        "Posição da zona vs Preço": "SEM DADOS",
    }

    if not (np.isfinite(banda) and np.isfinite(nivel_gex)):
        return vazio

    inferior = float(min(banda, nivel_gex))
    superior = float(max(banda, nivel_gex))
    centro = float((inferior + superior) / 2.0)
    largura = float(superior - inferior)

    largura_pct = (
        float(largura / spot_gex * 100.0)
        if np.isfinite(spot_gex) and spot_gex > 0
        else np.nan
    )

    if not np.isfinite(preco_atual) or preco_atual <= 0:
        vazio.update(
            {
                "Zona inferior": inferior,
                "Zona superior": superior,
                "Centro da zona": centro,
                "Largura zona R$": largura,
                "Largura zona %": largura_pct,
            }
        )
        return vazio

    if inferior <= preco_atual <= superior:
        dentro, dist_zona, posicao = True, 0.0, "PREÇO ATUAL DENTRO DA ZONA"
    elif preco_atual < inferior:
        dentro, dist_zona, posicao = (
            False,
            float(inferior - preco_atual),
            "ZONA ACIMA DO PREÇO ATUAL",
        )
    else:
        dentro, dist_zona, posicao = (
            False,
            float(preco_atual - superior),
            "ZONA ABAIXO DO PREÇO ATUAL",
        )

    dist_centro_assinada = float((centro / preco_atual - 1.0) * 100.0)

    return {
        "Zona inferior": inferior,
        "Zona superior": superior,
        "Centro da zona": centro,
        "Largura zona R$": largura,
        "Largura zona %": largura_pct,
        "Preço atual dentro da zona": dentro,
        "Dist Preço→Zona R$": dist_zona,
        "Dist Preço→Zona %": float(dist_zona / preco_atual * 100.0),
        "Dist Preço→Centro R$": float(abs(centro - preco_atual)),
        "Dist Preço→Centro %": float(abs(dist_centro_assinada)),
        "Dist Preço→Centro % assinada": dist_centro_assinada,
        "Posição da zona vs Preço": posicao,
    }


def comparar_bandas_regioes_v3(
    ativo,
    bloco,
    bandas,
    metrics,
    preco_garch,
    preco_atual,
    fonte_preco_atual,
    momento_preco_atual,
    camada,
    ranks,
):
    """Compara somente os pareamentos direcionais válidos.

    Regras:
    - Put W1/W2/W3 compara somente com -1,5σ e -2σ;
    - Call W1/W2/W3 compara somente com +1,5σ e +2σ;
    - Principal continua somente W1;
    - Secundária continua somente W2/W3;
    - Confluência % continua |Wall-Banda| / Spot GEX × 100;
    - Distância do mercado à zona continua usando somente o Preço atual.
    """
    if bandas is None or metrics is None:
        return [], []

    spot = numero_seguro(metrics.get("spot"))
    if not np.isfinite(spot) or spot <= 0:
        return [], []

    preco_atual = numero_seguro(preco_atual)
    quality = metrics.get("quality") or {}
    quality_score = numero_seguro(quality.get("score"))
    quality_label = str(quality.get("label", "N/D"))

    regioes = construir_regioes_direcionais_v3(
        metrics,
        ranks=ranks,
        camada=camada,
    )
    registros_regiao, comparacoes = [], []

    for regiao in regioes:
        base_regiao = {
            "Ativo": ativo,
            "Bloco": bloco["bloco"],
            "GARCH período": bloco["garch_periodo"],
            "GEX horizonte": bloco["gex_horizonte"],
            "Preço atual": preco_atual,
            "Fonte preço atual": fonte_preco_atual,
            "Momento preço atual": momento_preco_atual,
            "Preço GARCH": numero_seguro(preco_garch),
            "Spot GEX": spot,
            "Camada": camada,
            **regiao,
            "Qualidade GEX": quality_score,
            "Classe qualidade": quality_label,
            "Séries GEX": int(metrics.get("series_count", 0)),
            "Vencimentos GEX": int(metrics.get("expiry_count", 0)),
        }
        registros_regiao.append(base_regiao)
        nivel_gex = numero_seguro(regiao["Nível Wall/Região"])

        bandas_validas = bandas_direcionais_para_regiao_v3(regiao)

        for banda_rotulo, banda_chave in bandas_validas:
            banda_nivel = numero_seguro(bandas.get(banda_chave))
            if not (np.isfinite(banda_nivel) and np.isfinite(nivel_gex)):
                continue

            diferenca_reais = float(abs(nivel_gex - banda_nivel))

            # REGRA V3 PRESERVADA: a proximidade estrutural Banda↔Wall continua
            # normalizada pelo Spot GEX. O Preço atual NÃO entra nesta fórmula.
            diferenca_pct = float(diferenca_reais / spot * 100.0)

            zona = metricas_zona_v3(
                preco_atual,
                banda_nivel,
                nivel_gex,
                spot,
            )

            comparacoes.append(
                {
                    **base_regiao,
                    "Banda GARCH": banda_rotulo,
                    "Nível banda": banda_nivel,
                    "Diferença Wall↔Banda R$": diferenca_reais,
                    "Diferença Wall↔Banda %": diferenca_pct,
                    **zona,
                }
            )

    return registros_regiao, comparacoes


def melhor_comparacao_v3(comparacoes):
    validas = [
        c for c in comparacoes
        if np.isfinite(numero_seguro(c.get("Diferença Wall↔Banda %")))
    ]
    if not validas:
        return None

    def chave(c):
        return (
            numero_seguro(c["Diferença Wall↔Banda %"]),
            int(c.get("Rank Wall", 9)),
            0 if bool(c.get("Call/Put compartilhado", False)) else 1,
            str(c.get("Wall/Região GEX", "")),
            str(c.get("Banda GARCH", "")),
        )
    return sorted(validas, key=chave)[0]


def resultado_anual_garch(preco, bandas):
    """Replica a leitura do GARCH Radar atual para o período ANUAL."""
    preco = numero_seguro(preco)
    if bandas is None or not np.isfinite(preco) or preco <= 0:
        return None
    try:
        status, proxima, distancia, prioridade = garch.analisar_status(preco, bandas)
    except Exception:
        return None
    mapa = {"+2σ": "mais_2", "+1,5σ": "mais_15", "-1,5σ": "menos_15", "-2σ": "menos_2"}
    chave = mapa.get(proxima)
    nivel = numero_seguro(bandas.get(chave)) if chave else np.nan
    return {
        "rotulo": proxima,
        "nivel": nivel,
        "dist_pct": numero_seguro(distancia),
        "status": status,
        "prioridade": prioridade,
    }


def calcular_garch_ativo(ativo: str, data_referencia: pd.Timestamp | None = None) -> dict[str, Any]:
    """Executa o mesmo motor GARCH para um ativo, sem Streamlit."""
    if data_referencia is None:
        data_referencia = garch.agora_local().normalize()
    historico = garch.baixar_historico_diario(ativo)
    preco_atual, intervalo, momento = garch.baixar_preco_mais_recente(ativo, historico)

    ajustes_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    bandas_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    bandas: dict[str, Any] = {}
    erros: dict[str, Any] = {}

    for periodo in ("MENSAL", "SEMESTRAL", "ANUAL"):
        try:
            bandas[periodo] = garch.calcular_bandas_periodo(
                ativo,
                historico,
                periodo,
                garch.inicio_periodo(periodo, data_referencia),
                ajustes_cache,
                bandas_cache,
            )
            erros[periodo] = None
        except Exception as exc:
            bandas[periodo] = None
            erros[periodo] = f"{type(exc).__name__}: {exc}"

    return {
        "ok": True,
        "preco": float(preco_atual),
        "intervalo": intervalo,
        "momento": momento,
        "bandas": bandas,
        "erros": erros,
    }


def linha_vazia(ativo, bloco, camada):
    return {
        "Ativo": ativo,
        "Bloco": bloco["bloco"],
        "GARCH período": bloco["garch_periodo"],
        "GEX horizonte": bloco["gex_horizonte"],
        "Camada": camada,
    }


def calcular_confluencia_ativo(
    ativo: str,
    garch_resultado: dict[str, Any],
    gex_reference_date: pd.Timestamp,
    preco_atual: float = np.nan,
    fonte_preco_atual: str | None = None,
    momento_preco_atual: Any = pd.NaT,
) -> dict[str, Any]:
    """Calcula a V3 de um único ativo B3.

    Três referências de preço ficam deliberadamente separadas:
    - Preço atual: cotação independente usada somente para distância/posição da zona;
    - Preço GARCH: cotação usada pela leitura do motor GARCH;
    - Spot GEX: referência interna da base GEX e denominador da confluência estrutural.
    """
    preco_garch = numero_seguro(garch_resultado.get("preco"))
    preco_atual = numero_seguro(preco_atual)

    retorno = {
        "Ativo": ativo,
        "Empresa": garch.ATIVOS[ativo]["nome"],
        "Setor": garch.ATIVOS[ativo]["setor"],
        "Preço atual": preco_atual,
        "Fonte preço atual": fonte_preco_atual or "N/D",
        "Momento preço atual": momento_preco_atual,
        "Preço GARCH": preco_garch,
        "Fonte preço GARCH": garch_resultado.get("intervalo"),
        "Momento preço GARCH": garch_resultado.get("momento", pd.NaT),
        "Data efetiva GEX": pd.Timestamp(gex_reference_date).normalize(),
        "blocos": {},
    }

    spot_ref = np.nan
    for h in ("30 dias", "90 dias", "180 dias"):
        _, m = gex.get_metrics(ativo, h)
        if m is not None and np.isfinite(numero_seguro(m.get("spot"))):
            spot_ref = numero_seguro(m.get("spot"))
            break

    retorno["Spot GEX"] = spot_ref
    retorno["Preço GARCH × Spot GEX · Dif %"] = (
        float((preco_garch / spot_ref - 1.0) * 100.0)
        if np.isfinite(preco_garch) and np.isfinite(spot_ref) and spot_ref > 0
        else np.nan
    )

    for bloco in BLOCOS_CONFLUENCIA:
        bandas = garch_resultado.get("bandas", {}).get(bloco["garch_periodo"])
        _, metrics = gex.get_metrics(ativo, bloco["gex_horizonte"])

        regs_p, comps_p = comparar_bandas_regioes_v3(
            ativo,
            bloco,
            bandas,
            metrics,
            preco_garch,
            preco_atual,
            fonte_preco_atual,
            momento_preco_atual,
            "PRINCIPAL W1",
            {1},
        )
        regs_s, comps_s = comparar_bandas_regioes_v3(
            ativo,
            bloco,
            bandas,
            metrics,
            preco_garch,
            preco_atual,
            fonte_preco_atual,
            momento_preco_atual,
            "SECUNDÁRIA W2/W3",
            {2, 3},
        )

        retorno["blocos"][bloco["bloco"]] = {
            "config": bloco,
            "bandas": bandas,
            "metrics": metrics,
            "regioes_principal": regs_p,
            "regioes_secundaria": regs_s,
            "comparacoes_principal": comps_p,
            "comparacoes_secundaria": comps_s,
            "principal": melhor_comparacao_v3(comps_p),
            "secundaria": melhor_comparacao_v3(comps_s),
        }

    retorno["anual"] = resultado_anual_garch(
        preco_garch,
        garch_resultado.get("bandas", {}).get("ANUAL"),
    )
    return retorno


def formatar_par(item) -> str:
    if not item:
        return "SEM DADOS"
    return f"{item['Banda GARCH']} × {item['Wall/Região GEX']}"


def linha_radar(ativo_resultado: dict[str, Any]) -> dict[str, Any]:
    """Linha compacta para a tabela principal do Streamlit."""
    row = {
        "Ativo": ativo_resultado["Ativo"],
        "Empresa": ativo_resultado["Empresa"],
        "Preço atual": ativo_resultado.get("Preço atual", np.nan),
        "Preço GARCH": ativo_resultado.get("Preço GARCH", np.nan),
        "Spot GEX": ativo_resultado.get("Spot GEX", np.nan),
    }

    mapa = [
        ("30D", "Mensal × 30D"),
        ("90D", "Semestral × 90D"),
        ("180D", "Semestral × 180D"),
    ]

    for curto, nome in mapa:
        bloco = ativo_resultado["blocos"].get(nome, {})
        p = bloco.get("principal")
        s = bloco.get("secundaria")

        row[f"{curto} · Principal"] = formatar_par(p)
        row[f"{curto} · Confluência %"] = (
            numero_seguro(p.get("Diferença Wall↔Banda %")) if p else np.nan
        )
        row[f"{curto} · Dist Preço→Zona %"] = (
            numero_seguro(p.get("Dist Preço→Zona %")) if p else np.nan
        )
        row[f"{curto} · Qualidade"] = (
            str(p.get("Classe qualidade", "N/D")) if p else "N/D"
        )
        row[f"{curto} · Secundária"] = formatar_par(s)
        row[f"{curto} · Sec %"] = (
            numero_seguro(s.get("Diferença Wall↔Banda %")) if s else np.nan
        )

    anual = ativo_resultado.get("anual")
    row["Anual · Banda"] = anual["rotulo"] if anual else "SEM DADOS"
    row["Anual · Dist %"] = anual["dist_pct"] if anual else np.nan
    row["Anual · Status"] = anual["status"] if anual else "SEM DADOS"

    # Apenas ordenação relativa do painel: menor principal disponível.
    principais = [
        row.get("30D · Confluência %"),
        row.get("90D · Confluência %"),
        row.get("180D · Confluência %"),
    ]
    principais = [x for x in principais if np.isfinite(numero_seguro(x))]
    row["_ordem_principal"] = min(principais) if principais else np.nan
    return row


def dataframe_radar(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [linha_radar(r) for r in resultados.values()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["_ordem_principal", "Ativo"], na_position="last").reset_index(drop=True)
    return df


def dataframe_detalhes(resultados: dict[str, dict[str, Any]], camada: str) -> pd.DataFrame:
    rows = []
    chave = "principal" if camada == "PRINCIPAL W1" else "secundaria"
    for ativo_res in resultados.values():
        for nome, bloco in ativo_res["blocos"].items():
            item = bloco.get(chave)
            if item:
                rows.append(item)
            else:
                rows.append(
                    {
                        "Ativo": ativo_res["Ativo"],
                        "Bloco": nome,
                        "Camada": camada,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty and "Diferença Wall↔Banda %" in df.columns:
        df = df.sort_values(
            ["Diferença Wall↔Banda %", "Ativo", "Bloco"],
            na_position="last",
        ).reset_index(drop=True)
    return df


def dataframe_todas_comparacoes(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ativo_res in resultados.values():
        for bloco in ativo_res["blocos"].values():
            rows.extend(bloco.get("comparacoes_principal", []))
            rows.extend(bloco.get("comparacoes_secundaria", []))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Camada", "Diferença Wall↔Banda %", "Rank Wall", "Ativo", "Bloco"],
            na_position="last",
        ).reset_index(drop=True)
    return df
