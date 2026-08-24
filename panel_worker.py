# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import hashlib
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_SCHEMA = 4
PROJECT_VERSION = "GARCH × GEX — CONFLUÊNCIA V3 — CLOUD ROBUSTA — PREÇO ATUAL"
TIMEZONE_LOCAL = "America/Sao_Paulo"


def mem_max_mb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def mem_current_mb() -> float:
    """RSS atual no Linux/Streamlit Cloud, sem dependência externa."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def mem_text() -> str:
    atual = mem_current_mb()
    pico = mem_max_mb()
    atual_txt = f"{atual:.1f} MB" if pd.notna(atual) else "N/D"
    pico_txt = f"{pico:.1f} MB" if pd.notna(pico) else "N/D"
    return f"mem_atual={atual_txt} | mem_pico={pico_txt}"


def log(msg: str) -> None:
    print(f"[V3 WORKER] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def core_hashes(module_dir: Path) -> dict[str, str]:
    return {
        nome: sha256_file(module_dir / nome)
        for nome in (
            "gex_core.py",
            "garch_core.py",
            "confluence_core.py",
            "panel_worker.py",
        )
    }


def instalar_runtime_gex_sem_copia(gex, prepared: pd.DataFrame, metadata: dict) -> None:
    """Instala a base já preparada sem chamar prepare_panel_data() outra vez.

    Equivale apenas à instalação de estado de initialize_runtime(), preservando
    matemática, parâmetros, recortes e caches do GEX.

    O painel conjunto não usa COTAHIST na matemática GEX × GARCH, portanto
    historical_prices fica vazio nesta aplicação específica.
    """
    gex.metadata = dict(metadata or {})
    gex.gex_series = prepared
    gex.REFERENCE_DATE = pd.Timestamp(
        gex.metadata.get("reference_date", gex.current_brazil_date())
    )
    gex.RISK_FREE_RATE = float(
        gex.metadata.get("risk_free_rate_assumption", gex.TAXA_LIVRE_RISCO_ANUAL)
    )
    gex.MAX_BASE_DAYS = int(
        gex.metadata.get("max_days_to_expiry", gex.MAX_DIAS_ATE_VENCIMENTO)
    )
    gex.ASSETS = gex.ATIVOS_B3.copy()
    gex.DISPLAY_ASSETS = gex.ATIVOS_EXIBICAO.copy()
    gex.historical_prices = pd.DataFrame()
    gex.invalidate_metrics_cache()


def salvar_pickle_atomico(payload: dict, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temp = destino.with_name(destino.name + ".part")
    temp.unlink(missing_ok=True)

    with temp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    with temp.open("rb") as f:
        check = pickle.load(f)

    if not isinstance(check, dict) or check.get("cache_schema") != CACHE_SCHEMA:
        temp.unlink(missing_ok=True)
        raise RuntimeError("Cache temporário inválido; cache anterior foi preservado.")

    temp.replace(destino)


def _normalizar_momento_mercado(valor):
    try:
        ts = pd.Timestamp(valor)
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is not None:
            ts = ts.tz_convert(TIMEZONE_LOCAL).tz_localize(None)
        return ts
    except Exception:
        return pd.NaT


def _serie_close_de_download(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Extrai Close de um yf.download multi ou single ticker sem assumir a ordem do MultiIndex."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)

    serie = None

    if isinstance(frame.columns, pd.MultiIndex):
        nivel_0 = {str(x) for x in frame.columns.get_level_values(0)}
        nivel_1 = {str(x) for x in frame.columns.get_level_values(1)}

        if ticker in nivel_0:
            sub = frame[ticker]
            if "Close" in sub.columns:
                serie = sub["Close"]
        elif ticker in nivel_1:
            sub = frame.xs(ticker, axis=1, level=1)
            if "Close" in sub.columns:
                serie = sub["Close"]
    elif "Close" in frame.columns:
        serie = frame["Close"]

    if serie is None:
        return pd.Series(dtype=float)

    if isinstance(serie, pd.DataFrame):
        if serie.shape[1] == 0:
            return pd.Series(dtype=float)
        serie = serie.iloc[:, 0]

    return pd.to_numeric(serie, errors="coerce").dropna()


def buscar_precos_atuais(ativos: list[str], garch) -> dict[str, dict]:
    """Busca uma cotação de mercado independente do Spot GEX e do Preço GARCH.

    Fonte: Yahoo Finance, em consulta separada da chamada usada pelo motor GARCH.
    Preferência: 1 minuto no dia atual; fallback 5 minutos em 5 dias.
    Se a consulta independente não existir, NÃO usamos Spot GEX nem Preço GARCH
    como substitutos: a distância Preço→Zona ficará N/D para aquele ativo.
    """
    import yfinance as yf

    ativos = list(dict.fromkeys(str(a).upper().strip() for a in ativos))
    ticker_por_ativo = {ativo: garch.ticker_yahoo(ativo) for ativo in ativos}

    resultados = {
        ativo: {
            "preco": np.nan,
            "fonte": "N/D",
            "momento": pd.NaT,
            "ticker": ticker,
        }
        for ativo, ticker in ticker_por_ativo.items()
    }
    pendentes = set(ativos)

    tentativas = [
        ("1d", "1m"),
        ("5d", "5m"),
    ]

    for periodo, intervalo in tentativas:
        if not pendentes:
            break

        tickers = [ticker_por_ativo[a] for a in sorted(pendentes)]
        log(
            f"Preço atual independente: Yahoo {intervalo} para {len(tickers)} ativo(s) | "
            f"{mem_text()}"
        )

        try:
            frame = yf.download(
                tickers=tickers,
                period=periodo,
                interval=intervalo,
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                prepost=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            log(
                f"Preço atual Yahoo {intervalo} falhou em lote: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        for ativo in list(pendentes):
            ticker = ticker_por_ativo[ativo]
            serie = _serie_close_de_download(frame, ticker)
            if serie.empty:
                continue

            preco = float(serie.iloc[-1])
            if not np.isfinite(preco) or preco <= 0:
                continue

            resultados[ativo] = {
                "preco": preco,
                "fonte": f"Yahoo Finance {intervalo}",
                "momento": _normalizar_momento_mercado(serie.index[-1]),
                "ticker": ticker,
            }
            pendentes.discard(ativo)

        del frame
        gc.collect()

    # Fallback individual somente para eventuais ausências do download em lote.
    for ativo in list(pendentes):
        ticker = ticker_por_ativo[ativo]

        for periodo, intervalo in tentativas:
            try:
                frame = yf.Ticker(ticker).history(
                    period=periodo,
                    interval=intervalo,
                    auto_adjust=False,
                    actions=False,
                    prepost=False,
                )
                if frame is None or frame.empty or "Close" not in frame.columns:
                    continue

                serie = pd.to_numeric(frame["Close"], errors="coerce").dropna()
                if serie.empty:
                    continue

                preco = float(serie.iloc[-1])
                if not np.isfinite(preco) or preco <= 0:
                    continue

                resultados[ativo] = {
                    "preco": preco,
                    "fonte": f"Yahoo Finance {intervalo}",
                    "momento": _normalizar_momento_mercado(serie.index[-1]),
                    "ticker": ticker,
                }
                pendentes.discard(ativo)
                break
            except Exception:
                continue

    obtidos = sum(
        1
        for info in resultados.values()
        if np.isfinite(float(info.get("preco", np.nan)))
        and float(info.get("preco", np.nan)) > 0
    )

    log(
        f"Preço atual independente concluído: {obtidos}/{len(ativos)} ativo(s) | "
        f"{mem_text()}"
    )

    if pendentes:
        log(
            "Preço atual independente indisponível para: "
            + ", ".join(sorted(pendentes))
            + ". A distância Preço→Zona ficará N/D nesses ativos."
        )

    return resultados


def calcular_btc_anual(core, garch, data_ref):
    if "BTC-USD" not in garch.ATIVOS:
        return None

    try:
        gd = core.calcular_garch_ativo("BTC-USD", data_ref)
    except Exception as exc:
        log(f"BTC-USD falhou: {type(exc).__name__}: {exc}")
        return None

    if not gd.get("ok"):
        return None

    anual = core.resultado_anual_garch(
        gd.get("preco"),
        gd.get("bandas", {}).get("ANUAL"),
    )

    return {
        "preco": gd.get("preco"),
        "anual": anual,
        "intervalo": gd.get("intervalo"),
        "momento": gd.get("momento"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-gex", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    started = time.time()

    log(
        f"Início | force_gex={args.force_gex} | "
        f"Python={sys.version.split()[0]} | {mem_text()}"
    )

    # ------------------------------------------------------------------
    # ETAPA 1 — B3 / GEX
    # ------------------------------------------------------------------
    import gex_core as gex

    log("Executando pipeline B3/GEX com ingestão otimizada para memória...")
    raw_series, metadata = gex.run_full_pipeline(force=args.force_gex)

    log(
        f"Pipeline B3 concluído | raw_series={len(raw_series):,} | "
        f"base={metadata.get('reference_date')} | {mem_text()}"
    )

    # load_complete_bundle() faria este prepare_panel_data(), mas também
    # carregaria o COTAHIST. O painel conjunto não precisa do COTAHIST.
    # O worker possui raw_series exclusivamente. Preparar no próprio DataFrame
    # evita manter raw_series + uma cópia completa simultaneamente.
    prepared = gex.prepare_panel_data(
        raw_series,
        copy_frame=False,
    )
    del raw_series
    gc.collect()

    log(
        f"GEX preparado | linhas={len(prepared):,} | "
        f"{mem_text()}"
    )

    instalar_runtime_gex_sem_copia(gex, prepared, metadata)
    del prepared
    gc.collect()

    log(
        "Runtime GEX instalado sem segunda preparação/cópia | "
        f"{mem_text()}"
    )

    # ------------------------------------------------------------------
    # ETAPA 2 — GARCH
    # ------------------------------------------------------------------
    import garch_core as garch
    import confluence_core as core

    data_ref = garch.agora_local().normalize()
    gex_reference_date = pd.Timestamp(metadata["reference_date"])

    ativos_comuns = [
        codigo
        for codigo in garch.ATIVOS.keys()
        if codigo != "BTC-USD" and codigo in set(gex.ASSETS)
    ]

    log(
        f"Calculando GARCH de {len(ativos_comuns)} ativos | "
        f"{mem_text()}"
    )

    garch_resultados = {}
    erros = {}

    for i, ativo in enumerate(ativos_comuns, start=1):
        log(
            f"GARCH {i:02d}/{len(ativos_comuns):02d} — {ativo} iniciando | "
            f"{mem_text()}"
        )

        try:
            gd = core.calcular_garch_ativo(ativo, data_ref)
        except Exception as exc:
            erros[ativo] = f"GARCH — {type(exc).__name__}: {exc}"
            log(f"{ativo} falhou no GARCH: {erros[ativo]}")
            gc.collect()
            continue

        if not gd.get("ok"):
            erros[ativo] = str(gd.get("erros", {}))
            log(f"{ativo} retornou GARCH sem sucesso: {erros[ativo]}")
            gc.collect()
            continue

        garch_resultados[ativo] = gd
        gc.collect()

        log(
            f"GARCH {ativo} concluído | válidos={len(garch_resultados)} | "
            f"{mem_text()}"
        )

    if not garch_resultados:
        raise RuntimeError(
            "Nenhum ativo B3 teve GARCH calculado com sucesso. "
            "Nenhum cache novo será gravado."
        )

    # ------------------------------------------------------------------
    # ETAPA 3 — PREÇO ATUAL INDEPENDENTE
    # ------------------------------------------------------------------
    # Este preço NÃO substitui Spot GEX nem Preço GARCH e NÃO entra na
    # confluência Banda↔Wall. Ele serve somente para localizar o mercado atual
    # em relação à zona da confluência e para exibição no painel/gráfico.
    precos_atuais = buscar_precos_atuais(
        list(garch_resultados.keys()),
        garch,
    )
    gc.collect()

    # ------------------------------------------------------------------
    # ETAPA 4 — CONFLUÊNCIA
    # ------------------------------------------------------------------
    resultados = {}

    for i, (ativo, gd) in enumerate(garch_resultados.items(), start=1):
        quote = precos_atuais.get(ativo, {})

        log(
            f"Confluência {i:02d}/{len(garch_resultados):02d} — {ativo} | "
            f"preço_atual={quote.get('preco', np.nan)} | {mem_text()}"
        )

        try:
            resultados[ativo] = core.calcular_confluencia_ativo(
                ativo,
                gd,
                gex_reference_date,
                preco_atual=quote.get("preco", np.nan),
                fonte_preco_atual=quote.get("fonte", "N/D"),
                momento_preco_atual=quote.get("momento", pd.NaT),
            )
        except Exception as exc:
            erros[ativo] = f"Confluência — {type(exc).__name__}: {exc}"
            log(f"{ativo} falhou na confluência: {erros[ativo]}")
        finally:
            # Cada ativo usa seu próprio conjunto de chains. O resultado de
            # confluência já foi construído; manter as chains no cache do GEX
            # até o fim dos ativos só aumenta RAM sem mudar nenhum valor.
            gex.invalidate_metrics_cache()
            gc.collect()

        log(
            f"{ativo} concluído | resultados={len(resultados)} | "
            f"{mem_text()}"
        )

    ativos_garch_validos = len(garch_resultados)

    del garch_resultados
    del precos_atuais
    gc.collect()

    if not resultados:
        raise RuntimeError(
            "Nenhum ativo B3 foi calculado com sucesso. "
            "Nenhum cache novo será gravado."
        )

    # Bitcoin permanece somente no GARCH Anual.
    btc = calcular_btc_anual(core, garch, data_ref)
    gc.collect()

    generated_at = pd.Timestamp.now(tz=TIMEZONE_LOCAL).isoformat()

    payload = {
        "cache_schema": CACHE_SCHEMA,
        "project_version": PROJECT_VERSION,
        "generated_at": generated_at,
        "gex_reference_date": gex_reference_date,
        "garch_data_reference": data_ref,
        "resultados": resultados,
        "btc": btc,
        "erros": erros,
        "core_hashes": core_hashes(module_dir),
        "worker_info": {
            "python": sys.version,
            "elapsed_seconds": time.time() - started,
            "mem_current_mb": mem_current_mb(),
            "mem_max_mb": mem_max_mb(),
            "ativos_comuns": len(ativos_comuns),
            "ativos_garch_validos": ativos_garch_validos,
            "ativos_calculados": len(resultados),
            "force_gex": bool(args.force_gex),
            "cotahist_gex_carregado": False,
            "gex_prepare_repetido": False,
            "preco_atual_independente": True,
            "fonte_preco_atual": "Yahoo Finance intraday; sem fallback para Spot GEX/Preço GARCH",
        },
    }

    salvar_pickle_atomico(payload, output)

    log(
        f"Cache final salvo: {output.name} | "
        f"ativos={len(resultados)} | "
        f"tempo={time.time() - started:.1f}s | "
        f"{mem_text()}"
    )
    log("CONCLUÍDO.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        log(f"ERRO FATAL: {type(exc).__name__}: {exc}")
        raise
