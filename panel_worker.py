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
from datetime import datetime
from pathlib import Path

import pandas as pd

CACHE_SCHEMA = 3
PROJECT_VERSION = "GARCH × GEX — CONFLUÊNCIA V3 — CLOUD ROBUSTA"


def mem_max_mb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


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
        for nome in ("gex_core.py", "garch_core.py", "confluence_core.py")
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
        f"Python={sys.version.split()[0]} | mem_max={mem_max_mb():.1f} MB"
    )

    # ------------------------------------------------------------------
    # ETAPA 1 — B3 / GEX
    # ------------------------------------------------------------------
    import gex_core as gex

    log("Executando pipeline B3/GEX com ingestão otimizada para memória...")
    raw_series, metadata = gex.run_full_pipeline(force=args.force_gex)

    log(
        f"Pipeline B3 concluído | raw_series={len(raw_series):,} | "
        f"base={metadata.get('reference_date')} | mem_max={mem_max_mb():.1f} MB"
    )

    # load_complete_bundle() faria este prepare_panel_data(), mas também
    # carregaria o COTAHIST. O painel conjunto não precisa do COTAHIST.
    prepared = gex.prepare_panel_data(raw_series)
    del raw_series
    gc.collect()

    log(
        f"GEX preparado | linhas={len(prepared):,} | "
        f"mem_max={mem_max_mb():.1f} MB"
    )

    instalar_runtime_gex_sem_copia(gex, prepared, metadata)
    gc.collect()

    log(
        "Runtime GEX instalado sem segunda preparação/cópia | "
        f"mem_max={mem_max_mb():.1f} MB"
    )

    # ------------------------------------------------------------------
    # ETAPA 2 — GARCH + CONFLUÊNCIA
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
        f"Calculando {len(ativos_comuns)} ativos GARCH × GEX | "
        f"mem_max={mem_max_mb():.1f} MB"
    )

    resultados = {}
    erros = {}

    for i, ativo in enumerate(ativos_comuns, start=1):
        log(
            f"GARCH {i:02d}/{len(ativos_comuns):02d} — {ativo} iniciando | "
            f"mem_max={mem_max_mb():.1f} MB"
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

        try:
            resultados[ativo] = core.calcular_confluencia_ativo(
                ativo,
                gd,
                gex_reference_date,
            )
        except Exception as exc:
            erros[ativo] = f"Confluência — {type(exc).__name__}: {exc}"
            log(f"{ativo} falhou na confluência: {erros[ativo]}")
        finally:
            del gd
            gc.collect()

        log(
            f"{ativo} concluído | resultados={len(resultados)} | "
            f"mem_max={mem_max_mb():.1f} MB"
        )

    if not resultados:
        raise RuntimeError(
            "Nenhum ativo B3 foi calculado com sucesso. "
            "Nenhum cache novo será gravado."
        )

    # Bitcoin permanece somente no GARCH Anual.
    btc = calcular_btc_anual(core, garch, data_ref)
    gc.collect()

    payload = {
        "cache_schema": CACHE_SCHEMA,
        "project_version": PROJECT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "gex_reference_date": gex_reference_date,
        "garch_data_reference": data_ref,
        "resultados": resultados,
        "btc": btc,
        "erros": erros,
        "core_hashes": core_hashes(module_dir),
        "worker_info": {
            "python": sys.version,
            "elapsed_seconds": time.time() - started,
            "mem_max_mb": mem_max_mb(),
            "ativos_comuns": len(ativos_comuns),
            "ativos_calculados": len(resultados),
            "force_gex": bool(args.force_gex),
            "cotahist_gex_carregado": False,
            "gex_prepare_repetido": False,
        },
    }

    salvar_pickle_atomico(payload, output)

    log(
        f"Cache final salvo: {output.name} | "
        f"ativos={len(resultados)} | "
        f"tempo={time.time() - started:.1f}s | "
        f"mem_max={mem_max_mb():.1f} MB"
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
