# -*- coding: utf-8 -*-
from __future__ import annotations

# ============================================================
# GEX RADAR BRASIL — NÚCLEO MATEMÁTICO / DADOS B3
# Baseado na V21 Multi-Horizonte validada no Google Colab.
# Este módulo não contém interface Streamlit nem Probability Engine.
# Recortes do radar: DTE 1–30 / 31–60 / 61–90 / 91–180 dias, sem sobreposição.
# ============================================================

import base64
import io
import json
import math
import os
import re
import shutil
import time
import warnings
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from lxml import etree
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# ======================================================================================
# 3) CONFIGURAÇÕES DO MOTOR INTEGRADO
# ======================================================================================

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import re
import time

import requests
from lxml import etree
from scipy.optimize import brentq

# Universo B3 monitorado.
# Mantemos BOVA11, que já fazia parte do GEX, e acrescentamos todos os ativos B3
# mostrados no painel GARCH. BTC-USD fica apenas na camada de exibição porque o
# motor GEX desta versão usa exclusivamente opções negociadas na B3.
ATIVOS_B3 = [
    "PSSA3",
    "BBSE3",
    "CXSE3",
    "BBAS3",
    "EGIE3",
    "ITSA4",
    "EQTL3",
    "ITUB4",
    "BBDC4",
    "CPFE3",
    "ABEV3",
    "CMIG4",
    "SBSP3",
    "CPLE3",
    "BPAC11",
    "VALE3",
    "B3SA3",
    "GGBR4",
    "PETR4",
    "WEGE3",
    "BOVA11",
]

ATIVOS_EXIBICAO = ATIVOS_B3 + [
    "BTC-USD",
]

# Mantemos o nome ATIVOS_PILOTO como alias interno para não alterar a matemática
# já validada em funções antigas que usam esse identificador.
ATIVOS_PILOTO = ATIVOS_B3.copy()

ASSET_INFO = {
    "PSSA3": {"empresa": "Porto", "setor": "Seguros"},
    "BBSE3": {"empresa": "BB Seguridade", "setor": "Seguros"},
    "CXSE3": {"empresa": "Caixa Seguridade", "setor": "Seguros"},
    "BBAS3": {"empresa": "Banco do Brasil", "setor": "Bancos"},
    "EGIE3": {"empresa": "Engie Brasil", "setor": "Energia"},
    "ITSA4": {"empresa": "Itaúsa PN", "setor": "Holding"},
    "EQTL3": {"empresa": "Equatorial Energia", "setor": "Energia"},
    "ITUB4": {"empresa": "Itaú Unibanco", "setor": "Bancos"},
    "BBDC4": {"empresa": "Bradesco PN", "setor": "Bancos"},
    "CPFE3": {"empresa": "CPFL Energia", "setor": "Energia"},
    "ABEV3": {"empresa": "Ambev", "setor": "Consumo"},
    "CMIG4": {"empresa": "Cemig PN", "setor": "Energia"},
    "SBSP3": {"empresa": "Sabesp", "setor": "Saneamento"},
    "CPLE3": {"empresa": "Copel", "setor": "Energia"},
    "BPAC11": {"empresa": "BTG Pactual", "setor": "Bancos"},
    "VALE3": {"empresa": "Vale", "setor": "Mineração"},
    "B3SA3": {"empresa": "B3", "setor": "Mercado Financeiro"},
    "GGBR4": {"empresa": "Gerdau PN", "setor": "Siderurgia"},
    "PETR4": {"empresa": "Petrobras PN", "setor": "Petróleo e Gás"},
    "WEGE3": {"empresa": "WEG", "setor": "Indústria"},
    "BOVA11": {"empresa": "BOVA11", "setor": "ETF"},
    "BTC-USD": {"empresa": "Bitcoin", "setor": "Criptoativos"},
}

# Procura automaticamente a última data em que Cadastro + PriceReport
# estiverem disponíveis na mesma sessão.
RETROCEDER_DIAS = 10

# Mantemos o mesmo universo matemático já validado nas Etapas 2 e 3.
MAX_DIAS_ATE_VENCIMENTO = 180
MONEYNESS_MINIMO = 0.50
MONEYNESS_MAXIMO = 1.50

# Hipótese plana já usada e validada no protótipo.
# Está concentrada em uma única configuração para futura troca por curva DI.
TAXA_LIVRE_RISCO_ANUAL = 0.1415

# Faixa de cenário do Gamma Flip.
FAIXA_FLIP_INFERIOR = 0.70
FAIXA_FLIP_SUPERIOR = 1.30
PONTOS_CURVA_FLIP = 201

# Walls.
# A tabela principal continua enxuta e mostra apenas a Wall principal.
# No detalhe, o radar mostra até três regiões distintas de concentração.
NUM_WALLS_DETALHE = 3

# Para não chamar três strikes quase colados de três Walls diferentes,
# exigimos separação mínima baseada na própria malha de strikes:
# duas vezes o espaçamento típico (percentil 75 dos intervalos positivos).
WALL_GAP_MULTIPLIER = 2.0

# Camada de leitura / triagem.
# São classificações de DISTÂNCIA, não sinais de compra ou venda.
PROXIMIDADE_EM_CIMA_PCT = 0.50
PROXIMIDADE_MUITO_PROXIMO_PCT = 1.00
PROXIMIDADE_PROXIMO_PCT = 2.00

# Call Wall principal e Put Wall principal são consideradas em confluência
# quando caem no mesmo centavo.
CONFLUENCIA_WALL_ATOL = 0.01

# Histórico COTAHIST usado no gráfico de preço.
# O gráfico é SINCRONIZADO ao horizonte GEX:
#   30 dias  -> 30 pregões no gráfico + GEX de opções com DTE de 1 a 30 dias
#   60 dias  -> 60 pregões no gráfico + GEX de opções com DTE de 31 a 60 dias
#   90 dias  -> 90 pregões no gráfico + GEX de opções com DTE de 61 a 90 dias
#   180 dias -> 180 pregões no gráfico + GEX de opções com DTE de 91 a 180 dias
# Os quatro recortes são independentes, não cumulativos e não se sobrepõem.
MAX_HISTORICO_PREGOES = MAX_DIAS_ATE_VENCIMENTO

# Normalmente deixe None. Serve apenas para auditoria histórica manual.
DATA_REFERENCIA_MANUAL = None

# Cache local do Colab: rápido para extração e cálculo.
MODULE_DIR = Path(__file__).resolve().parent
INTEGRATED_DIR = Path(
    os.environ.get(
        "GEX_CACHE_DIR",
        str(MODULE_DIR / ".gex_cache" / "gex_radar_brasil_integrado"),
    )
).resolve()
RAW_CACHE_DIR = INTEGRATED_DIR / "raw"
WORK_DIR = INTEGRATED_DIR / "work"
HISTORY_CACHE_DIR = INTEGRATED_DIR / "historico_precos"

RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# 4) DOWNLOAD E LEITURA DOS ARQUIVOS PÚBLICOS DA B3
# ======================================================================================

def current_brazil_date():
    try:
        return datetime.now(
            ZoneInfo("America/Sao_Paulo")
        ).date()
    except Exception:
        return date.today()


def valid_zip_file(path):
    path = Path(path)
    return (
        path.exists()
        and path.stat().st_size >= 50
        and zipfile.is_zipfile(path)
    )


def make_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 GEXRadarBrasil/1.0",
            "Accept": "*/*",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
        }
    )
    return session


def download_pregao(
    session,
    filename,
    destination,
    force=False,
):
    """Baixa arquivo do Pesquisa por Pregão com cache local atômico.

    Um download novo só substitui o cache depois de ser validado como ZIP.
    Assim, uma falha de rede não destrói uma cópia válida já existente
    durante a sessão atual do Colab.
    """
    destination = Path(destination)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached_is_valid = valid_zip_file(
        destination
    )

    if (
        not force
        and cached_is_valid
    ):
        return (
            True,
            f"CACHE — {destination.stat().st_size / 1024 / 1024:.1f} MB",
        )

    temporary = destination.with_name(
        destination.name + ".part"
    )
    temporary.unlink(missing_ok=True)

    url = (
        "https://www.b3.com.br/"
        "pesquisapregao/download?filelist="
        f"{filename}"
    )

    try:
        with session.get(
            url,
            timeout=(20, 180),
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            with temporary.open("wb") as output:
                for chunk in response.iter_content(
                    1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)

        if (
            not temporary.exists()
            or temporary.stat().st_size < 50
        ):
            temporary.unlink(missing_ok=True)
            if cached_is_valid:
                return (
                    True,
                    "CACHE PRESERVADO — nova resposta vazia",
                )
            return False, "arquivo vazio"

        with temporary.open("rb") as handle:
            head = handle.read(2048).lstrip().lower()

        if (
            head.startswith(b"<html")
            or head.startswith(b"<!doctype html")
            or b"captcha" in head
            or b"access denied" in head
        ):
            temporary.unlink(missing_ok=True)
            if cached_is_valid:
                return (
                    True,
                    "CACHE PRESERVADO — nova resposta HTML/bloqueio",
                )
            return False, "resposta HTML/bloqueio"

        if not zipfile.is_zipfile(
            temporary
        ):
            temporary.unlink(missing_ok=True)
            if cached_is_valid:
                return (
                    True,
                    "CACHE PRESERVADO — nova resposta não é ZIP válido",
                )
            return False, "resposta não é ZIP válido"

        temporary.replace(destination)

        return (
            True,
            f"OK — {destination.stat().st_size / 1024 / 1024:.1f} MB",
        )

    except Exception as exc:
        temporary.unlink(missing_ok=True)

        if cached_is_valid:
            return (
                True,
                "CACHE PRESERVADO — "
                f"falha na atualização ({type(exc).__name__})",
            )

        return False, f"{type(exc).__name__}: {exc}"


def extract_recursive(
    archive,
    destination,
    depth=0,
):
    if depth > 5:
        return []

    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zipfile.is_zipfile(archive):
        subdir = destination / (
            f"{archive.stem}_extraido"
        )
        subdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extracted = []

        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                safe_name = Path(
                    info.filename
                ).name

                if not safe_name:
                    continue

                target = subdir / safe_name

                with (
                    zf.open(info) as source,
                    target.open("wb") as output,
                ):
                    shutil.copyfileobj(
                        source,
                        output,
                    )

                extracted.extend(
                    extract_recursive(
                        target,
                        subdir,
                        depth + 1,
                    )
                )

        return extracted

    return [archive]


def xml_creation_time(path):
    path = Path(path)
    text = (
        path.read_bytes()[:20000]
        .decode("utf-8", errors="ignore")
    )

    match = re.search(
        r"<CreDtAndTm>(.*?)</CreDtAndTm>",
        text,
    )

    if match:
        try:
            return datetime.fromisoformat(
                match.group(1).replace(
                    "Z",
                    "+00:00",
                )
            ).replace(tzinfo=None)
        except Exception:
            pass

    return datetime.fromtimestamp(
        path.stat().st_mtime
    )


def choose_latest_xml(
    files,
    message_name,
):
    xmls = [
        Path(path)
        for path in files
        if Path(path).suffix.lower() == ".xml"
    ]

    if not xmls:
        raise FileNotFoundError(
            f"Nenhum XML encontrado para {message_name}."
        )

    return max(
        xmls,
        key=xml_creation_time,
    )


def choose_reference_text(files):
    """Seleciona o arquivo de Prêmio de Referência extraído da sessão.

    O PE é distribuído pela B3 em contêiner ZIP autoextraível (.ex_).
    Depois da extração, o motor validado espera um TXT/CSV. A seleção
    continua sendo pelo maior arquivo, como na V21, mas agora a ausência
    do conteúdo esperado é tratada antes de aceitar a data como completa.
    """
    candidates = [
        Path(path)
        for path in files
        if Path(path).suffix.lower() in {".txt", ".csv"}
    ]

    if not candidates:
        raise FileNotFoundError(
            "Nenhum TXT/CSV encontrado para Prêmio de Referência."
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_size,
    )


def extract_and_validate_session(selected_paths, candidate_date):
    """Extrai e valida semanticamente uma sessão antes de aceitá-la.

    Antes desta correção, download_pregao() validava apenas se IN/PR/PE
    eram contêineres ZIP válidos. A B3 pode disponibilizar, durante a
    formação do fechamento, um ZIP tecnicamente válido mas ainda sem o
    XML/TXT esperado pelo motor. Nesse caso a data era aceita e o erro
    aparecia depois em choose_latest_xml().

    Agora uma data só é considerada COMPLETA quando:
    - IN contém pelo menos um XML de Cadastro de Instrumentos;
    - PR contém pelo menos um XML de PriceReport;
    - PE contém pelo menos um TXT/CSV de Prêmio de Referência.

    Se qualquer conteúdo estiver ausente, a data é rejeitada e o motor
    continua retrocedendo, sem alterar a matemática de IV/Gamma/GEX.
    """
    candidate_work_dir = WORK_DIR / candidate_date.isoformat()

    if candidate_work_dir.exists():
        shutil.rmtree(candidate_work_dir)

    candidate_work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted = {}

    for source_name, path in selected_paths.items():
        path = Path(path)

        if not valid_zip_file(path):
            raise FileNotFoundError(
                f"Arquivo {path.name} não é um ZIP válido para a sessão {candidate_date}."
            )

        print(f"  Validando conteúdo de {path.name}...")
        extracted[source_name] = extract_recursive(
            path,
            candidate_work_dir / source_name,
        )

    instrument_xml = choose_latest_xml(
        extracted.get("instruments", []),
        "Cadastro de Instrumentos",
    )
    price_xml = choose_latest_xml(
        extracted.get("prices", []),
        "PriceReport",
    )
    reference_txt = choose_reference_text(
        extracted.get("reference", []),
    )

    return {
        "work_dir": candidate_work_dir,
        "extracted": extracted,
        "instrument_xml": instrument_xml,
        "price_xml": price_xml,
        "reference_txt": reference_txt,
    }


def invalidate_semantically_bad_cache(path):
    """Remove apenas um cache ZIP que passou na estrutura mas falhou no conteúdo.

    Isso evita que um ZIP incompleto, porém tecnicamente válido, seja reutilizado
    indefinidamente na mesma instância do Streamlit. Um download válido posterior
    poderá recriar o arquivo normalmente.
    """
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass



# ======================================================================================
# 4.1) HISTÓRICO DE PREÇOS B3 — COTAHIST
# ======================================================================================

def download_cotahist_year(
    session,
    year,
    force=False,
):
    """Baixa a série anual COTAHIST da B3 com cache seguro.

    O histórico serve apenas para o gráfico de preço do ativo.
    Uma falha nessa fonte NÃO interrompe o motor GEX.
    """
    year = int(year)

    destination = (
        HISTORY_CACHE_DIR
        / f"COTAHIST_A{year}.ZIP"
    )

    cached_is_valid = valid_zip_file(
        destination
    )

    if (
        cached_is_valid
        and not force
    ):
        return destination

    temporary = destination.with_name(
        destination.name + ".part"
    )
    temporary.unlink(
        missing_ok=True
    )

    filename = (
        f"COTAHIST_A{year}.ZIP"
    )

    urls = [
        (
            "https://bvmf.bmfbovespa.com.br/"
            "InstDados/SerHist/"
            f"{filename}"
        ),
        (
            "http://bvmf.bmfbovespa.com.br/"
            "InstDados/SerHist/"
            f"{filename}"
        ),
    ]

    last_error = None

    for url in urls:
        try:
            with session.get(
                url,
                timeout=(20, 240),
                stream=True,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                with temporary.open(
                    "wb"
                ) as output:
                    for chunk in response.iter_content(
                        1024 * 1024
                    ):
                        if chunk:
                            output.write(
                                chunk
                            )

            if (
                temporary.exists()
                and temporary.stat().st_size >= 50
                and zipfile.is_zipfile(
                    temporary
                )
            ):
                temporary.replace(
                    destination
                )

                return destination

            temporary.unlink(
                missing_ok=True
            )

            last_error = (
                "resposta não é ZIP válido"
            )

        except Exception as exc:
            temporary.unlink(
                missing_ok=True
            )

            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

    if cached_is_valid:
        return destination

    raise RuntimeError(
        "Não foi possível baixar "
        f"{filename}: {last_error}"
    )


def parse_cotahist_assets(
    zip_path,
    assets,
):
    """Lê apenas os ativos necessários no COTAHIST.

    Layout posicional oficial:
    DATA, CODNEG, TPMERC e preços OHLC.
    Para o gráfico, usamos mercado à vista (TPMERC=010).
    """
    assets = set(
        str(asset).strip()
        for asset in assets
    )

    rows = []

    with zipfile.ZipFile(
        zip_path
    ) as archive:
        txt_names = [
            name
            for name in archive.namelist()
            if name.upper().endswith(
                ".TXT"
            )
        ]

        if not txt_names:
            raise RuntimeError(
                "COTAHIST sem arquivo TXT."
            )

        txt_name = txt_names[0]

        with archive.open(
            txt_name
        ) as raw:
            for raw_line in raw:
                line = raw_line.decode(
                    "latin-1",
                    errors="ignore",
                )

                if len(line) < 245:
                    continue

                # 00 = cabeçalho; 99 = trailer; 01 = registro de cotação.
                if line[0:2] != "01":
                    continue

                ticker = (
                    line[12:24]
                    .strip()
                )

                if ticker not in assets:
                    continue

                market_type = (
                    line[24:27]
                    .strip()
                )

                # 010 = mercado à vista.
                if market_type != "010":
                    continue

                try:
                    trade_date = (
                        pd.to_datetime(
                            line[2:10],
                            format="%Y%m%d",
                            errors="raise",
                        )
                    )

                    open_price = (
                        int(
                            line[56:69]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                    high_price = (
                        int(
                            line[69:82]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                    low_price = (
                        int(
                            line[82:95]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                    average_price = (
                        int(
                            line[95:108]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                    close_price = (
                        int(
                            line[108:121]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                    trades = int(
                        line[147:152]
                        .strip()
                        or 0
                    )

                    quantity = int(
                        line[152:170]
                        .strip()
                        or 0
                    )

                    financial_volume = (
                        int(
                            line[170:188]
                            .strip()
                            or 0
                        )
                        / 100.0
                    )

                except Exception:
                    continue

                if close_price <= 0:
                    continue

                rows.append(
                    {
                        "date": trade_date,
                        "ticker": ticker,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "average": average_price,
                        "close": close_price,
                        "trades": trades,
                        "quantity": quantity,
                        "financial_volume": financial_volume,
                        "source": "B3_COTAHIST",
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "average",
                "close",
                "trades",
                "quantity",
                "financial_volume",
                "source",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .drop_duplicates(
            [
                "ticker",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


def load_b3_price_history(
    reference_date,
    assets,
):
    """Carrega histórico B3 para o gráfico, sem tornar o painel dependente dele."""
    reference_date = pd.Timestamp(
        reference_date
    )

    session = make_session()

    frames = []

    current_year = int(
        reference_date.year
    )

    years = [
        current_year
    ]

    try:
        current_zip = (
            download_cotahist_year(
                session,
                current_year,
                force=False,
            )
        )

        current_frame = (
            parse_cotahist_assets(
                current_zip,
                assets,
            )
        )

        # O arquivo do ano corrente é cumulativo.
        # Se o cache ainda não alcançou a data efetiva do painel,
        # tentamos renová-lo uma vez.
        if not current_frame.empty:
            max_date = (
                current_frame[
                    "date"
                ].max()
            )

            if (
                pd.notna(max_date)
                and max_date.normalize()
                < reference_date.normalize()
            ):
                current_zip = (
                    download_cotahist_year(
                        session,
                        current_year,
                        force=True,
                    )
                )

                current_frame = (
                    parse_cotahist_assets(
                        current_zip,
                        assets,
                    )
                )

        if not current_frame.empty:
            frames.append(
                current_frame
            )

    except Exception as exc:
        print(
            "Histórico COTAHIST do ano corrente "
            f"indisponível: {type(exc).__name__}: {exc}"
        )

    # Se ainda não houver histórico suficiente para o maior horizonte
    # do painel (atualmente 180 pregões), complementamos com o ano anterior.
    need_previous_year = True

    if frames:
        combined_current = pd.concat(
            frames,
            ignore_index=True,
        )

        counts = (
            combined_current[
                combined_current[
                    "date"
                ].le(
                    reference_date
                )
            ]
            .groupby(
                "ticker"
            )
            .size()
        )

        need_previous_year = any(
            int(
                counts.get(
                    asset,
                    0,
                )
            )
            < MAX_HISTORICO_PREGOES
            for asset in assets
        )

    if need_previous_year:
        previous_year = (
            current_year - 1
        )

        years.append(
            previous_year
        )

        try:
            previous_zip = (
                download_cotahist_year(
                    session,
                    previous_year,
                    force=False,
                )
            )

            previous_frame = (
                parse_cotahist_assets(
                    previous_zip,
                    assets,
                )
            )

            if not previous_frame.empty:
                frames.append(
                    previous_frame
                )

        except Exception as exc:
            print(
                "Histórico COTAHIST do ano anterior "
                f"indisponível: {type(exc).__name__}: {exc}"
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "average",
                "close",
                "trades",
                "quantity",
                "financial_volume",
                "source",
            ]
        )

    history = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .query(
            "date <= @reference_date"
        )
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .drop_duplicates(
            [
                "ticker",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return history


def child_text(
    element,
    path,
):
    node = element.find(path)

    if (
        node is not None
        and node.text
    ):
        return node.text.strip()

    return None


def parse_instruments(path):
    """Lê BVBG.028.02 e mapeia opções ao ativo-objeto."""
    ns = "{urn:bvmf.100.02.xsd}"
    option_records = []
    instrument_records = []

    started = time.time()
    count = 0

    context = etree.iterparse(
        str(path),
        events=("end",),
        tag=ns + "Instrm",
        huge_tree=True,
        recover=True,
    )

    for _, element in context:
        count += 1

        instrument_id = child_text(
            element,
            f"./{ns}FinInstrmId/{ns}OthrId/{ns}Id",
        )

        asset = child_text(
            element,
            f"./{ns}FinInstrmAttrCmon/{ns}Asst",
        )

        asset_description = child_text(
            element,
            f"./{ns}FinInstrmAttrCmon/{ns}AsstDesc",
        )

        ticker_node = element.find(
            f".//{ns}TckrSymb"
        )
        isin_node = element.find(
            f".//{ns}ISIN"
        )

        ticker = (
            ticker_node.text.strip()
            if ticker_node is not None
            and ticker_node.text
            else None
        )

        isin = (
            isin_node.text.strip()
            if isin_node is not None
            and isin_node.text
            else None
        )

        if instrument_id and ticker:
            instrument_records.append(
                {
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "asset": asset,
                    "asset_description": asset_description,
                    "isin": isin,
                }
            )

        option = element.find(
            f".//{ns}OptnOnEqtsInf"
        )
        record_type = "OptnOnEqtsInf"

        if option is None:
            option = element.find(
                f".//{ns}OptnOnSpotAndFutrsInf"
            )
            record_type = "OptnOnSpotAndFutrsInf"

        if option is not None and ticker:
            def opt_text(name):
                node = option.find(
                    f".//{ns}{name}"
                )
                return (
                    node.text.strip()
                    if node is not None
                    and node.text
                    else None
                )

            underlying_id = child_text(
                option,
                f"./{ns}UndrlygInstrmId/{ns}OthrId/{ns}Id",
            )

            option_records.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": ticker,
                    "isin": isin,
                    "asset": asset,
                    "asset_description": asset_description,
                    "option_type": opt_text("OptnTp"),
                    "option_style": (
                        opt_text("OptnStyle")
                        or opt_text("ExrcStyle")
                    ),
                    "strike": opt_text("ExrcPric"),
                    "maturity_date": opt_text("XprtnDt"),
                    "trading_start_date": opt_text("TradgStartDt"),
                    "trading_end_date": opt_text("TradgEndDt"),
                    # AllcnRndLot é lote de alocação; não é
                    # multiplicador econômico do GEX.
                    "contract_size": opt_text("AllcnRndLot"),
                    "price_factor": opt_text("PricFctr"),
                    "underlying_id": underlying_id,
                    "record_type": record_type,
                }
            )

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]

        if count % 50000 == 0:
            print(
                f"    Cadastro: {count:,} instrumentos..."
            )

    del context

    instruments = (
        pd.DataFrame(instrument_records)
        .drop_duplicates(
            "instrument_id",
            keep="last",
        )
    )

    options = (
        pd.DataFrame(option_records)
        .drop_duplicates(
            "instrument_id",
            keep="last",
        )
    )

    ticker_map = instruments.set_index(
        "instrument_id"
    )["ticker"]

    options["underlying_ticker"] = (
        options["underlying_id"]
        .map(ticker_map)
    )

    for column in (
        "strike",
        "contract_size",
        "price_factor",
    ):
        options[column] = pd.to_numeric(
            options[column],
            errors="coerce",
        )

    for column in (
        "maturity_date",
        "trading_start_date",
        "trading_end_date",
    ):
        options[column] = pd.to_datetime(
            options[column],
            errors="coerce",
        )

    options = options[
        options[
            "underlying_ticker"
        ].isin(ATIVOS_PILOTO)
    ].copy()

    print(
        f"    Cadastro concluído: {len(instruments):,} instrumentos; "
        f"{len(options):,} opções dos ativos B3 monitorados; "
        f"{time.time() - started:.1f}s."
    )

    return (
        instruments.reset_index(drop=True),
        options.reset_index(drop=True),
    )


def parse_price_report(path):
    """Lê BVBG.086.01 PriceReport: preço, OI, bid/ask e negociação."""
    ns = "{urn:bvmf.217.01.xsd}"
    records = []
    started = time.time()
    count = 0

    context = etree.iterparse(
        str(path),
        events=("end",),
        tag=ns + "PricRpt",
        huge_tree=True,
        recover=True,
    )

    for _, element in context:
        count += 1

        def value(name):
            node = element.find(
                f".//{ns}{name}"
            )
            return (
                node.text.strip()
                if node is not None
                and node.text
                else None
            )

        instrument_id = child_text(
            element,
            f"./{ns}FinInstrmId/{ns}OthrId/{ns}Id",
        )

        records.append(
            {
                "instrument_id": instrument_id,
                "price_symbol": value("TckrSymb"),
                "price_date": value("Dt"),
                "open_interest": value("OpnIntrst"),
                "best_bid": value("BestBidPric"),
                "best_ask": value("BestAskPric"),
                "first_price": value("FrstPric"),
                "minimum_price": value("MinPric"),
                "maximum_price": value("MaxPric"),
                "average_price": value("TradAvrgPric"),
                "last_price": value("LastPric"),
                "trade_count": (
                    value("RglrTxsQty")
                    or value("TradQty")
                ),
                "traded_quantity": (
                    value("RglrTraddCtrcts")
                    or value("FinInstrmQty")
                ),
                "financial_volume": (
                    value("NtlRglrVol")
                    or value("NtlFinVol")
                ),
            }
        )

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]

        if count % 20000 == 0:
            print(
                f"    PriceReport: {count:,} registros..."
            )

    del context

    prices = pd.DataFrame(records)

    numeric_columns = [
        "open_interest",
        "best_bid",
        "best_ask",
        "first_price",
        "minimum_price",
        "maximum_price",
        "average_price",
        "last_price",
        "trade_count",
        "traded_quantity",
        "financial_volume",
    ]

    for column in numeric_columns:
        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce",
        )

    prices["price_date"] = pd.to_datetime(
        prices["price_date"],
        errors="coerce",
    )

    prices["data_points"] = (
        prices[numeric_columns]
        .notna()
        .sum(axis=1)
    )

    prices = (
        prices.sort_values(
            [
                "instrument_id",
                "data_points",
                "financial_volume",
            ],
            na_position="first",
        )
        .drop_duplicates(
            "instrument_id",
            keep="last",
        )
        .drop(
            columns="data_points"
        )
        .reset_index(drop=True)
    )

    print(
        f"    PriceReport concluído: {len(prices):,} instrumentos; "
        f"{time.time() - started:.1f}s."
    )

    return prices


def parse_reference_premium(path):
    columns = [
        "symbol",
        "reference_option_type",
        "reference_option_style",
        "reference_maturity_date",
        "reference_strike",
        "theoretic_price",
        "published_volatility",
    ]

    if path is None or not Path(path).exists():
        return pd.DataFrame(
            columns=columns
        )

    frame = pd.read_csv(
        path,
        sep=";",
        skiprows=1,
        names=columns,
        dtype={
            "symbol": "string",
            "reference_option_type": "string",
            "reference_option_style": "string",
            "reference_maturity_date": "string",
        },
        engine="python",
    )

    for column in (
        "reference_strike",
        "theoretic_price",
        "published_volatility",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["reference_maturity_date"] = pd.to_datetime(
        frame["reference_maturity_date"],
        format="%Y%m%d",
        errors="coerce",
    )

    return (
        frame.drop_duplicates(
            "symbol",
            keep="last",
        )
        .reset_index(drop=True)
    )


def build_market_base(
    options,
    prices,
    reference,
    selected_date,
):
    option_prices = prices.rename(
        columns={
            "price_symbol": "option_price_symbol",
            "price_date": "option_price_date",
            "best_bid": "option_best_bid",
            "best_ask": "option_best_ask",
            "first_price": "option_first_price",
            "minimum_price": "option_minimum_price",
            "maximum_price": "option_maximum_price",
            "average_price": "option_average_price",
            "last_price": "option_last_price",
            "trade_count": "option_trade_count",
            "traded_quantity": "option_traded_quantity",
            "financial_volume": "option_financial_volume",
        }
    )

    merged = options.merge(
        option_prices,
        on="instrument_id",
        how="left",
        validate="1:1",
    )

    spot_prices = prices.rename(
        columns={
            "instrument_id": "underlying_id",
            "price_symbol": "spot_symbol",
            "price_date": "spot_price_date",
            "open_interest": "spot_open_interest",
            "best_bid": "spot_best_bid",
            "best_ask": "spot_best_ask",
            "first_price": "spot_first_price",
            "minimum_price": "spot_minimum_price",
            "maximum_price": "spot_maximum_price",
            "average_price": "spot_average_price",
            "last_price": "spot_last_price",
            "trade_count": "spot_trade_count",
            "traded_quantity": "spot_traded_quantity",
            "financial_volume": "spot_financial_volume",
        }
    )

    merged = merged.merge(
        spot_prices,
        on="underlying_id",
        how="left",
        validate="m:1",
    )

    merged = merged.merge(
        reference,
        on="symbol",
        how="left",
        validate="1:1",
    )

    valid_mid = (
        merged["option_best_bid"].gt(0)
        & merged["option_best_ask"].gt(
            merged["option_best_bid"]
        )
    )

    merged["selected_option_price"] = np.nan
    merged["option_price_source"] = pd.NA

    merged.loc[
        valid_mid,
        "selected_option_price",
    ] = (
        merged.loc[
            valid_mid,
            "option_best_bid",
        ]
        + merged.loc[
            valid_mid,
            "option_best_ask",
        ]
    ) / 2.0

    merged.loc[
        valid_mid,
        "option_price_source",
    ] = "mid_bid_ask"

    mask = (
        merged["selected_option_price"].isna()
        & merged["option_last_price"].gt(0)
    )
    merged.loc[
        mask,
        "selected_option_price",
    ] = merged.loc[
        mask,
        "option_last_price",
    ]
    merged.loc[
        mask,
        "option_price_source",
    ] = "ultimo_negocio"

    mask = (
        merged["selected_option_price"].isna()
        & merged["option_average_price"].gt(0)
    )
    merged.loc[
        mask,
        "selected_option_price",
    ] = merged.loc[
        mask,
        "option_average_price",
    ]
    merged.loc[
        mask,
        "option_price_source",
    ] = "preco_medio"

    mask = (
        merged["selected_option_price"].isna()
        & merged["theoretic_price"].gt(0)
    )
    merged.loc[
        mask,
        "selected_option_price",
    ] = merged.loc[
        mask,
        "theoretic_price",
    ]
    merged.loc[
        mask,
        "option_price_source",
    ] = "premio_referencia_b3"

    merged["selected_spot_price"] = (
        merged["spot_last_price"]
    )

    mask = (
        merged["selected_spot_price"].isna()
        & merged["spot_average_price"].gt(0)
    )
    merged.loc[
        mask,
        "selected_spot_price",
    ] = merged.loc[
        mask,
        "spot_average_price",
    ]

    spot_mid_valid = (
        merged["spot_best_bid"].gt(0)
        & merged["spot_best_ask"].gt(
            merged["spot_best_bid"]
        )
    )

    mask = (
        merged["selected_spot_price"].isna()
        & spot_mid_valid
    )

    merged.loc[
        mask,
        "selected_spot_price",
    ] = (
        merged.loc[
            mask,
            "spot_best_bid",
        ]
        + merged.loc[
            mask,
            "spot_best_ask",
        ]
    ) / 2.0

    selected_ts = pd.Timestamp(
        selected_date
    )

    merged["is_not_expired"] = (
        merged["maturity_date"].notna()
        & merged["maturity_date"].ge(
            selected_ts
        )
    )
    merged["has_price_report"] = (
        merged["option_price_date"].notna()
    )
    merged["has_positive_open_interest"] = (
        merged["open_interest"]
        .fillna(0)
        .gt(0)
    )
    merged["has_trading"] = (
        merged["option_traded_quantity"]
        .fillna(0)
        .gt(0)
    )
    merged["has_reference_premium"] = (
        merged["theoretic_price"].notna()
    )
    merged["has_spot_price"] = (
        merged["selected_spot_price"].gt(0)
    )
    merged["has_selected_option_price"] = (
        merged["selected_option_price"].gt(0)
    )

    merged["usable_for_gex"] = (
        merged["is_not_expired"]
        & merged["has_positive_open_interest"]
        & merged["has_selected_option_price"]
        & merged["has_spot_price"]
        & merged["strike"].gt(0)
        & merged["contract_size"].gt(0)
    )

    usable = merged[
        merged["usable_for_gex"]
    ].copy()

    usable["calendar_days"] = (
        usable["maturity_date"]
        - selected_ts
    ).dt.days

    usable["business_days"] = (
        usable["maturity_date"]
        .apply(
            lambda maturity: (
                np.busday_count(
                    selected_ts.date(),
                    maturity.date(),
                )
                if pd.notna(maturity)
                and maturity > selected_ts
                else 0
            )
        )
    )

    usable["time_years"] = (
        usable["business_days"]
        / 252.0
    )

    usable["moneyness"] = (
        usable["strike"]
        / usable["selected_spot_price"]
    )

    usable = usable[
        usable["calendar_days"].between(
            1,
            MAX_DIAS_ATE_VENCIMENTO,
        )
        & usable["moneyness"].between(
            MONEYNESS_MINIMO,
            MONEYNESS_MAXIMO,
        )
    ].copy()

    return usable


# ======================================================================================
# 5) MOTOR DE IV, GAMMA E GEX
# ======================================================================================

def weighted_median(values, weights):
    values = np.asarray(
        values,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    return values[
        np.searchsorted(
            cumulative,
            weights.sum() / 2.0,
        )
    ]


def bsm_price(
    spot,
    strike,
    time_years,
    rate,
    dividend_yield,
    volatility,
    option_type,
):
    values = [
        spot,
        strike,
        time_years,
        rate,
        dividend_yield,
        volatility,
    ]

    if not all(
        np.isfinite(values)
    ):
        return np.nan

    if (
        spot <= 0
        or strike <= 0
        or time_years <= 0
        or volatility <= 0
    ):
        return np.nan

    sqrt_time = math.sqrt(
        time_years
    )

    d1 = (
        math.log(spot / strike)
        + (
            rate
            - dividend_yield
            + 0.5 * volatility**2
        )
        * time_years
    ) / (
        volatility * sqrt_time
    )

    d2 = d1 - volatility * sqrt_time

    discounted_spot = (
        spot
        * math.exp(
            -dividend_yield
            * time_years
        )
    )

    discounted_strike = (
        strike
        * math.exp(
            -rate
            * time_years
        )
    )

    if option_type == "CALL":
        return (
            discounted_spot
            * norm.cdf(d1)
            - discounted_strike
            * norm.cdf(d2)
        )

    return (
        discounted_strike
        * norm.cdf(-d2)
        - discounted_spot
        * norm.cdf(-d1)
    )


def solve_market_iv(
    option_price,
    spot,
    strike,
    time_years,
    rate,
    dividend_yield,
    option_type,
):
    values = [
        option_price,
        spot,
        strike,
        time_years,
        rate,
        dividend_yield,
    ]

    if not all(
        np.isfinite(values)
    ):
        return np.nan

    if (
        option_price <= 0
        or spot <= 0
        or strike <= 0
        or time_years <= 0
    ):
        return np.nan

    discounted_spot = (
        spot
        * math.exp(
            -dividend_yield
            * time_years
        )
    )

    discounted_strike = (
        strike
        * math.exp(
            -rate
            * time_years
        )
    )

    if option_type == "CALL":
        lower_bound = max(
            discounted_spot
            - discounted_strike,
            0.0,
        )
        upper_bound = discounted_spot
    else:
        lower_bound = max(
            discounted_strike
            - discounted_spot,
            0.0,
        )
        upper_bound = discounted_strike

    if (
        option_price < lower_bound - 1e-7
        or option_price > upper_bound + 1e-7
    ):
        return np.nan

    def objective(volatility):
        return (
            bsm_price(
                spot,
                strike,
                time_years,
                rate,
                dividend_yield,
                volatility,
                option_type,
            )
            - option_price
        )

    try:
        low_vol = 0.005
        high_vol = 5.0

        if (
            objective(low_vol)
            * objective(high_vol)
            > 0
        ):
            return np.nan

        return brentq(
            objective,
            low_vol,
            high_vol,
            xtol=1e-8,
            maxiter=100,
        )

    except Exception:
        return np.nan


def bsm_gamma(
    spot,
    strike,
    time_years,
    rate,
    dividend_yield,
    volatility,
):
    values = [
        spot,
        strike,
        time_years,
        rate,
        dividend_yield,
        volatility,
    ]

    if not all(
        np.isfinite(values)
    ):
        return np.nan

    if (
        spot <= 0
        or strike <= 0
        or time_years <= 0
        or volatility <= 0
    ):
        return np.nan

    sqrt_time = math.sqrt(
        time_years
    )

    d1 = (
        math.log(spot / strike)
        + (
            rate
            - dividend_yield
            + 0.5 * volatility**2
        )
        * time_years
    ) / (
        volatility * sqrt_time
    )

    return (
        math.exp(
            -dividend_yield
            * time_years
        )
        * norm.pdf(d1)
        / (
            spot
            * volatility
            * sqrt_time
        )
    )


def estimate_forwards_and_carry(pilot):
    rows = []

    for (
        underlying,
        maturity,
    ), full_chain in pilot.groupby(
        [
            "underlying_ticker",
            "maturity_date",
        ]
    ):
        spot = float(
            full_chain[
                "selected_spot_price"
            ].median()
        )
        time_years = float(
            full_chain[
                "time_years"
            ].median()
        )

        chain = full_chain[
            full_chain["option_style"].eq(
                "EURO"
            )
            & full_chain[
                "theoretic_price"
            ].gt(0)
        ].copy()

        chain = (
            chain.sort_values(
                "open_interest"
            )
            .drop_duplicates(
                subset=[
                    "strike",
                    "option_type_norm",
                ],
                keep="last",
            )
        )

        discount_factor = math.exp(
            -TAXA_LIVRE_RISCO_ANUAL
            * time_years
        )

        calls = (
            chain[
                chain[
                    "option_type_norm"
                ].eq("CALL")
            ]
            .set_index("strike")
        )

        puts = (
            chain[
                chain[
                    "option_type_norm"
                ].eq("PUT")
            ]
            .set_index("strike")
        )

        common_strikes = (
            calls.index.intersection(
                puts.index
            )
        )

        forward_candidates = []
        candidate_weights = []

        for strike in common_strikes:
            call_row = calls.loc[strike]
            put_row = puts.loc[strike]

            if isinstance(
                call_row,
                pd.DataFrame,
            ):
                call_row = call_row.iloc[-1]

            if isinstance(
                put_row,
                pd.DataFrame,
            ):
                put_row = put_row.iloc[-1]

            call_price = float(
                call_row[
                    "theoretic_price"
                ]
            )
            put_price = float(
                put_row[
                    "theoretic_price"
                ]
            )

            forward_candidate = (
                float(strike)
                + (
                    call_price
                    - put_price
                )
                / discount_factor
            )

            near_spot = (
                abs(
                    math.log(
                        float(strike)
                        / spot
                    )
                )
                <= 0.30
            )

            plausible = (
                0.50 * spot
                < forward_candidate
                < 1.80 * spot
            )

            if near_spot and plausible:
                forward_candidates.append(
                    forward_candidate
                )
                candidate_weights.append(
                    math.sqrt(
                        float(
                            call_row[
                                "open_interest"
                            ]
                        )
                        + float(
                            put_row[
                                "open_interest"
                            ]
                        )
                        + 1.0
                    )
                )

        pair_count = len(
            forward_candidates
        )

        if pair_count >= 2:
            forward_price = weighted_median(
                forward_candidates,
                candidate_weights,
            )

            dividend_yield = (
                TAXA_LIVRE_RISCO_ANUAL
                - math.log(
                    forward_price / spot
                )
                / time_years
            )

            dividend_yield = float(
                np.clip(
                    dividend_yield,
                    -0.10,
                    0.50,
                )
            )
            source = "put_call_parity"
        else:
            forward_price = np.nan
            dividend_yield = np.nan
            source = "insufficient_pairs"

        rows.append(
            {
                "underlying_ticker": underlying,
                "maturity_date": maturity,
                "spot": spot,
                "time_years_forward": time_years,
                "pair_count": pair_count,
                "forward": forward_price,
                "q": dividend_yield,
                "forward_source": source,
            }
        )

    forward_table = pd.DataFrame(rows)

    valid_q = forward_table[
        forward_table["pair_count"].ge(2)
        & forward_table["q"].notna()
    ]

    asset_median_q = (
        valid_q.groupby(
            "underlying_ticker"
        )["q"].median()
    )

    forward_table[
        "asset_median_q"
    ] = forward_table[
        "underlying_ticker"
    ].map(asset_median_q)

    missing = forward_table["q"].isna()

    forward_table.loc[
        missing,
        "q",
    ] = (
        forward_table.loc[
            missing,
            "asset_median_q",
        ]
        .fillna(0.0)
    )

    forward_table.loc[
        missing,
        "forward",
    ] = (
        forward_table.loc[
            missing,
            "spot",
        ]
        * np.exp(
            (
                TAXA_LIVRE_RISCO_ANUAL
                - forward_table.loc[
                    missing,
                    "q",
                ]
            )
            * forward_table.loc[
                missing,
                "time_years_forward",
            ]
        )
    )

    forward_table.loc[
        missing,
        "forward_source",
    ] = "asset_q_fallback"

    return forward_table


def compute_iv_gamma_gex(pilot):
    pilot = pilot.copy()

    pilot["option_type_norm"] = (
        pilot["option_type"]
        .replace(
            {
                "PUTT": "PUT",
                "PUT": "PUT",
                "CALL": "CALL",
            }
        )
    )

    pilot["iv_b3"] = (
        pd.to_numeric(
            pilot[
                "published_volatility"
            ],
            errors="coerce",
        )
        / 100.0
    )

    print(
        "  Estimando forward/carry por paridade call-put..."
    )

    forward_table = (
        estimate_forwards_and_carry(
            pilot
        )
    )

    pilot = pilot.merge(
        forward_table[
            [
                "underlying_ticker",
                "maturity_date",
                "forward",
                "q",
                "pair_count",
                "forward_source",
            ]
        ],
        on=[
            "underlying_ticker",
            "maturity_date",
        ],
        how="left",
    )

    asset_q_fallback = (
        pilot.groupby(
            "underlying_ticker"
        )["q"].transform("median")
    )

    pilot["q"] = (
        pilot["q"]
        .fillna(asset_q_fallback)
        .fillna(0.0)
    )

    pilot["forward"] = (
        pilot["forward"]
        .fillna(
            pilot[
                "selected_spot_price"
            ]
            * np.exp(
                (
                    TAXA_LIVRE_RISCO_ANUAL
                    - pilot["q"]
                )
                * pilot[
                    "time_years"
                ]
            )
        )
    )

    print(
        "  Calculando IV de mercado quando a série permite..."
    )

    market_iv_mask = (
        pilot["option_style"].eq(
            "EURO"
        )
        & pilot[
            "option_price_source"
        ].isin(
            [
                "mid_bid_ask",
                "ultimo_negocio",
            ]
        )
    )

    pilot["iv_market"] = np.nan
    market_indices = pilot.index[
        market_iv_mask
    ]

    values = []

    for position, index in enumerate(
        market_indices,
        start=1,
    ):
        row = pilot.loc[index]

        values.append(
            solve_market_iv(
                option_price=float(
                    row[
                        "selected_option_price"
                    ]
                ),
                spot=float(
                    row[
                        "selected_spot_price"
                    ]
                ),
                strike=float(
                    row["strike"]
                ),
                time_years=float(
                    row["time_years"]
                ),
                rate=(
                    TAXA_LIVRE_RISCO_ANUAL
                ),
                dividend_yield=float(
                    row["q"]
                ),
                option_type=row[
                    "option_type_norm"
                ],
            )
        )

        if position % 500 == 0:
            print(
                f"    IV: {position:,} séries..."
            )

    pilot.loc[
        market_indices,
        "iv_market",
    ] = values

    pilot["iv_used"] = (
        pilot["iv_b3"]
    )
    pilot["iv_source"] = (
        "volatilidade_referencia_b3"
    )

    pilot["iv_difference_pct"] = (
        (
            pilot["iv_market"]
            - pilot["iv_b3"]
        )
        .abs()
        / pilot["iv_b3"]
        * 100.0
    )

    use_mid_iv = (
        pilot["iv_market"].between(
            0.05,
            3.00,
        )
        & pilot["iv_b3"].between(
            0.05,
            3.00,
        )
        & pilot[
            "option_price_source"
        ].eq("mid_bid_ask")
        & pilot[
            "iv_difference_pct"
        ].le(75.0)
    )

    use_last_iv = (
        pilot["iv_market"].between(
            0.05,
            3.00,
        )
        & pilot["iv_b3"].between(
            0.05,
            3.00,
        )
        & pilot[
            "option_price_source"
        ].eq("ultimo_negocio")
        & pilot[
            "iv_difference_pct"
        ].le(50.0)
    )

    use_market_iv = (
        use_mid_iv | use_last_iv
    )

    pilot.loc[
        use_market_iv,
        "iv_used",
    ] = pilot.loc[
        use_market_iv,
        "iv_market",
    ]

    pilot.loc[
        use_market_iv,
        "iv_source",
    ] = "iv_mercado_calculada"

    pilot = pilot[
        pilot["iv_used"].between(
            0.05,
            3.00,
        )
    ].copy()

    print(
        f"    IV de mercado aceita: "
        f"{pilot['iv_source'].eq('iv_mercado_calculada').sum():,}; "
        f"referência B3: "
        f"{pilot['iv_source'].eq('volatilidade_referencia_b3').sum():,}."
    )

    print(
        "  Calculando Gamma e GEX..."
    )

    pilot["gamma"] = [
        bsm_gamma(
            spot=float(
                row.selected_spot_price
            ),
            strike=float(
                row.strike
            ),
            time_years=float(
                row.time_years
            ),
            rate=TAXA_LIVRE_RISCO_ANUAL,
            dividend_yield=float(
                row.q
            ),
            volatility=float(
                row.iv_used
            ),
        )
        for row in pilot.itertuples()
    ]

    pilot["gamma_model"] = np.where(
        pilot["option_style"].eq(
            "EURO"
        ),
        "BSM_FORWARD",
        "BSM_PROXY_EXERCICIO_ANTECIPADO",
    )

    # O PriceReport registra quantidade de opções/contratos em aberto.
    # Nos dados reais validados, volume financeiro = quantidade negociada
    # x preço médio da opção. Portanto não multiplicamos AllcnRndLot.
    pilot["gross_gamma_1pct"] = (
        pilot["gamma"]
        * pilot["open_interest"]
        * pilot[
            "selected_spot_price"
        ] ** 2
        * 0.01
    )

    pilot[
        "signed_gex_proxy_1pct"
    ] = np.where(
        pilot[
            "option_type_norm"
        ].eq("CALL"),
        pilot[
            "gross_gamma_1pct"
        ],
        -pilot[
            "gross_gamma_1pct"
        ],
    )

    pilot["market_price_flag"] = (
        pilot[
            "option_price_source"
        ].isin(
            [
                "mid_bid_ask",
                "ultimo_negocio",
            ]
        )
    )

    return (
        pilot.reset_index(drop=True),
        forward_table.reset_index(drop=True),
    )


def run_full_pipeline(force=False):
    """B3 → arquivos → opções → IV/Gamma/GEX. Nenhum ZIP manual."""
    print("\n" + "=" * 100)
    print("GEX RADAR BRASIL — ATUALIZAÇÃO INTEGRADA")
    print("=" * 100)

    requested_date = (
        datetime.strptime(
            DATA_REFERENCIA_MANUAL,
            "%Y-%m-%d",
        ).date()
        if DATA_REFERENCIA_MANUAL
        else current_brazil_date()
    )

    session = make_session()
    selected_date = None
    selected_paths = {}
    selected_content = None

    try:
        for offset in range(
            RETROCEDER_DIAS + 1
        ):
            candidate = (
                requested_date
                - timedelta(days=offset)
            )
            yymmdd = candidate.strftime(
                "%y%m%d"
            )
            candidate_dir = (
                RAW_CACHE_DIR
                / candidate.isoformat()
            )

            instrument_name = (
                f"IN{yymmdd}.zip"
            )
            price_name = (
                f"PR{yymmdd}.zip"
            )

            print(
                f"Testando {candidate.isoformat()}..."
            )

            instrument_path = (
                candidate_dir
                / instrument_name
            )
            price_path = (
                candidate_dir
                / price_name
            )

            ok_in, msg_in = download_pregao(
                session,
                instrument_name,
                instrument_path,
                force=force,
            )
            print(
                f"  {instrument_name}: {msg_in}"
            )

            ok_pr, msg_pr = download_pregao(
                session,
                price_name,
                price_path,
                force=force,
            )
            print(
                f"  {price_name}: {msg_pr}"
            )

            if not (ok_in and ok_pr):
                continue

            reference_name = (
                f"PE{yymmdd}.ex_"
            )
            reference_path = (
                candidate_dir
                / reference_name
            )

            ok_pe, msg_pe = download_pregao(
                session,
                reference_name,
                reference_path,
                force=force,
            )
            print(
                f"  {reference_name}: {msg_pe}"
            )

            if not ok_pe:
                print(
                    "  Sessão ainda incompleta para o GEX; "
                    "tentando a data anterior."
                )
                continue

            candidate_paths = {
                "instruments": instrument_path,
                "prices": price_path,
                "reference": reference_path,
            }

            # CORREÇÃO STREAMLIT / REABERTURA:
            # não basta o contêiner ser um ZIP válido. Validamos o conteúdo
            # esperado ANTES de aceitar a data como sessão completa.
            try:
                candidate_content = extract_and_validate_session(
                    candidate_paths,
                    candidate,
                )
            except Exception as exc:
                print(
                    "  Sessão rejeitada: arquivos compactados disponíveis, "
                    "mas conteúdo interno incompleto/incompatível "
                    f"({type(exc).__name__}: {exc})."
                )

                # Descarta apenas o cache semanticamente inválido da data,
                # permitindo nova tentativa futura sem reutilizar o mesmo ZIP.
                message = str(exc)
                if "Cadastro de Instrumentos" in message or instrument_name in message:
                    invalidate_semantically_bad_cache(instrument_path)
                if "PriceReport" in message or price_name in message:
                    invalidate_semantically_bad_cache(price_path)
                if "Prêmio de Referência" in message or reference_name in message:
                    invalidate_semantically_bad_cache(reference_path)

                bad_work_dir = WORK_DIR / candidate.isoformat()
                if bad_work_dir.exists():
                    shutil.rmtree(
                        bad_work_dir,
                        ignore_errors=True,
                    )

                print(
                    "  Tentando a data anterior."
                )
                continue

            # A volatilidade de referência da B3 é parte importante do motor
            # validado. A data só é aceita quando IN + PR + PE possuem também
            # o conteúdo interno esperado pelo pipeline.
            selected_date = candidate
            selected_paths = candidate_paths
            selected_content = candidate_content
            break

    finally:
        session.close()

    if selected_date is None or selected_content is None:
        raise RuntimeError(
            "Não foi possível obter uma sessão completa com Cadastro de Instrumentos, "
            "PriceReport e Prêmio de Referência dentro da janela pesquisada. "
            "Arquivos ZIP sem o conteúdo interno esperado são ignorados automaticamente."
        )

    print(
        f"\nData efetiva selecionada: {selected_date.isoformat()}"
    )

    # A sessão escolhida já foi extraída e validada durante a seleção.
    # Reutilizamos exatamente esses arquivos para evitar uma segunda extração.
    run_work_dir = selected_content["work_dir"]
    extracted = selected_content["extracted"]
    instrument_xml = selected_content["instrument_xml"]
    price_xml = selected_content["price_xml"]
    reference_txt = selected_content["reference_txt"]

    print("\nLendo Cadastro de Instrumentos...")
    instruments, options = (
        parse_instruments(
            instrument_xml
        )
    )

    print("\nLendo PriceReport...")
    prices = parse_price_report(
        price_xml
    )

    print("\nLendo Prêmio de Referência...")
    reference = parse_reference_premium(
        reference_txt
    )
    print(
        f"    Prêmios de referência: {len(reference):,}."
    )

    print("\nMontando universo utilizável...")
    market_base = build_market_base(
        options,
        prices,
        reference,
        selected_date,
    )
    print(
        f"    Séries B3 monitoradas após filtros: {len(market_base):,}."
    )

    if market_base.empty:
        raise RuntimeError(
            "Nenhuma série passou pelos filtros do GEX."
        )

    print("\nExecutando motor de IV/Gamma/GEX...")
    result, forward_table = (
        compute_iv_gamma_gex(
            market_base
        )
    )

    if result.empty:
        raise RuntimeError(
            "O motor de IV/Gamma não produziu séries válidas."
        )

    print("\nResumo da atualização")
    for asset in ATIVOS_PILOTO:
        chain = result[
            result[
                "underlying_ticker"
            ].eq(asset)
        ]
        if chain.empty:
            print(
                f"  {asset}: sem séries válidas"
            )
        else:
            print(
                f"  {asset}: {len(chain):,} séries; "
                f"spot {chain['selected_spot_price'].median():.2f}"
            )

    metadata = {
        "reference_date": selected_date.isoformat(),
        "requested_date": requested_date.isoformat(),
        "assets": ATIVOS_PILOTO,
        "max_days_to_expiry": MAX_DIAS_ATE_VENCIMENTO,
        "moneyness_range": [
            MONEYNESS_MINIMO,
            MONEYNESS_MAXIMO,
        ],
        "risk_free_rate_assumption": TAXA_LIVRE_RISCO_ANUAL,
        "instrument_file": instrument_xml.name,
        "price_file": price_xml.name,
        "reference_file": (
            reference_txt.name
            if reference_txt is not None
            else None
        ),
        "instrument_count": int(
            len(instruments)
        ),
        "pilot_option_count_before_math": int(
            len(market_base)
        ),
        "series_after_iv_gamma": int(
            len(result)
        ),
        "forward_expiry_count": int(
            len(forward_table)
        ),
        "net_gex_hypothesis": (
            "Calls positivas e puts negativas. "
            "Não representa dealer Gamma observado."
        ),
        "gamma_method": (
            "BSM para exercício no vencimento; BSM como proxy "
            "para contratos que admitem exercício antecipado."
        ),
        "open_interest_treatment": (
            "Open interest do PriceReport usado diretamente. "
            "AllcnRndLot é lote de alocação e não é multiplicado no GEX."
        ),
    }

    print(
        f"\nAtualização concluída com base em {selected_date.isoformat()}."
    )

    return result, metadata



# ======================================================================================
# 6) NORMALIZAÇÃO PARA O PAINEL
# ======================================================================================

def prepare_panel_data(frame):
    frame = frame.copy()

    numeric_columns = [
        "strike",
        "open_interest",
        "selected_option_price",
        "selected_spot_price",
        "calendar_days",
        "business_days",
        "time_years",
        "iv_b3",
        "iv_market",
        "iv_used",
        "gamma",
        "gross_gamma_1pct",
        "signed_gex_proxy_1pct",
        "q",
        "pair_count",
        "option_traded_quantity",
        "option_financial_volume",
        "option_best_bid",
        "option_best_ask",
        "theoretic_price",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame["maturity_date"] = pd.to_datetime(
        frame["maturity_date"],
        errors="coerce",
    )

    frame["option_type_norm"] = (
        frame["option_type_norm"]
        .replace(
            {
                "PUTT": "PUT",
                "PUT": "PUT",
                "CALL": "CALL",
            }
        )
    )

    frame["exercise_style_display"] = (
        frame["option_style"]
        .map(
            {
                "AMER": "Exercício antecipado",
                "EURO": "Exercício no vencimento",
            }
        )
        .fillna(
            frame["option_style"].astype(str)
        )
    )

    frame["option_type_display"] = (
        frame["option_type_norm"]
        .map(
            {
                "CALL": "Call",
                "PUT": "Put",
            }
        )
        .fillna(
            frame["option_type_norm"]
        )
    )

    if "market_price_flag" not in frame.columns:
        frame["market_price_flag"] = (
            frame[
                "option_price_source"
            ].isin(
                [
                    "mid_bid_ask",
                    "ultimo_negocio",
                ]
            )
        )

    if "has_trading" not in frame.columns:
        frame["has_trading"] = (
            frame[
                "option_traded_quantity"
            ]
            .fillna(0)
            .gt(0)
        )

    return frame



# ============================================================
# RUNTIME — preenchido pela interface Streamlit após carregar a B3.
# ============================================================
gex_series = pd.DataFrame()
historical_prices = pd.DataFrame()
metadata = {}
REFERENCE_DATE = pd.Timestamp(current_brazil_date())
RISK_FREE_RATE = float(TAXA_LIVRE_RISCO_ANUAL)
MAX_BASE_DAYS = int(MAX_DIAS_ATE_VENCIMENTO)
ASSETS = ATIVOS_B3.copy()
DISPLAY_ASSETS = ATIVOS_EXIBICAO.copy()


# ======================================================================================
# 7) CONFIGURAÇÃO DOS HORIZONTES — MULTI-HORIZONTE
# ======================================================================================

HORIZONS = {
    "30 dias": 30,
    "60 dias": 60,
    "90 dias": 90,
    "180 dias": 180,
}

# Faixas exclusivas de DTE usadas no cálculo de cada coluna do radar.
# Não há acumulação nem sobreposição entre os quatro horizontes.
HORIZON_DTE_RANGES = {
    "30 dias": (1, 30),
    "60 dias": (31, 60),
    "90 dias": (61, 90),
    "180 dias": (91, 180),
}

HORIZON_ORDER = [
    "30 dias",
    "60 dias",
    "90 dias",
    "180 dias",
]

HORIZON_SHORT = {
    "30 dias": "30d",
    "60 dias": "60d",
    "90 dias": "90d",
    "180 dias": "180d",
}


def chart_trading_days_for_horizon(
    horizon_label,
):
    """Quantidade de pregões do gráfico correspondente ao recorte GEX."""
    horizon_days = int(
        HORIZONS[horizon_label]
    )

    return max(
        1,
        min(
            horizon_days,
            int(MAX_HISTORICO_PREGOES),
        ),
    )


def gex_scope_text(
    horizon_label,
    exact_expiry=None,
):
    """Texto curto usado no título do gráfico para deixar o recorte explícito."""
    if exact_expiry is not None:
        return (
            "GEX: vencimento "
            + pd.Timestamp(
                exact_expiry
            ).strftime("%d/%m/%Y")
        )

    dte_min, dte_max = HORIZON_DTE_RANGES[
        horizon_label
    ]

    return (
        f"GEX: opções com DTE de {dte_min} a {dte_max} dias"
    )

# ======================================================================================
# 7) FUNÇÕES DE FORMATAÇÃO
# ======================================================================================

def br_number(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "—"

    text = f"{value:,.{decimals}f}"

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def br_money(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "—"

    return f"R$ {br_number(value, decimals)}"


def compact_brl(value):
    if value is None or not np.isfinite(value):
        return "—"

    sign = "-" if value < 0 else ""
    absolute = abs(float(value))

    if absolute >= 1_000_000_000:
        return (
            f"{sign}R$ "
            f"{br_number(absolute / 1_000_000_000, 2)} bi"
        )

    if absolute >= 1_000_000:
        return (
            f"{sign}R$ "
            f"{br_number(absolute / 1_000_000, 2)} mi"
        )

    if absolute >= 1_000:
        return (
            f"{sign}R$ "
            f"{br_number(absolute / 1_000, 2)} mil"
        )

    return (
        f"{sign}R$ "
        f"{br_number(absolute, 2)}"
    )


def br_pct(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "—"

    prefix = "+" if value > 0 else ""

    return (
        f"{prefix}"
        f"{br_number(value, decimals)}%"
    )


def format_level(level, spot):
    if level is None or not np.isfinite(level):
        return {
            "level": "Não identificado",
            "distance": "—",
            "distance_pct": "—",
        }

    distance = level - spot
    distance_pct = (
        (level / spot - 1.0) * 100.0
    )

    return {
        "level": br_money(level),
        "distance": br_money(distance),
        "distance_pct": br_pct(distance_pct),
    }


# ======================================================================================
# 8) FUNÇÕES DE CÁLCULO
# ======================================================================================

def filter_horizon(
    frame,
    horizon_label,
):
    """Recorta as opções pela faixa exclusiva de DTE do horizonte.

    Regra operacional definida para o radar:
    - 30 dias  -> calendar_days entre 1 e 30, inclusive
    - 60 dias  -> calendar_days entre 31 e 60, inclusive
    - 90 dias  -> calendar_days entre 61 e 90, inclusive
    - 180 dias -> calendar_days entre 91 e 180, inclusive

    Os horizontes são independentes, não cumulativos e não se sobrepõem.
    Cada série entra em no máximo um dos quatro recortes.
    """
    result = frame.copy()

    dte_min, dte_max = HORIZON_DTE_RANGES[
        horizon_label
    ]

    return result[
        result["calendar_days"].between(
            int(dte_min),
            int(dte_max),
            inclusive="both",
        )
    ].copy()


def filter_asset(
    asset,
    horizon_label,
    exact_expiry=None,
):
    frame = gex_series[
        gex_series[
            "underlying_ticker"
        ].eq(asset)
    ].copy()

    # O modo de vencimento específico é uma investigação independente do radar
    # 30/60/90/180. Por isso, quando uma data exata é escolhida, filtramos pela
    # data de vencimento e não aplicamos também o DTE do horizonte.
    if exact_expiry is not None:
        exact_expiry = pd.Timestamp(
            exact_expiry
        )

        return frame[
            frame["maturity_date"].eq(
                exact_expiry
            )
        ].copy()

    return filter_horizon(
        frame,
        horizon_label,
    )


def aggregate_by_strike(
    chain,
):
    if chain.empty:
        return pd.DataFrame(
            columns=[
                "strike",
                "call_gex_1pct",
                "put_gex_1pct",
                "gross_gamma_1pct",
                "net_gex_proxy_1pct",
                "open_interest",
                "series_count",
            ]
        )

    rows = []

    for strike, group in chain.groupby(
        "strike"
    ):
        call_gex = (
            group.loc[
                group[
                    "option_type_norm"
                ].eq("CALL"),
                "gross_gamma_1pct",
            ]
            .sum()
        )

        put_gex = (
            group.loc[
                group[
                    "option_type_norm"
                ].eq("PUT"),
                "gross_gamma_1pct",
            ]
            .sum()
        )

        rows.append(
            {
                "strike": float(strike),
                "call_gex_1pct": float(call_gex),
                "put_gex_1pct": float(put_gex),
                "gross_gamma_1pct": float(
                    call_gex + put_gex
                ),
                "net_gex_proxy_1pct": float(
                    call_gex - put_gex
                ),
                "open_interest": float(
                    group[
                        "open_interest"
                    ].sum()
                ),
                "series_count": int(
                    len(group)
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("strike")
        .reset_index(drop=True)
    )



def wall_separation_threshold(
    by_strike,
    spot,
):
    """Define a separação mínima entre Walls a partir da malha real de strikes."""
    if by_strike.empty:
        return 0.0

    local = by_strike[
        by_strike[
            "strike"
        ].between(
            spot * 0.70,
            spot * 1.30,
        )
    ].copy()

    strikes = np.sort(
        local["strike"]
        .dropna()
        .unique()
        .astype(float)
    )

    if len(strikes) < 2:
        return 0.0

    diffs = np.diff(
        strikes
    )

    diffs = diffs[
        np.isfinite(diffs)
        & (diffs > 1e-9)
    ]

    if not len(diffs):
        return 0.0

    # Percentil 75 reduz a influência de strikes ajustados
    # extremamente próximos entre si.
    typical_gap = float(
        np.quantile(
            diffs,
            0.75,
        )
    )

    return (
        typical_gap
        * WALL_GAP_MULTIPLIER
    )


def select_distinct_walls(
    by_strike,
    gamma_column,
    spot,
    count=NUM_WALLS_DETALHE,
):
    """Seleciona até N concentrações distintas de Gamma.

    1) Prioriza máximos locais.
    2) Ordena pela concentração de Gross Gamma.
    3) Impede que strikes praticamente colados sejam chamados de Walls diferentes.
    """
    if (
        by_strike.empty
        or gamma_column
        not in by_strike.columns
    ):
        return []

    candidates = (
        by_strike[
            [
                "strike",
                gamma_column,
            ]
        ]
        .dropna()
        .copy()
    )

    candidates = candidates[
        candidates[
            gamma_column
        ].gt(0)
    ].sort_values(
        "strike"
    )

    if candidates.empty:
        return []

    total_gamma = float(
        candidates[
            gamma_column
        ].sum()
    )

    separation = (
        wall_separation_threshold(
            by_strike,
            spot,
        )
    )

    values = (
        candidates[
            gamma_column
        ].to_numpy(
            dtype=float
        )
    )

    # Máximos locais na malha de strikes.
    local_peak_mask = np.ones(
        len(candidates),
        dtype=bool,
    )

    if len(candidates) >= 2:
        local_peak_mask[0] = (
            values[0]
            >= values[1]
        )

        local_peak_mask[-1] = (
            values[-1]
            >= values[-2]
        )

    if len(candidates) >= 3:
        local_peak_mask[
            1:-1
        ] = (
            (
                values[1:-1]
                >= values[:-2]
            )
            & (
                values[1:-1]
                >= values[2:]
            )
        )

    local_peaks = (
        candidates[
            local_peak_mask
        ]
        .sort_values(
            gamma_column,
            ascending=False,
        )
    )

    all_ranked = (
        candidates
        .sort_values(
            gamma_column,
            ascending=False,
        )
    )

    selected = []

    def try_add(row):
        strike = float(
            row["strike"]
        )

        if any(
            abs(
                strike
                - item["strike"]
            )
            < separation
            for item in selected
        ):
            return

        gamma_value = float(
            row[
                gamma_column
            ]
        )

        selected.append(
            {
                "strike": strike,
                "gamma_1pct": gamma_value,
                "share_pct": (
                    gamma_value
                    / total_gamma
                    * 100.0
                    if total_gamma > 0
                    else np.nan
                ),
                "distance": (
                    strike - spot
                ),
                "distance_pct": (
                    (
                        strike / spot
                        - 1.0
                    )
                    * 100.0
                    if spot > 0
                    else np.nan
                ),
                "separation_used": (
                    separation
                ),
            }
        )

    for _, row in (
        local_peaks.iterrows()
    ):
        try_add(row)

        if len(selected) >= count:
            break

    # Caso existam poucos máximos locais, completa com as maiores
    # concentrações restantes, mantendo a mesma regra de separação.
    if len(selected) < count:
        selected_strikes = {
            item["strike"]
            for item in selected
        }

        for _, row in (
            all_ranked.iterrows()
        ):
            if float(
                row["strike"]
            ) in selected_strikes:
                continue

            try_add(row)

            selected_strikes = {
                item["strike"]
                for item in selected
            }

            if len(selected) >= count:
                break

    for rank, item in enumerate(
        selected,
        start=1,
    ):
        item["rank"] = rank

    return selected


def calculate_flip_curve(
    chain,
    lower_multiplier=0.70,
    upper_multiplier=1.30,
    points=201,
):
    if chain.empty:
        return np.nan, pd.DataFrame()

    required = chain[
        [
            "strike",
            "time_years",
            "iv_used",
            "q",
            "open_interest",
            "option_type_norm",
            "selected_spot_price",
        ]
    ].dropna()

    required = required[
        required["time_years"].gt(0)
        & required["iv_used"].gt(0)
        & required["strike"].gt(0)
        & required["open_interest"].gt(0)
    ]

    if required.empty:
        return np.nan, pd.DataFrame()

    spot = float(
        required[
            "selected_spot_price"
        ].median()
    )

    scenarios = np.linspace(
        spot * lower_multiplier,
        spot * upper_multiplier,
        points,
    )

    strikes = (
        required["strike"]
        .to_numpy(dtype=float)[:, None]
    )

    times = (
        required["time_years"]
        .to_numpy(dtype=float)[:, None]
    )

    vols = (
        required["iv_used"]
        .to_numpy(dtype=float)[:, None]
    )

    qs = (
        required["q"]
        .to_numpy(dtype=float)[:, None]
    )

    open_interest = (
        required["open_interest"]
        .to_numpy(dtype=float)[:, None]
    )

    signs = np.where(
        required[
            "option_type_norm"
        ].eq("CALL"),
        1.0,
        -1.0,
    )[:, None]

    spot_matrix = scenarios[
        None,
        :
    ]

    d1 = (
        np.log(
            spot_matrix / strikes
        )
        + (
            RISK_FREE_RATE
            - qs
            + 0.5 * vols**2
        )
        * times
    ) / (
        vols * np.sqrt(times)
    )

    gamma = (
        np.exp(-qs * times)
        * np.exp(-0.5 * d1**2)
        / np.sqrt(2.0 * np.pi)
        / (
            spot_matrix
            * vols
            * np.sqrt(times)
        )
    )

    net_gex = (
        gamma
        * open_interest
        * spot_matrix**2
        * 0.01
        * signs
    ).sum(axis=0)

    roots = []

    for index in range(
        len(scenarios) - 1
    ):
        left = net_gex[index]
        right = net_gex[
            index + 1
        ]

        if left == 0:
            roots.append(
                scenarios[index]
            )

        elif left * right < 0:
            root = (
                scenarios[index]
                + (
                    -left
                )
                * (
                    scenarios[
                        index + 1
                    ]
                    - scenarios[index]
                )
                / (
                    right - left
                )
            )

            roots.append(root)

    gamma_flip = (
        min(
            roots,
            key=lambda value: abs(
                value - spot
            ),
        )
        if roots
        else np.nan
    )

    curve = pd.DataFrame(
        {
            "spot_scenario": scenarios,
            "net_gex_proxy_1pct": net_gex,
        }
    )

    return gamma_flip, curve


def calculate_quality(
    chain,
):
    if chain.empty:
        return {
            "score": 0.0,
            "label": "BAIXA",
            "market_price_oi_share_pct": 0.0,
            "iv_market_oi_share_pct": 0.0,
            "traded_series_share_pct": 0.0,
            "parity_expiry_share_pct": 0.0,
            "exercise_at_expiry_oi_share_pct": 0.0,
            "gamma_oi_coverage_pct": 0.0,
            "iv_oi_coverage_pct": 0.0,
        }

    total_oi = float(
        chain[
            "open_interest"
        ].fillna(0).sum()
    )

    if total_oi <= 0:
        total_oi = 1.0

    gamma_oi_coverage = (
        chain.loc[
            chain["gamma"].notna(),
            "open_interest",
        ]
        .fillna(0)
        .sum()
        / total_oi
    )

    iv_oi_coverage = (
        chain.loc[
            chain["iv_used"].notna(),
            "open_interest",
        ]
        .fillna(0)
        .sum()
        / total_oi
    )

    market_price_oi_share = (
        chain.loc[
            chain[
                "market_price_flag"
            ].fillna(False),
            "open_interest",
        ]
        .fillna(0)
        .sum()
        / total_oi
    )

    iv_market_oi_share = (
        chain.loc[
            chain[
                "iv_source"
            ].eq(
                "iv_mercado_calculada"
            ),
            "open_interest",
        ]
        .fillna(0)
        .sum()
        / total_oi
    )

    traded_series_share = float(
        chain[
            "has_trading"
        ]
        .fillna(False)
        .mean()
    )

    exercise_at_expiry_oi_share = (
        chain.loc[
            chain[
                "option_style"
            ].eq("EURO"),
            "open_interest",
        ]
        .fillna(0)
        .sum()
        / total_oi
    )

    expiry_quality = (
        chain.groupby(
            "maturity_date"
        )["pair_count"]
        .max()
        .ge(2)
    )

    parity_expiry_share = (
        float(
            expiry_quality.mean()
        )
        if len(expiry_quality)
        else 0.0
    )

    score = (
        35.0 * gamma_oi_coverage
        + 20.0 * iv_oi_coverage
        + 15.0 * market_price_oi_share
        + 15.0 * parity_expiry_share
        + 10.0 * traded_series_share
        + 5.0 * exercise_at_expiry_oi_share
    )

    if score >= 85:
        label = "ALTA"

    elif score >= 70:
        label = "BOA"

    elif score >= 50:
        label = "LIMITADA"

    else:
        label = "BAIXA"

    return {
        "score": round(
            float(score),
            1,
        ),
        "label": label,
        "market_price_oi_share_pct": round(
            100.0 * market_price_oi_share,
            2,
        ),
        "iv_market_oi_share_pct": round(
            100.0 * iv_market_oi_share,
            2,
        ),
        "traded_series_share_pct": round(
            100.0 * traded_series_share,
            2,
        ),
        "parity_expiry_share_pct": round(
            100.0 * parity_expiry_share,
            2,
        ),
        "exercise_at_expiry_oi_share_pct": round(
            100.0
            * exercise_at_expiry_oi_share,
            2,
        ),
        "gamma_oi_coverage_pct": round(
            100.0 * gamma_oi_coverage,
            2,
        ),
        "iv_oi_coverage_pct": round(
            100.0 * iv_oi_coverage,
            2,
        ),
    }



def proximity_status(
    distance_pct,
):
    """Classifica apenas a proximidade matemática ao nível estrutural."""
    if not np.isfinite(
        distance_pct
    ):
        return "SEM NÍVEL"

    absolute = abs(
        float(
            distance_pct
        )
    )

    if absolute <= PROXIMIDADE_EM_CIMA_PCT:
        return "EM CIMA DO NÍVEL"

    if absolute <= PROXIMIDADE_MUITO_PROXIMO_PCT:
        return "MUITO PRÓXIMO"

    if absolute <= PROXIMIDADE_PROXIMO_PCT:
        return "PRÓXIMO"

    return "DISTANTE"


def decision_layer_metrics(
    spot,
    gross_gamma,
    net_gex,
    call_walls,
    put_walls,
    gamma_flip,
):
    """Organiza as métricas já calculadas para leitura rápida.

    Não cria previsão direcional nem sinal de compra/venda.
    """
    gross_gamma = float(
        gross_gamma
    )

    net_gex = float(
        net_gex
    )

    asymmetry_pct = (
        abs(net_gex)
        / gross_gamma
        * 100.0
        if np.isfinite(gross_gamma)
        and gross_gamma > 0
        else np.nan
    )

    call_primary = (
        call_walls[0]
        if call_walls
        else None
    )

    put_primary = (
        put_walls[0]
        if put_walls
        else None
    )

    call_level = (
        float(
            call_primary["strike"]
        )
        if call_primary is not None
        else np.nan
    )

    put_level = (
        float(
            put_primary["strike"]
        )
        if put_primary is not None
        else np.nan
    )

    confluence = (
        np.isfinite(call_level)
        and np.isfinite(put_level)
        and np.isclose(
            call_level,
            put_level,
            atol=CONFLUENCIA_WALL_ATOL,
            rtol=0.0,
        )
    )

    confluence_level = (
        float(
            (
                call_level
                + put_level
            )
            / 2.0
        )
        if confluence
        else np.nan
    )

    candidates = []

    if confluence:
        distance = (
            confluence_level
            - spot
        )

        distance_pct = (
            distance
            / spot
            * 100.0
        )

        call_share = (
            float(
                call_primary.get(
                    "share_pct",
                    np.nan,
                )
            )
            if call_primary is not None
            else np.nan
        )

        put_share = (
            float(
                put_primary.get(
                    "share_pct",
                    np.nan,
                )
            )
            if put_primary is not None
            else np.nan
        )

        candidates.append(
            {
                "type": "CONFLUÊNCIA",
                "label": "Confluência Call/Put W1",
                "level": confluence_level,
                "distance": distance,
                "distance_pct": distance_pct,
                "abs_distance_pct": abs(
                    distance_pct
                ),
                "concentration_text": (
                    "Call "
                    f"{call_share:.2f}%"
                    " • Put "
                    f"{put_share:.2f}%"
                ),
            }
        )

    else:
        if call_primary is not None:
            call_distance_pct = (
                (
                    call_level
                    / spot
                    - 1.0
                )
                * 100.0
            )

            candidates.append(
                {
                    "type": "CALL_WALL",
                    "label": "Call Wall 1",
                    "level": call_level,
                    "distance": (
                        call_level
                        - spot
                    ),
                    "distance_pct": call_distance_pct,
                    "abs_distance_pct": abs(
                        call_distance_pct
                    ),
                    "concentration_text": (
                        f"{float(call_primary.get('share_pct', np.nan)):.2f}% "
                        "do Gamma das calls"
                    ),
                }
            )

        if put_primary is not None:
            put_distance_pct = (
                (
                    put_level
                    / spot
                    - 1.0
                )
                * 100.0
            )

            candidates.append(
                {
                    "type": "PUT_WALL",
                    "label": "Put Wall 1",
                    "level": put_level,
                    "distance": (
                        put_level
                        - spot
                    ),
                    "distance_pct": put_distance_pct,
                    "abs_distance_pct": abs(
                        put_distance_pct
                    ),
                    "concentration_text": (
                        f"{float(put_primary.get('share_pct', np.nan)):.2f}% "
                        "do Gamma das puts"
                    ),
                }
            )

    # Gamma Flip continua sendo calculado e armazenado nas métricas,
    # mas não participa da TRIAGEM PRINCIPAL / zona de atenção.
    # A proximidade principal compara somente Call W1, Put W1 e
    # a confluência Call/Put W1, conforme definido para a V21.

    nearest = (
        min(
            candidates,
            key=lambda item: item[
                "abs_distance_pct"
            ],
        )
        if candidates
        else None
    )

    if nearest is None:
        nearest_label = "Sem nível"
        nearest_type = "NONE"
        nearest_level = np.nan
        nearest_distance = np.nan
        nearest_distance_pct = np.nan
        nearest_abs_distance_pct = np.nan
        nearest_concentration_text = "—"
        nearest_status = "SEM NÍVEL"

    else:
        nearest_label = nearest[
            "label"
        ]
        nearest_type = nearest[
            "type"
        ]
        nearest_level = float(
            nearest[
                "level"
            ]
        )
        nearest_distance = float(
            nearest[
                "distance"
            ]
        )
        nearest_distance_pct = float(
            nearest[
                "distance_pct"
            ]
        )
        nearest_abs_distance_pct = float(
            nearest[
                "abs_distance_pct"
            ]
        )
        nearest_concentration_text = nearest[
            "concentration_text"
        ]
        nearest_status = proximity_status(
            nearest_distance_pct
        )

    if (
        np.isfinite(
            call_level
        )
        and np.isfinite(
            put_level
        )
    ):
        range_low = min(
            call_level,
            put_level,
        )

        range_high = max(
            call_level,
            put_level,
        )

        range_width = (
            range_high
            - range_low
        )

        range_width_pct = (
            range_width
            / spot
            * 100.0
            if spot > 0
            else np.nan
        )

        if confluence:
            range_position_pct = np.nan
            range_location = "CONFLUÊNCIA"

        elif spot < range_low:
            range_position_pct = (
                (
                    spot
                    - range_low
                )
                / range_width
                * 100.0
                if range_width > 0
                else np.nan
            )
            range_location = "ABAIXO DA FAIXA"

        elif spot > range_high:
            range_position_pct = (
                (
                    spot
                    - range_low
                )
                / range_width
                * 100.0
                if range_width > 0
                else np.nan
            )
            range_location = "ACIMA DA FAIXA"

        else:
            range_position_pct = (
                (
                    spot
                    - range_low
                )
                / range_width
                * 100.0
                if range_width > 0
                else np.nan
            )
            range_location = "DENTRO DA FAIXA"

    else:
        range_low = np.nan
        range_high = np.nan
        range_width = np.nan
        range_width_pct = np.nan
        range_position_pct = np.nan
        range_location = "SEM FAIXA"

    return {
        "gex_asymmetry_pct": asymmetry_pct,
        "nearest_level_label": nearest_label,
        "nearest_level_type": nearest_type,
        "nearest_level": nearest_level,
        "nearest_distance": nearest_distance,
        "nearest_distance_pct": nearest_distance_pct,
        "nearest_abs_distance_pct": nearest_abs_distance_pct,
        "nearest_concentration_text": nearest_concentration_text,
        "proximity_status": nearest_status,
        "primary_wall_confluence": bool(
            confluence
        ),
        "primary_wall_confluence_level": confluence_level,
        "range_low": range_low,
        "range_high": range_high,
        "range_width": range_width,
        "range_width_pct": range_width_pct,
        "range_position_pct": range_position_pct,
        "range_location": range_location,
    }


def calculate_metrics(
    chain,
):
    if chain.empty:
        return None

    spot = float(
        chain[
            "selected_spot_price"
        ].median()
    )

    by_strike = aggregate_by_strike(
        chain
    )

    call_walls = (
        select_distinct_walls(
            by_strike,
            "call_gex_1pct",
            spot,
            count=NUM_WALLS_DETALHE,
        )
    )

    put_walls = (
        select_distinct_walls(
            by_strike,
            "put_gex_1pct",
            spot,
            count=NUM_WALLS_DETALHE,
        )
    )

    # Compatibilidade com tudo que já estava validado:
    # Call Wall / Put Wall continuam significando a Wall principal.
    call_wall = (
        call_walls[0]["strike"]
        if call_walls
        else np.nan
    )

    put_wall = (
        put_walls[0]["strike"]
        if put_walls
        else np.nan
    )

    gamma_flip, flip_curve = (
        calculate_flip_curve(
            chain
        )
    )

    quality = calculate_quality(
        chain
    )

    gross_gamma = float(
        chain[
            "gross_gamma_1pct"
        ].sum()
    )

    net_gex = float(
        chain[
            "signed_gex_proxy_1pct"
        ].sum()
    )

    decision = decision_layer_metrics(
        spot=spot,
        gross_gamma=gross_gamma,
        net_gex=net_gex,
        call_walls=call_walls,
        put_walls=put_walls,
        gamma_flip=gamma_flip,
    )

    return {
        "spot": spot,
        "series_count": int(
            len(chain)
        ),
        "expiry_count": int(
            chain[
                "maturity_date"
            ].nunique()
        ),
        "open_interest": float(
            chain[
                "open_interest"
            ].sum()
        ),
        "gross_gamma_1pct": gross_gamma,
        "net_gex_proxy_1pct": net_gex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_walls": call_walls,
        "put_walls": put_walls,
        "gamma_flip": gamma_flip,
        "quality": quality,
        "by_strike": by_strike,
        "flip_curve": flip_curve,
        **decision,
    }


# ======================================================================================
# 9) CACHE DOS RESUMOS POR HORIZONTE
# ======================================================================================

metrics_cache = {}


def invalidate_metrics_cache():
    """Invalida somente as métricas agregadas por horizonte/vencimento."""
    metrics_cache.clear()


def get_metrics(
    asset,
    horizon_label,
    exact_expiry=None,
):
    key = (
        asset,
        horizon_label,
        str(exact_expiry)
        if exact_expiry is not None
        else "ALL",
    )

    if key not in metrics_cache:
        chain = filter_asset(
            asset,
            horizon_label,
            exact_expiry,
        )

        metrics_cache[key] = (
            chain,
            calculate_metrics(chain),
        )

    return metrics_cache[key]


def summary_for_horizon(
    horizon_label,
):
    rows = []

    for asset in ASSETS:
        chain, metrics = get_metrics(
            asset,
            horizon_label,
        )

        if metrics is None:
            continue

        spot = metrics["spot"]

        call_wall = metrics[
            "call_wall"
        ]

        put_wall = metrics[
            "put_wall"
        ]

        call_walls = metrics.get(
            "call_walls",
            [],
        )

        put_walls = metrics.get(
            "put_walls",
            [],
        )

        def wall_value(
            walls,
            index,
            field,
        ):
            if len(walls) > index:
                return walls[index].get(
                    field,
                    np.nan,
                )

            return np.nan

        rows.append(
            {
                "Ativo": asset,
                "Preço": spot,
                "Gross Gamma": metrics[
                    "gross_gamma_1pct"
                ],
                "Net GEX Proxy": metrics[
                    "net_gex_proxy_1pct"
                ],
                "Status GEX": (
                    "Positivo"
                    if metrics[
                        "net_gex_proxy_1pct"
                    ] >= 0
                    else "Negativo"
                ),
                "Assimetria GEX %": metrics[
                    "gex_asymmetry_pct"
                ],
                "Nível mais próximo": metrics[
                    "nearest_level_label"
                ],
                "Nível mais próximo preço": metrics[
                    "nearest_level"
                ],
                "Dist. Nível %": metrics[
                    "nearest_distance_pct"
                ],
                "Dist. Nível Abs %": metrics[
                    "nearest_abs_distance_pct"
                ],
                "Proximidade": metrics[
                    "proximity_status"
                ],
                "Confluência W1": (
                    "SIM"
                    if metrics[
                        "primary_wall_confluence"
                    ]
                    else "NÃO"
                ),
                "Call Wall": call_wall,
                "Dist. Call Wall %": (
                    (
                        call_wall / spot
                        - 1.0
                    )
                    * 100.0
                    if np.isfinite(
                        call_wall
                    )
                    else np.nan
                ),
                "Call Wall 2": wall_value(
                    call_walls,
                    1,
                    "strike",
                ),
                "Dist. Call Wall 2 %": wall_value(
                    call_walls,
                    1,
                    "distance_pct",
                ),
                "Call Wall 3": wall_value(
                    call_walls,
                    2,
                    "strike",
                ),
                "Dist. Call Wall 3 %": wall_value(
                    call_walls,
                    2,
                    "distance_pct",
                ),
                "Put Wall": put_wall,
                "Dist. Put Wall %": (
                    (
                        put_wall / spot
                        - 1.0
                    )
                    * 100.0
                    if np.isfinite(
                        put_wall
                    )
                    else np.nan
                ),
                "Put Wall 2": wall_value(
                    put_walls,
                    1,
                    "strike",
                ),
                "Dist. Put Wall 2 %": wall_value(
                    put_walls,
                    1,
                    "distance_pct",
                ),
                "Put Wall 3": wall_value(
                    put_walls,
                    2,
                    "strike",
                ),
                "Dist. Put Wall 3 %": wall_value(
                    put_walls,
                    2,
                    "distance_pct",
                ),
                "Qualidade": (
                    metrics[
                        "quality"
                    ]["score"]
                ),
                "Classe": (
                    metrics[
                        "quality"
                    ]["label"]
                ),
                "Séries": (
                    metrics[
                        "series_count"
                    ]
                ),
                "Vencimentos": (
                    metrics[
                        "expiry_count"
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    # Triagem automática: mais perto de nível estrutural principal aparece primeiro.
    # Isto NÃO representa ranking de compra ou venda.
    if (
        not summary.empty
        and "Dist. Nível Abs %" in summary.columns
    ):
        summary = (
            summary.sort_values(
                [
                    "Dist. Nível Abs %",
                    "Qualidade",
                ],
                ascending=[
                    True,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    return summary



# ======================================================================================
# 10) RESUMO MULTI-HORIZONTE PARA A TABELA PRINCIPAL
# ======================================================================================

def compact_nearest_label(label):
    mapping = {
        "Call Wall 1": "Call W1",
        "Put Wall 1": "Put W1",
        "Confluência Call/Put W1": "Call/Put W1",
        "Sem nível": "N/D",
    }
    return mapping.get(
        str(label),
        str(label),
    )


def summary_multi_horizon():
    """Constrói uma linha por ativo com 30/60/90/180 dias simultaneamente.

    A ordenação considera somente a menor distância absoluta a Call W1, Put W1
    ou confluência Call/Put W1 em qualquer horizonte. W2/W3 não entram na triagem.
    """
    rows = []

    for asset in DISPLAY_ASSETS:
        info = ASSET_INFO.get(
            asset,
            {
                "empresa": asset,
                "setor": "—",
            },
        )

        row = {
            "Ativo": asset,
            "Empresa": info["empresa"],
            "Setor": info["setor"],
            "Preço": np.nan,
        }

        best_abs_distance = np.nan
        best_quality = np.nan
        best_horizon = None
        best_status = "SEM DADOS"

        if asset == "BTC-USD":
            # Não fabricamos GEX para BTC-USD. O motor é exclusivamente B3.
            for horizon_label in HORIZON_ORDER:
                short = HORIZON_SHORT[horizon_label]
                row[f"{short} Wall"] = "N/D — sem cadeia GEX B3"
                row[f"{short} Wall Preço"] = np.nan
                row[f"{short} Dist %"] = np.nan
                row[f"{short} Dist Abs %"] = np.nan
                row[f"{short} Status"] = "SEM DADOS"
                row[f"{short} Qualidade"] = np.nan
                row[f"{short} Classe"] = "N/D"
        else:
            for horizon_label in HORIZON_ORDER:
                short = HORIZON_SHORT[horizon_label]
                _chain, metrics = get_metrics(
                    asset,
                    horizon_label,
                )

                if metrics is None:
                    row[f"{short} Wall"] = "N/D"
                    row[f"{short} Wall Preço"] = np.nan
                    row[f"{short} Dist %"] = np.nan
                    row[f"{short} Dist Abs %"] = np.nan
                    row[f"{short} Status"] = "SEM DADOS"
                    row[f"{short} Qualidade"] = np.nan
                    row[f"{short} Classe"] = "N/D"
                    continue

                if not np.isfinite(row["Preço"]):
                    row["Preço"] = float(
                        metrics["spot"]
                    )

                wall_label = compact_nearest_label(
                    metrics["nearest_level_label"]
                )
                wall_price = float(
                    metrics["nearest_level"]
                ) if np.isfinite(
                    metrics["nearest_level"]
                ) else np.nan

                distance_pct = float(
                    metrics["nearest_distance_pct"]
                ) if np.isfinite(
                    metrics["nearest_distance_pct"]
                ) else np.nan

                abs_distance_pct = float(
                    metrics["nearest_abs_distance_pct"]
                ) if np.isfinite(
                    metrics["nearest_abs_distance_pct"]
                ) else np.nan

                quality_score = float(
                    metrics["quality"]["score"]
                ) if np.isfinite(
                    metrics["quality"]["score"]
                ) else np.nan

                row[f"{short} Wall"] = wall_label
                row[f"{short} Wall Preço"] = wall_price
                row[f"{short} Dist %"] = distance_pct
                row[f"{short} Dist Abs %"] = abs_distance_pct
                row[f"{short} Status"] = metrics["proximity_status"]
                row[f"{short} Qualidade"] = quality_score
                row[f"{short} Classe"] = metrics["quality"]["label"]

                if np.isfinite(abs_distance_pct) and (
                    not np.isfinite(best_abs_distance)
                    or abs_distance_pct < best_abs_distance
                    or (
                        np.isclose(
                            abs_distance_pct,
                            best_abs_distance,
                            atol=1e-12,
                            rtol=0.0,
                        )
                        and np.isfinite(quality_score)
                        and (
                            not np.isfinite(best_quality)
                            or quality_score > best_quality
                        )
                    )
                ):
                    best_abs_distance = abs_distance_pct
                    best_quality = quality_score
                    best_horizon = horizon_label
                    best_status = metrics["proximity_status"]

        row["Melhor Dist Abs %"] = best_abs_distance
        row["Melhor Qualidade"] = best_quality
        row["Melhor Horizonte"] = best_horizon or "—"
        row["Melhor Status"] = best_status
        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = (
            summary.sort_values(
                [
                    "Melhor Dist Abs %",
                    "Melhor Qualidade",
                    "Ativo",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    return summary


# ======================================================================================
# 11) PACOTE DE DIAGNÓSTICO OPCIONAL
# ======================================================================================

EXPORT_DIR = INTEGRATED_DIR / "exports"


def build_export_package():
    """Cria pacote opcional de diagnóstico do radar multi-horizonte de Walls."""
    if EXPORT_DIR.exists():
        shutil.rmtree(
            EXPORT_DIR
        )

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resumo longo por horizonte, útil para auditoria dos valores da V20/V21.
    summary_exports = []

    for horizon_label in HORIZON_ORDER:
        summary = summary_for_horizon(
            horizon_label
        ).copy()
        summary["Horizonte"] = horizon_label
        summary_exports.append(summary)

    if summary_exports:
        all_summaries = pd.concat(
            summary_exports,
            ignore_index=True,
        )
    else:
        all_summaries = pd.DataFrame()

    all_summaries.to_csv(
        EXPORT_DIR
        / "resumo_por_horizonte.csv",
        index=False,
    )

    summary_multi_horizon().to_csv(
        EXPORT_DIR
        / "resumo_multi_horizonte.csv",
        index=False,
    )

    gex_series.to_csv(
        EXPORT_DIR
        / "gex_series_ativos_b3.csv",
        index=False,
    )

    if (
        "historical_prices"
        in globals()
        and isinstance(
            historical_prices,
            pd.DataFrame,
        )
        and not historical_prices.empty
    ):
        historical_prices.to_csv(
            EXPORT_DIR
            / "historico_precos_b3.csv",
            index=False,
        )

    metadata_panel = {
        "project": "GEX Radar Brasil",
        "version": "V34_Faixas_DTE_Exclusivas",
        "mode": "integrated_colab",
        "reference_date": str(
            REFERENCE_DATE.date()
        ),
        "assets_b3": ASSETS,
        "display_assets": DISPLAY_ASSETS,
        "btc_usd_gex": "N/D — sem cadeia de opções B3 compatível com este motor",
        "horizons": HORIZON_ORDER,
        "horizon_selection": {
            "30 dias": "1 <= calendar_days <= 30",
            "60 dias": "31 <= calendar_days <= 60",
            "90 dias": "61 <= calendar_days <= 90",
            "180 dias": "91 <= calendar_days <= 180",
            "cumulative": False,
            "overlap": False,
        },
        "max_base_days": MAX_BASE_DAYS,
        "risk_free_rate_assumption": RISK_FREE_RATE,
        "source_files": {
            "instruments": metadata.get(
                "instrument_file"
            ),
            "prices": metadata.get(
                "price_file"
            ),
            "reference": metadata.get(
                "reference_file"
            ),
        },
        "net_gex_hypothesis": (
            "Calls positivas e puts negativas. "
            "É uma proxy; não representa a posição "
            "observada dos formadores de mercado."
        ),
        "gamma_flip_internal": {
            "calculated": True,
            "displayed_as_attention_level": False,
            "displayed_in_charts": False,
        },
        "open_interest_treatment": (
            "Open interest do PriceReport usado diretamente. "
            "O lote de alocação não é multiplicado."
        ),
        "exercise_style_note": (
            "AMER/EURO são estilos de exercício de opções "
            "negociadas na B3. Não indicam mercado dos EUA."
        ),
        "walls": {
            "detail_count": NUM_WALLS_DETALHE,
            "selection": (
                "Máximos locais de Gross Gamma, ordenados por concentração, "
                "com separação mínima baseada na malha típica de strikes."
            ),
            "main_table": (
                "Cada horizonte usa o nível mais próximo somente entre Call W1, "
                "Put W1 e confluência Call/Put W1. W2/W3 ficam no detalhe."
            ),
        },
        "multi_horizon_sort": (
            "Menor distância absoluta a uma Wall W1 entre 30/60/90/180 dias; "
            "em empate, maior qualidade do mesmo horizonte. Não é sinal de trade."
        ),
        "price_history": {
            "source": "B3 COTAHIST",
            "mapping": {
                "30 dias": 30,
                "60 dias": 60,
                "90 dias": 90,
                "180 dias": 180,
            },
            "note": (
                "Cotações históricas sem ajuste por inflação/proventos, "
                "conforme característica do arquivo público da B3."
            ),
        },
    }

    (
        EXPORT_DIR
        / "metadata_painel.json"
    ).write_text(
        json.dumps(
            metadata_panel,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    package_base = (
        INTEGRATED_DIR
        / (
            "gex_radar_multi_horizonte_"
            f"{REFERENCE_DATE.date()}"
        )
    )

    package_path = Path(
        shutil.make_archive(
            str(package_base),
            "zip",
            root_dir=str(
                EXPORT_DIR
            ),
        )
    )

    return package_path





# ======================================================================================
# 12) ESTILO VISUAL
# ======================================================================================

BASE_CSS = """
<style>
.gex-wrap {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 1600px;
    margin: 0 auto;
}
.gex-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 4px;
}
.gex-subtitle {
    font-size: 14px;
    opacity: 0.78;
    margin-bottom: 16px;
}
.gex-banner {
    border: 1px solid rgba(128,128,128,.35);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 10px 0 16px 0;
}
.gex-quick {
    border: 2px solid rgba(128,128,128,.45);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 12px 0 16px 0;
}
.gex-quick-title {
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 10px;
}
.gex-quick-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 8px 16px;
}
.gex-quick-item {
    padding: 7px 0;
}
.gex-quick-label {
    font-size: 11px;
    opacity: .68;
    text-transform: uppercase;
    letter-spacing: .35px;
}
.gex-quick-value {
    font-size: 17px;
    font-weight: 750;
    margin-top: 2px;
}
.gex-status-pill {
    display: inline-block;
    border: 1px solid rgba(128,128,128,.45);
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 12px;
    font-weight: 750;
}
.gex-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin: 12px 0 18px 0;
}
.gex-card {
    border: 1px solid rgba(128,128,128,.35);
    border-radius: 12px;
    padding: 12px 14px;
}
.gex-card-title {
    font-size: 12px;
    opacity: .72;
    margin-bottom: 4px;
}
.gex-card-value {
    font-size: 21px;
    font-weight: 750;
}
.gex-card-small {
    font-size: 12px;
    opacity: .74;
    margin-top: 4px;
}
.gex-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}
.gex-table th,
.gex-table td {
    border-bottom: 1px solid rgba(128,128,128,.25);
    padding: 8px 7px;
    text-align: right;
    white-space: nowrap;
}
.gex-table th:first-child,
.gex-table td:first-child {
    text-align: left;
}
.gex-table th {
    font-weight: 700;
}
.gex-multi-table {
    min-width: 1900px;
}
.gex-multi-table th {
    text-align: center;
    vertical-align: middle;
}
.gex-multi-table th.base-col {
    text-align: left;
}
.gex-multi-table td.base-col {
    text-align: left;
}
.gex-horizon-head {
    border-left: 2px solid rgba(128,128,128,.35) !important;
    border-right: 2px solid rgba(128,128,128,.35) !important;
}
.gex-horizon-start {
    border-left: 2px solid rgba(128,128,128,.35) !important;
}
.gex-quality-mini {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    opacity: .72;
}
.gex-section-title {
    font-size: 24px;
    font-weight: 800;
    margin: 22px 0 8px 0;
    padding-top: 8px;
    border-top: 2px solid rgba(128,128,128,.30);
}
.gex-method {
    line-height: 1.55;
    font-size: 14px;
}
.gex-note {
    font-size: 12px;
    opacity: .78;
    margin-top: 10px;
}
</style>
"""



def proximity_badge(
    status,
):
    return (
        '<span class="gex-status-pill">'
        + str(status)
        + "</span>"
    )


def quick_read_html(
    metrics,
):
    """Leitura rápida: organização, não sinal de trade."""
    spot = metrics[
        "spot"
    ]

    nearest_level = metrics[
        "nearest_level"
    ]

    nearest_distance_pct = metrics[
        "nearest_distance_pct"
    ]

    nearest_distance = metrics[
        "nearest_distance"
    ]

    nearest_level_text = (
        br_money(
            nearest_level
        )
        if np.isfinite(
            nearest_level
        )
        else "—"
    )

    nearest_distance_text = (
        f"{br_money(nearest_distance)} • "
        f"{br_pct(nearest_distance_pct)}"
        if np.isfinite(
            nearest_distance_pct
        )
        else "—"
    )

    asymmetry_text = (
        br_pct(
            metrics[
                "gex_asymmetry_pct"
            ]
        )
        if np.isfinite(
            metrics[
                "gex_asymmetry_pct"
            ]
        )
        else "—"
    )

    call_level = metrics.get(
        "call_wall",
        np.nan,
    )

    put_level = metrics.get(
        "put_wall",
        np.nan,
    )

    if metrics[
        "primary_wall_confluence"
    ]:
        structure_text = (
            "Confluência Call/Put W1 em "
            + br_money(
                metrics[
                    "primary_wall_confluence_level"
                ]
            )
        )

    elif (
        np.isfinite(
            call_level
        )
        and np.isfinite(
            put_level
        )
    ):
        call_distance_pct = (
            (
                float(call_level)
                / float(spot)
                - 1.0
            )
            * 100.0
        )

        put_distance_pct = (
            (
                float(put_level)
                / float(spot)
                - 1.0
            )
            * 100.0
        )

        if (
            call_distance_pct > 0
            and put_distance_pct > 0
        ):
            relative_text = (
                "Preço abaixo das duas Walls principais"
            )

        elif (
            call_distance_pct < 0
            and put_distance_pct < 0
        ):
            relative_text = (
                "Preço acima das duas Walls principais"
            )

        else:
            relative_text = (
                "Preço entre as duas Walls principais"
            )

        structure_text = (
            "Call W1 "
            + br_money(
                call_level
            )
            + " ("
            + br_pct(
                call_distance_pct
            )
            + ")"
            + " • Put W1 "
            + br_money(
                put_level
            )
            + " ("
            + br_pct(
                put_distance_pct
            )
            + ")"
            + "<br><span style='font-size:12px;opacity:.78'>"
            + relative_text
            + "</span>"
        )

    else:
        structure_text = (
            "Walls principais não identificadas"
        )

    return f"""
      <div class="gex-quick">
        <div class="gex-quick-title">Leitura Rápida</div>

        <div class="gex-quick-grid">

          <div class="gex-quick-item">
            <div class="gex-quick-label">Estrutura Net GEX Proxy</div>
            <div class="gex-quick-value">
              {'POSITIVA' if metrics['net_gex_proxy_1pct'] >= 0 else 'NEGATIVA'}
            </div>
          </div>

          <div class="gex-quick-item">
            <div class="gex-quick-label">Assimetria GEX</div>
            <div class="gex-quick-value">{asymmetry_text}</div>
            <div class="gex-card-small">|Net GEX| ÷ Gross Gamma</div>
          </div>

          <div class="gex-quick-item">
            <div class="gex-quick-label">Nível estrutural mais próximo</div>
            <div class="gex-quick-value">
              {metrics['nearest_level_label']} • {nearest_level_text}
            </div>
            <div class="gex-card-small">
              {nearest_distance_text}
            </div>
          </div>

          <div class="gex-quick-item">
            <div class="gex-quick-label">Proximidade</div>
            <div class="gex-quick-value">
              {proximity_badge(metrics['proximity_status'])}
            </div>
            <div class="gex-card-small">
              {metrics['nearest_concentration_text']}
            </div>
          </div>

          <div class="gex-quick-item">
            <div class="gex-quick-label">Walls principais</div>
            <div class="gex-quick-value">{structure_text}</div>
            <div class="gex-card-small">
              Spot: {br_money(spot)}
            </div>
          </div>

          <div class="gex-quick-item">
            <div class="gex-quick-label">Qualidade</div>
            <div class="gex-quick-value">
              {br_number(metrics['quality']['score'], 1)}
              — {metrics['quality']['label']}
            </div>
            <div class="gex-card-small">
              Leitura estrutural; não é sinal de compra/venda
            </div>
          </div>

        </div>
      </div>
    """


def summary_multi_table_html(
    summary,
):
    if summary.empty:
        return (
            BASE_CSS
            + "<p>Nenhum dado disponível.</p>"
        )

    rows = []

    for row in summary.to_dict(
        orient="records"
    ):
        cells = [
            f"<td class='base-col'><b>{row['Ativo']}</b></td>",
            f"<td class='base-col'>{row['Empresa']}</td>",
            f"<td class='base-col'>{row['Setor']}</td>",
            f"<td>{br_money(row['Preço'])}</td>",
        ]

        for horizon_label in HORIZON_ORDER:
            short = HORIZON_SHORT[
                horizon_label
            ]

            wall_label = row.get(
                f"{short} Wall",
                "N/D",
            )
            wall_price = row.get(
                f"{short} Wall Preço",
                np.nan,
            )
            distance_pct = row.get(
                f"{short} Dist %",
                np.nan,
            )
            status = row.get(
                f"{short} Status",
                "SEM DADOS",
            )
            quality = row.get(
                f"{short} Qualidade",
                np.nan,
            )
            quality_class = row.get(
                f"{short} Classe",
                "N/D",
            )

            if np.isfinite(
                wall_price
            ):
                wall_text = (
                    f"{wall_label} • "
                    f"{br_money(wall_price)}"
                )
            else:
                wall_text = str(
                    wall_label
                )

            if np.isfinite(
                quality
            ):
                quality_text = (
                    f"Q {br_number(quality, 1)} "
                    f"— {quality_class}"
                )
            else:
                quality_text = "Q N/D"

            cells.extend(
                [
                    f"<td class='gex-horizon-start'>{wall_text}</td>",
                    f"<td>{br_pct(distance_pct)}</td>",
                    (
                        "<td>"
                        f"{proximity_badge(status)}"
                        f"<span class='gex-quality-mini'>{quality_text}</span>"
                        "</td>"
                    ),
                ]
            )

        rows.append(
            "<tr>"
            + "".join(cells)
            + "</tr>"
        )

    horizon_headers = "".join(
        f"<th class='gex-horizon-head' colspan='3'>{horizon_label}</th>"
        for horizon_label in HORIZON_ORDER
    )

    subheaders = "".join(
        (
            "<th class='gex-horizon-start'>Wall W1 mais próxima</th>"
            "<th>Dist.</th>"
            "<th>Status / Qualidade</th>"
        )
        for _ in HORIZON_ORDER
    )

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap">
          <div class="gex-title">GEX Radar Brasil</div>
          <div class="gex-subtitle">
            Dados de fechamento da B3 • Data: {REFERENCE_DATE.date().strftime('%d/%m/%Y')}
            • 30, 60, 90 e 180 dias calculados simultaneamente
          </div>
          <div class="gex-banner">
            <b>Radar estrutural de Walls:</b> cada horizonte mostra somente a Wall W1
            mais próxima entre Call W1, Put W1 ou confluência Call/Put W1.
            W2/W3 permanecem no detalhe. Não é sinal de compra/venda.
          </div>
          <div style="overflow-x:auto">
          <table class="gex-table gex-multi-table">
            <thead>
              <tr>
                <th class="base-col" rowspan="2">Ativo</th>
                <th class="base-col" rowspan="2">Empresa</th>
                <th class="base-col" rowspan="2">Setor</th>
                <th rowspan="2">Preço</th>
                {horizon_headers}
              </tr>
              <tr>
                {subheaders}
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
          </div>
          <div class="gex-note">
            Ordenação: primeiro o ativo com menor distância absoluta a uma Wall W1
            em qualquer horizonte; em empate, maior qualidade do mesmo recorte.
            EM CIMA / MUITO PRÓXIMO / PRÓXIMO / DISTANTE medem apenas distância.
            BTC-USD fica visível para espelhar a lista do GARCH, mas não recebe GEX
            porque esta versão usa somente cadeias de opções negociadas na B3.
          </div>
        </div>
        """
    )


def cards_html(


    asset,
    metrics,
    horizon_label,
    exact_expiry,
):
    spot = metrics[
        "spot"
    ]

    call_info = format_level(
        metrics[
            "call_wall"
        ],
        spot,
    )

    put_info = format_level(
        metrics[
            "put_wall"
        ],
        spot,
    )

    quality = metrics[
        "quality"
    ]

    dte_min, dte_max = HORIZON_DTE_RANGES[
        horizon_label
    ]

    expiry_text = (
        f"DTE: {dte_min} a {dte_max} dias"
        if exact_expiry is None
        else pd.Timestamp(
            exact_expiry
        ).strftime("%d/%m/%Y")
    )

    call_w1 = (
        metrics["call_walls"][0]
        if metrics.get("call_walls")
        else None
    )

    put_w1 = (
        metrics["put_walls"][0]
        if metrics.get("put_walls")
        else None
    )

    call_w1_gamma = (
        compact_brl(
            call_w1["gamma_1pct"]
        )
        if call_w1 is not None
        else "—"
    )

    put_w1_gamma = (
        compact_brl(
            put_w1["gamma_1pct"]
        )
        if put_w1 is not None
        else "—"
    )

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap">
          <div class="gex-title">{asset}</div>
          <div class="gex-subtitle">
            {(
                f"Horizonte GEX: {horizon_label} • Histórico: {chart_trading_days_for_horizon(horizon_label)} pregões • Recorte: {expiry_text}"
                if exact_expiry is None
                else f"Vencimento específico: {expiry_text} • Histórico de referência: {chart_trading_days_for_horizon(horizon_label)} pregões"
            )}
          </div>

          <div class="gex-banner">
            <b>Recorte efetivamente calculado:</b>
            {metrics['series_count']} séries •
            {metrics['expiry_count']} vencimentos •
            Gross {compact_brl(metrics['gross_gamma_1pct'])} •
            Net {compact_brl(metrics['net_gex_proxy_1pct'])}
            <br>
            <b>Call W1:</b> {br_money(metrics['call_wall'])}
            • Gamma {call_w1_gamma}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Put W1:</b> {br_money(metrics['put_wall'])}
            • Gamma {put_w1_gamma}
            <br>
            <span style="opacity:.72;">
              O mesmo strike pode permanecer como Wall em horizontes diferentes
              se continuar sendo a maior concentração de Gamma.
            </span>
          </div>

          {quick_read_html(metrics)}

          <div class="gex-grid">

            <div class="gex-card">
              <div class="gex-card-title">Preço</div>
              <div class="gex-card-value">{br_money(spot)}</div>
              <div class="gex-card-small">Fechamento da base</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Net GEX Proxy</div>
              <div class="gex-card-value">{compact_brl(metrics['net_gex_proxy_1pct'])}</div>
              <div class="gex-card-small">
                {'Positivo' if metrics['net_gex_proxy_1pct'] >= 0 else 'Negativo'}
              </div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Gross Gamma</div>
              <div class="gex-card-value">{compact_brl(metrics['gross_gamma_1pct'])}</div>
              <div class="gex-card-small">Movimento de 1% no ativo</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Call Wall principal</div>
              <div class="gex-card-value">{call_info['level']}</div>
              <div class="gex-card-small">
                {call_info['distance']} • {call_info['distance_pct']}
              </div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Put Wall principal</div>
              <div class="gex-card-value">{put_info['level']}</div>
              <div class="gex-card-small">
                {put_info['distance']} • {put_info['distance_pct']}
              </div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Qualidade dos dados</div>
              <div class="gex-card-value">
                {br_number(quality['score'], 1)} — {quality['label']}
              </div>
              <div class="gex-card-small">
                {metrics['series_count']} séries • {metrics['expiry_count']} vencimentos
              </div>
            </div>

          </div>
        </div>
        """
    )



def walls_detail_html(
    metrics,
):
    spot = metrics[
        "spot"
    ]

    def rows_for(
        walls,
        wall_type,
    ):
        rows = []

        for wall in walls:
            strike = wall[
                "strike"
            ]

            label = (
                "Principal"
                if wall["rank"] == 1
                else f"Wall {wall['rank']}"
            )

            rows.append(
                "<tr>"
                f"<td>{wall_type}</td>"
                f"<td>{label}</td>"
                f"<td>{br_money(strike)}</td>"
                f"<td>{br_money(wall['distance'])}</td>"
                f"<td>{br_pct(wall['distance_pct'])}</td>"
                f"<td>{proximity_badge(proximity_status(wall['distance_pct']))}</td>"
                f"<td>{br_pct(wall['share_pct'])}</td>"
                f"<td>{compact_brl(wall['gamma_1pct'])}</td>"
                "</tr>"
            )

        return rows

    rows = []

    rows.extend(
        rows_for(
            metrics.get(
                "call_walls",
                [],
            ),
            "Call",
        )
    )

    rows.extend(
        rows_for(
            metrics.get(
                "put_walls",
                [],
            ),
            "Put",
        )
    )

    if not rows:
        return (
            BASE_CSS
            + "<p>Nenhuma Wall identificada no recorte.</p>"
        )

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap">
          <div class="gex-title" style="font-size:22px">Walls do recorte</div>
          <div class="gex-subtitle">
            Spot: {br_money(spot)} • Até {NUM_WALLS_DETALHE} concentrações distintas de calls e puts
          </div>
          <div style="overflow-x:auto">
          <table class="gex-table">
            <thead>
              <tr>
                <th>Tipo</th>
                <th>Ranking</th>
                <th>Nível</th>
                <th>Distância</th>
                <th>Distância %</th>
                <th>Proximidade</th>
                <th>Participação no Gamma do lado</th>
                <th>Gross Gamma</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
          </div>
          <div class="gex-note">
            O ranking privilegia máximos locais de Gross Gamma e evita classificar
            strikes praticamente colados como Walls diferentes.
          </div>
        </div>
        """
    )


# ======================================================================================
# 12) GRÁFICOS — MATPLOTLIB / PNG
# ======================================================================================

# Esta versão não depende de JavaScript para mostrar gráficos.
# Os widgets continuam interativos, mas cada mudança de filtro regenera
# uma imagem PNG dentro do Colab. Isso evita o espaço em branco observado
# no tablet quando o Plotly era renderizado por callback.


def axis_brl_formatter(value, _position=None):
    """Formatação compacta em R$ para eixos de GEX."""
    absolute = abs(float(value))
    sign = "-" if value < 0 else ""

    if absolute >= 1_000_000_000:
        return (
            f"{sign}R$ "
            f"{absolute / 1_000_000_000:.1f} bi"
        )

    if absolute >= 1_000_000:
        return (
            f"{sign}R$ "
            f"{absolute / 1_000_000:.1f} mi"
        )

    if absolute >= 1_000:
        return (
            f"{sign}R$ "
            f"{absolute / 1_000:.1f} mil"
        )

    return (
        f"{sign}R$ "
        f"{absolute:.0f}"
    )


def default_chart_colors():
    """
    Usa somente o ciclo padrão do Matplotlib.
    Assim não dependemos de uma paleta fixa do código.
    """
    colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()
        .get(
            "color",
            [],
        )
    )

    if not colors:
        return [
            None,
            None,
            None,
            None,
        ]

    while len(colors) < 4:
        colors = colors + colors

    return colors[:4]


def add_price_level_lines(
    ax,
    metrics,
):
    """Adiciona Spot e Walls sem duplicar linhas no mesmo strike.

    Quando calls e puts compartilham o mesmo strike, o gráfico desenha apenas
    uma linha para esse nível e a legenda informa todas as Walls coincidentes.
    A tabela detalhada das Walls continua mostrando cada Call/Put separadamente.
    O Gamma Flip continua calculado, mas não é desenhado no gráfico de preço.
    """
    spot = float(
        metrics[
            "spot"
        ]
    )

    colors = default_chart_colors()

    call_color = colors[0]
    put_color = colors[1]
    spot_color = colors[3]

    ax.axhline(
        spot,
        linestyle="-",
        linewidth=1.6,
        color=spot_color,
        label=(
            f"Spot {spot:.2f}"
        ),
    )

    wall_linestyles = {
        1: "-",
        2: "--",
        3: ":",
    }

    wall_widths = {
        1: 2.2,
        2: 1.6,
        3: 1.4,
    }

    # Agrupar visualmente por centavo. Isso NÃO altera o cálculo.
    grouped_levels = {}

    for side, walls in (
        (
            "Call",
            metrics.get(
                "call_walls",
                [],
            ),
        ),
        (
            "Put",
            metrics.get(
                "put_walls",
                [],
            ),
        ),
    ):
        for wall in walls:
            strike = float(
                wall[
                    "strike"
                ]
            )

            key = round(
                strike,
                2,
            )

            grouped_levels.setdefault(
                key,
                [],
            ).append(
                {
                    "side": side,
                    "rank": int(
                        wall[
                            "rank"
                        ]
                    ),
                }
            )

    for strike_key in sorted(
        grouped_levels
    ):
        entries = grouped_levels[
            strike_key
        ]

        labels = [
            f"{entry['side']} W{entry['rank']}"
            for entry in entries
        ]

        min_rank = min(
            entry[
                "rank"
            ]
            for entry in entries
        )

        has_call = any(
            entry[
                "side"
            ] == "Call"
            for entry in entries
        )

        has_put = any(
            entry[
                "side"
            ] == "Put"
            for entry in entries
        )

        if (
            has_call
            and has_put
        ):
            line_color = call_color
        elif has_call:
            line_color = call_color
        else:
            line_color = put_color

        distance_pct = (
            (
                float(strike_key)
                / spot
                - 1.0
            )
            * 100.0
        )

        ax.axhline(
            float(
                strike_key
            ),
            linestyle=(
                wall_linestyles.get(
                    min_rank,
                    ":",
                )
            ),
            linewidth=(
                wall_widths.get(
                    min_rank,
                    1.3,
                )
            ),
            color=line_color,
            label=(
                " + ".join(
                    labels
                )
                + f" {strike_key:.2f} "
                + f"({distance_pct:+.2f}%)"
            ),
        )

    # Gamma Flip permanece apenas no cálculo interno e não é desenhado.


def plot_price_with_gex_levels(
    asset,
    metrics,
    trading_days,
    horizon_label,
    exact_expiry=None,
):
    """
    Candles do ativo com Spot, até 3 Call Walls e até 3 Put Walls.

    O Gamma Flip continua sendo calculado, mas não é desenhado neste gráfico.
    Retorna uma Figure do Matplotlib, depois convertida em PNG.
    """
    if (
        "historical_prices"
        not in globals()
        or historical_prices.empty
    ):
        return None

    history = (
        historical_prices[
            historical_prices[
                "ticker"
            ].eq(asset)
            & historical_prices[
                "date"
            ].le(
                REFERENCE_DATE
            )
        ]
        .sort_values(
            "date"
        )
        .tail(
            int(trading_days)
        )
        .copy()
    )

    if history.empty:
        return None

    price_frame = (
        history[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .dropna()
        .copy()
    )

    if price_frame.empty:
        return None

    price_frame = (
        price_frame
        .set_index("date")
        .rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )
    )

    # Import tardio: o painel de confluência não usa este gráfico no startup.
    # Mantém a função disponível sem carregar Matplotlib/mplfinance durante o pico de memória B3.
    import mplfinance as mpf

    fig, axes = mpf.plot(
        price_frame,
        type="candle",
        returnfig=True,
        figsize=(12.5, 7.4),
        volume=False,
        xrotation=0,
        datetime_format="%d/%m",
        tight_layout=False,
        ylabel="Preço (R$)",
    )

    ax = axes[0]

    add_price_level_lines(
        ax,
        metrics,
    )

    # Título fora da área das velas.
    fig.suptitle(
        (
            f"{asset} — {int(trading_days)} pregões | "
            f"{gex_scope_text(horizon_label, exact_expiry)}"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    # Legenda fora da área principal do preço para não cobrir candles/níveis.
    handles, labels = (
        ax.get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                0.015,
            ),
            fontsize=8,
            ncol=3,
            frameon=True,
        )

    # Reservar espaço para título e legenda.
    fig.subplots_adjust(
        top=0.90,
        bottom=0.18,
        left=0.08,
        right=0.98,
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    return fig


def plot_net_gex_by_strike(
    asset,
    metrics,
):
    by_strike = metrics[
        "by_strike"
    ].copy()

    spot = float(
        metrics[
            "spot"
        ]
    )

    visible = by_strike[
        by_strike[
            "strike"
        ].between(
            spot * 0.70,
            spot * 1.30,
        )
    ].copy()

    if visible.empty:
        visible = (
            by_strike.copy()
        )

    if visible.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(12.5, 6.4)
    )

    x = visible[
        "strike"
    ].to_numpy(
        dtype=float
    )

    y = visible[
        "net_gex_proxy_1pct"
    ].to_numpy(
        dtype=float
    )

    if len(x) > 1:
        positive_diffs = np.diff(
            np.sort(
                np.unique(x)
            )
        )

        positive_diffs = (
            positive_diffs[
                positive_diffs > 0
            ]
        )

        bar_width = (
            float(
                np.median(
                    positive_diffs
                )
            )
            * 0.75
            if len(
                positive_diffs
            )
            else max(
                spot * 0.003,
                0.01,
            )
        )
    else:
        bar_width = max(
            spot * 0.003,
            0.01,
        )

    ax.bar(
        x,
        y,
        width=bar_width,
        label="Net GEX Proxy",
    )

    colors = default_chart_colors()

    ax.axvline(
        spot,
        linestyle="--",
        linewidth=1.5,
        color=colors[3],
        label=(
            f"Spot {spot:.2f}"
        ),
    )

    if np.isfinite(
        metrics[
            "call_wall"
        ]
    ):
        ax.axvline(
            float(
                metrics[
                    "call_wall"
                ]
            ),
            linestyle=":",
            linewidth=1.8,
            color=colors[0],
            label=(
                "Call Wall principal "
                f"{metrics['call_wall']:.2f}"
            ),
        )

    if np.isfinite(
        metrics[
            "put_wall"
        ]
    ):
        ax.axvline(
            float(
                metrics[
                    "put_wall"
                ]
            ),
            linestyle=":",
            linewidth=1.8,
            color=colors[1],
            label=(
                "Put Wall principal "
                f"{metrics['put_wall']:.2f}"
            ),
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        f"{asset} — Net GEX Proxy por strike"
    )

    ax.set_xlabel(
        "Strike"
    )

    ax.set_ylabel(
        "Exposição para movimento de 1%"
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            axis_brl_formatter
        )
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    return fig


def plot_gross_gamma_calls_puts(
    asset,
    metrics,
):
    by_strike = metrics[
        "by_strike"
    ].copy()

    spot = float(
        metrics[
            "spot"
        ]
    )

    visible = by_strike[
        by_strike[
            "strike"
        ].between(
            spot * 0.70,
            spot * 1.30,
        )
    ].copy()

    if visible.empty:
        visible = (
            by_strike.copy()
        )

    if visible.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(12.5, 6.4)
    )

    x = visible[
        "strike"
    ].to_numpy(
        dtype=float
    )

    call_values = visible[
        "call_gex_1pct"
    ].to_numpy(
        dtype=float
    )

    put_values = visible[
        "put_gex_1pct"
    ].to_numpy(
        dtype=float
    )

    if len(x) > 1:
        positive_diffs = np.diff(
            np.sort(
                np.unique(x)
            )
        )

        positive_diffs = (
            positive_diffs[
                positive_diffs > 0
            ]
        )

        base_width = (
            float(
                np.median(
                    positive_diffs
                )
            )
            if len(
                positive_diffs
            )
            else max(
                spot * 0.003,
                0.01,
            )
        )
    else:
        base_width = max(
            spot * 0.003,
            0.01,
        )

    bar_width = (
        base_width * 0.38
    )

    ax.bar(
        x - bar_width / 2.0,
        call_values,
        width=bar_width,
        label="Gross Gamma Calls",
    )

    ax.bar(
        x + bar_width / 2.0,
        put_values,
        width=bar_width,
        label="Gross Gamma Puts",
    )

    ax.axvline(
        spot,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"Spot {spot:.2f}"
        ),
    )

    ax.set_title(
        f"{asset} — Gross Gamma por strike"
    )

    ax.set_xlabel(
        "Strike"
    )

    ax.set_ylabel(
        "Exposição para movimento de 1%"
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            axis_brl_formatter
        )
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    return fig


def plot_by_expiry(
    asset,
    chain,
):
    if chain.empty:
        return None

    expiry_table = (
        chain.groupby(
            "maturity_date",
            as_index=False,
        )
        .agg(
            gross_gamma_1pct=(
                "gross_gamma_1pct",
                "sum",
            ),
            net_gex_proxy_1pct=(
                "signed_gex_proxy_1pct",
                "sum",
            ),
            open_interest=(
                "open_interest",
                "sum",
            ),
            series_count=(
                "symbol",
                "count",
            ),
        )
        .sort_values(
            "maturity_date"
        )
    )

    if expiry_table.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(12.5, 6.3)
    )

    dates = pd.to_datetime(
        expiry_table[
            "maturity_date"
        ]
    )

    ax.bar(
        dates,
        expiry_table[
            "net_gex_proxy_1pct"
        ],
        width=3.2,
        label="Net GEX Proxy",
    )

    ax.plot(
        dates,
        expiry_table[
            "gross_gamma_1pct"
        ],
        marker="o",
        linewidth=2.0,
        label="Gross Gamma",
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        f"{asset} — Gamma por vencimento"
    )

    ax.set_xlabel(
        "Vencimento"
    )

    ax.set_ylabel(
        "Exposição para movimento de 1%"
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            axis_brl_formatter
        )
    )

    fig.autofmt_xdate(
        rotation=35
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    return fig



def matplotlib_figure_to_data_uri(
    figure,
):
    """Converte Figure do Matplotlib em PNG embutível no widgets.HTML.

    Não usa JavaScript e não cria um Output novo no notebook.
    """
    if figure is None:
        return None

    buffer = io.BytesIO()

    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=145,
            bbox_inches="tight",
        )

        buffer.seek(0)

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode(
            "ascii"
        )

        return (
            "data:image/png;base64,"
            + encoded
        )

    finally:
        buffer.close()
        plt.close(
            figure
        )


def responsive_png_html(
    figure,
    alt_text="Gráfico GEX",
):
    """HTML responsivo para uma Figure Matplotlib."""
    data_uri = (
        matplotlib_figure_to_data_uri(
            figure
        )
    )

    if not data_uri:
        return ""

    return (
        '<div style="width:100%;margin:6px 0 10px 0;">'
        f'<img src="{data_uri}" alt="{alt_text}" '
        'style="display:block;width:100%;max-width:100%;height:auto;'
        'border:0;margin:0 auto;" />'
        "</div>"
    )



# ======================================================================================
# 13) TABELA DE SÉRIES
# ======================================================================================

def series_table_html(
    chain,
    limit=80,
):
    if chain.empty:
        return (
            BASE_CSS
            + "<p>Nenhuma série encontrada.</p>"
        )

    view = chain.copy()

    view["gex_abs"] = (
        view[
            "gross_gamma_1pct"
        ].abs()
    )

    view = (
        view
        .sort_values(
            "gex_abs",
            ascending=False,
        )
        .head(limit)
    )

    rows = []

    for row in view.itertuples():
        iv_pct = (
            row.iv_used * 100.0
            if np.isfinite(
                row.iv_used
            )
            else np.nan
        )

        rows.append(
            "<tr>"
            f"<td>{row.symbol}</td>"
            f"<td>{row.option_type_display}</td>"
            f"<td>{row.exercise_style_display}</td>"
            f"<td>{br_money(row.strike)}</td>"
            f"<td>{row.maturity_date.strftime('%d/%m/%Y')}</td>"
            f"<td>{br_number(row.open_interest, 0)}</td>"
            f"<td>{br_money(row.selected_option_price)}</td>"
            f"<td>{row.option_price_source}</td>"
            f"<td>{br_pct(iv_pct)}</td>"
            f"<td>{compact_brl(row.gross_gamma_1pct)}</td>"
            f"<td>{compact_brl(row.signed_gex_proxy_1pct)}</td>"
            "</tr>"
        )

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap">
          <div style="overflow-x:auto">
          <table class="gex-table">
            <thead>
              <tr>
                <th>Série</th>
                <th>Tipo</th>
                <th>Exercício</th>
                <th>Strike</th>
                <th>Vencimento</th>
                <th>Open Interest</th>
                <th>Preço usado</th>
                <th>Fonte do preço</th>
                <th>IV usada</th>
                <th>Gross Gamma</th>
                <th>Net GEX contrib.</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
          </div>
          <div class="gex-note">
            Exibindo as {min(limit, len(view))} séries com maior Gross Gamma no recorte selecionado.
          </div>
        </div>
        """
    )


# ======================================================================================
# 14) PAINEL DE QUALIDADE
# ======================================================================================

def quality_html(
    metrics,
):
    q = metrics[
        "quality"
    ]

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap">
          <div class="gex-title">Qualidade dos dados</div>
          <div class="gex-grid">

            <div class="gex-card">
              <div class="gex-card-title">Nota</div>
              <div class="gex-card-value">{br_number(q['score'], 1)} — {q['label']}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">OI com Gamma válido</div>
              <div class="gex-card-value">{br_pct(q['gamma_oi_coverage_pct'])}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">OI com IV válida</div>
              <div class="gex-card-value">{br_pct(q['iv_oi_coverage_pct'])}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">OI com preço de mercado</div>
              <div class="gex-card-value">{br_pct(q['market_price_oi_share_pct'])}</div>
              <div class="gex-card-small">Midpoint ou último negócio</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">OI com IV recalculada</div>
              <div class="gex-card-value">{br_pct(q['iv_market_oi_share_pct'])}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Séries negociadas no dia</div>
              <div class="gex-card-value">{br_pct(q['traded_series_share_pct'])}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">Vencimentos com paridade suficiente</div>
              <div class="gex-card-value">{br_pct(q['parity_expiry_share_pct'])}</div>
            </div>

            <div class="gex-card">
              <div class="gex-card-title">OI em exercício no vencimento</div>
              <div class="gex-card-value">{br_pct(q['exercise_at_expiry_oi_share_pct'])}</div>
            </div>

          </div>
        </div>
        """
    )


# ======================================================================================
# 15) METODOLOGIA
# ======================================================================================

def methodology_html():
    """Metodologia e hipóteses do radar estrutural multi-horizonte de Walls."""
    rate_pct = (
        RISK_FREE_RATE * 100.0
    )

    return (
        BASE_CSS
        + f"""
        <div class="gex-wrap gex-method">
          <div class="gex-title">Metodologia</div>

          <p><b>Mercado:</b> o cálculo GEX usa somente instrumentos negociados na B3 presentes na base validada. BTC-USD aparece apenas para espelhar a lista visual do painel GARCH e fica como N/D, sem cálculo GEX nesta versão.</p>

          <p>
            <b>Fontes:</b> Cadastro de Instrumentos, PriceReport e Prêmio de Referência
            da própria B3, usando dados de fechamento da última sessão completa encontrada.
          </p>

          <p>
            <b>Estilo de exercício:</b> quando a base mostra AMER, isso significa apenas
            que a opção pode admitir exercício antecipado. Não significa mercado dos Estados Unidos.
            EURO significa exercício no vencimento.
          </p>

          <p>
            <b>Gross Gamma:</b> exposição agregada calculada por série e somada por strike.
            A unidade utilizada é a variação aproximada, em reais, do delta agregado para
            um movimento de 1% no ativo.
          </p>

          <p>
            <b>Net GEX Proxy:</b> calls recebem sinal positivo e puts sinal negativo.
            Essa é uma convenção de cálculo. Os dados públicos da B3 não informam a
            posição direcional dos formadores de mercado, portanto o indicador não é
            dealer Gamma observado.
          </p>

          <p>
            <b>Call Walls:</b> a Call W1 é a maior concentração relevante de Gross Gamma
            das calls no recorte. O detalhe mantém até {NUM_WALLS_DETALHE} concentrações
            distintas. O algoritmo prioriza máximos locais, ordena pelo Gross Gamma e
            exige separação mínima baseada na própria malha de strikes.
          </p>

          <p>
            <b>Put Walls:</b> seguem a mesma regra das Call Walls. W1 participa da
            triagem principal; W2 e W3 permanecem no detalhe e no gráfico de preço.
          </p>

          <p>
            <b>Gamma Flip:</b> continua calculado internamente por compatibilidade do
            motor já validado. Não participa da tabela principal, da ordenação, dos cards
            nem dos gráficos desta versão. A leitura operacional fica focada nas Walls.
          </p>

          <p>
            <b>Assimetria GEX:</b> |Net GEX Proxy| dividido pelo Gross Gamma, em
            percentual. Mede quanto da exposição bruta permanece após a compensação
            entre o proxy positivo das calls e o proxy negativo das puts. Não mede
            direção real dos dealers.
          </p>

          <p>
            <b>Nível estrutural principal mais próximo:</b> dentro de cada horizonte,
            compara somente Call W1 e Put W1. Quando as duas Walls principais estão no
            mesmo centavo, o painel as trata como uma única confluência Call/Put W1.
          </p>

          <p>
            <b>Classificação de proximidade:</b>
            até {PROXIMIDADE_EM_CIMA_PCT:.2f}% = EM CIMA DO NÍVEL;
            acima de {PROXIMIDADE_EM_CIMA_PCT:.2f}% até {PROXIMIDADE_MUITO_PROXIMO_PCT:.2f}% = MUITO PRÓXIMO;
            acima de {PROXIMIDADE_MUITO_PROXIMO_PCT:.2f}% até {PROXIMIDADE_PROXIMO_PCT:.2f}% = PRÓXIMO;
            acima de {PROXIMIDADE_PROXIMO_PCT:.2f}% = DISTANTE.
            Essa classificação representa somente distância matemática ao nível.
          </p>

          <p>
            <b>Multi-horizonte:</b> a base mantém as séries válidas de 1 a 180 dias e
            divide esse universo em quatro faixas exclusivas de DTE em dias corridos:
            1–30, 31–60, 61–90 e 91–180 dias. Os quatro recortes são independentes,
            não cumulativos e não se sobrepõem; cada série entra em no máximo um horizonte.
            IV e Gamma são calculados uma vez por série; o painel não baixa nem recalcula
            os arquivos da B3 quatro vezes. Em cada faixa, o painel recalcula os agregados,
            Gross Gamma, Net GEX Proxy, Walls, Assimetria e Qualidade correspondentes.
          </p>

          <p>
            <b>Ordenação da tabela:</b> procura a menor distância absoluta a uma Wall W1
            entre 30, 60, 90 e 180 dias. W2/W3 não alteram a posição do ativo na tabela.
            Em empate de distância, usa a maior qualidade do mesmo recorte. Isso é uma
            regra de triagem, não um ranking de compra ou venda.
          </p>

          <p>
            <b>Confluências no gráfico:</b> quando uma Call Wall e uma Put Wall caem
            no mesmo strike, o gráfico desenha uma única linha para aquele preço e reúne
            os respectivos rankings na legenda. A tabela detalhada continua mostrando
            Call e Put separadamente.
          </p>

          <p>
            <b>Volatilidade implícita:</b> o motor recalcula IV para opções com preço de
            mercado confiável quando possível. Nas demais séries válidas, utiliza a
            volatilidade publicada no arquivo de prêmio de referência da B3.
          </p>

          <p>
            <b>Taxa livre de risco desta base:</b> {br_number(rate_pct, 2)}% a.a.
            como hipótese plana atual. A estrutura continua preparada para futura
            substituição por curva DI por vencimento, sem fazer essa alteração agora.
          </p>

          <p>
            <b>Opções com exercício antecipado:</b> o Gamma é calculado por aproximação
            BSM. Isso permanece explicitado e não é apresentado como Gamma exato de um
            modelo específico para exercício antecipado.
          </p>

          <p>
            <b>Open interest:</b> o valor do PriceReport é usado diretamente.
            O lote de alocação não é multiplicado novamente.
          </p>

          <p>
            <b>Gráficos:</b> 30 dias mostram 30 pregões com Walls calculadas com opções
            de DTE entre 1 e 30 dias; 60 mostram 60 pregões com DTE de 31 a 60; 90 mostram
            90 pregões com DTE de 61 a 90; e 180 mostram 180 pregões com DTE de 91 a 180.
            O histórico é o COTAHIST público da B3 e permanece sem ajuste por inflação
            ou proventos.
          </p>

          <p>
            <b>Vencimento específico:</b> continua disponível apenas como ferramenta de
            investigação do ativo. Ao escolher uma data, o cálculo usa somente as séries
            daquele vencimento dentro do universo máximo de 180 dias.
          </p>

          <p>
            <b>Qualidade:</b> combina cobertura de Gamma/IV por open interest,
            participação de preços de mercado, qualidade da estimativa de forward/carry,
            negociação no dia e participação de contratos de exercício no vencimento.
          </p>

          <p>
            <b>Data:</b> {REFERENCE_DATE.date().strftime('%d/%m/%Y')}.
            O painel usa dados de fechamento. O botão Atualizar consulta novamente a B3;
            não é tempo real.
          </p>

        </div>
        """
    )




# ============================================================
# INICIALIZAÇÃO DO RUNTIME
# ============================================================
def initialize_runtime(series: pd.DataFrame, meta: dict, history: pd.DataFrame | None = None) -> None:
    """Instala no módulo a base carregada pela interface e limpa métricas agregadas."""
    global gex_series, historical_prices, metadata
    global REFERENCE_DATE, RISK_FREE_RATE, MAX_BASE_DAYS, ASSETS, DISPLAY_ASSETS

    metadata = dict(meta or {})
    gex_series = prepare_panel_data(series)
    REFERENCE_DATE = pd.Timestamp(metadata.get("reference_date", current_brazil_date()))
    RISK_FREE_RATE = float(metadata.get("risk_free_rate_assumption", TAXA_LIVRE_RISCO_ANUAL))
    MAX_BASE_DAYS = int(metadata.get("max_days_to_expiry", MAX_DIAS_ATE_VENCIMENTO))
    ASSETS = ATIVOS_B3.copy()
    DISPLAY_ASSETS = ATIVOS_EXIBICAO.copy()
    historical_prices = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
    invalidate_metrics_cache()


def load_complete_bundle(force: bool = False):
    """Executa a mesma cadeia validada da V21 e devolve séries, metadados e COTAHIST."""
    series, meta = run_full_pipeline(force=force)
    prepared = prepare_panel_data(series)
    reference_date = pd.Timestamp(meta["reference_date"])
    history = load_b3_price_history(reference_date, ATIVOS_B3)
    return prepared, meta, history
