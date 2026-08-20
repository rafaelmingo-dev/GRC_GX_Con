# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import warnings
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
try:
    import exchange_calendars as xcals
except Exception:
    xcals = None
warnings.filterwarnings("ignore")
CAL_B3 = None
ERRO_CALENDARIO_B3 = None
if xcals is not None:
    try:
        CAL_B3 = xcals.get_calendar("BVMF")
    except Exception as erro:
        ERRO_CALENDARIO_B3 = str(erro)
else:
    ERRO_CALENDARIO_B3 = "exchange_calendars não pôde ser importado."

ATIVOS: dict[str, dict[str, str]] = {'PETR4': {'nome': 'Petrobras PN', 'setor': 'Petróleo e Gás', 'mercado': 'B3', 'moeda': 'BRL'}, 'VALE3': {'nome': 'Vale', 'setor': 'Mineração', 'mercado': 'B3', 'moeda': 'BRL'}, 'ITUB4': {'nome': 'Itaú Unibanco', 'setor': 'Bancos', 'mercado': 'B3', 'moeda': 'BRL'}, 'BBDC4': {'nome': 'Bradesco PN', 'setor': 'Bancos', 'mercado': 'B3', 'moeda': 'BRL'}, 'BBAS3': {'nome': 'Banco do Brasil', 'setor': 'Bancos', 'mercado': 'B3', 'moeda': 'BRL'}, 'ITSA4': {'nome': 'Itaúsa PN', 'setor': 'Holding', 'mercado': 'B3', 'moeda': 'BRL'}, 'BBSE3': {'nome': 'BB Seguridade', 'setor': 'Seguros', 'mercado': 'B3', 'moeda': 'BRL'}, 'CXSE3': {'nome': 'Caixa Seguridade', 'setor': 'Seguros', 'mercado': 'B3', 'moeda': 'BRL'}, 'PSSA3': {'nome': 'Porto', 'setor': 'Seguros', 'mercado': 'B3', 'moeda': 'BRL'}, 'ABEV3': {'nome': 'Ambev', 'setor': 'Consumo', 'mercado': 'B3', 'moeda': 'BRL'}, 'WEGE3': {'nome': 'WEG', 'setor': 'Indústria', 'mercado': 'B3', 'moeda': 'BRL'}, 'B3SA3': {'nome': 'B3', 'setor': 'Mercado Financeiro', 'mercado': 'B3', 'moeda': 'BRL'}, 'BPAC11': {'nome': 'BTG Pactual', 'setor': 'Bancos', 'mercado': 'B3', 'moeda': 'BRL'}, 'GGBR4': {'nome': 'Gerdau PN', 'setor': 'Siderurgia', 'mercado': 'B3', 'moeda': 'BRL'}, 'CMIG4': {'nome': 'Cemig PN', 'setor': 'Energia', 'mercado': 'B3', 'moeda': 'BRL'}, 'CPFE3': {'nome': 'CPFL Energia', 'setor': 'Energia', 'mercado': 'B3', 'moeda': 'BRL'}, 'EGIE3': {'nome': 'Engie Brasil', 'setor': 'Energia', 'mercado': 'B3', 'moeda': 'BRL'}, 'EQTL3': {'nome': 'Equatorial Energia', 'setor': 'Energia', 'mercado': 'B3', 'moeda': 'BRL'}, 'SBSP3': {'nome': 'Sabesp', 'setor': 'Saneamento', 'mercado': 'B3', 'moeda': 'BRL'}, 'CPLE3': {'nome': 'Copel', 'setor': 'Energia', 'mercado': 'B3', 'moeda': 'BRL'}, 'BTC-USD': {'nome': 'Bitcoin', 'setor': 'Criptoativos', 'mercado': 'CRIPTO', 'moeda': 'USD'}}

HISTORICO_ANOS = 10

MIN_OBSERVACOES = 500

LIMIAR_PROXIMO_PCT = 1.0

DISTRIBUICAO_GARCH = 'normal'

TIMEZONE_LOCAL = 'America/Sao_Paulo'

PERIODOS_CALCULO = ('MENSAL', 'SEMESTRAL', 'ANUAL')

def agora_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE_LOCAL).tz_localize(None)

def eh_cripto(codigo: str) -> bool:
    return ATIVOS.get(codigo, {}).get('mercado', 'B3') == 'CRIPTO'

def ticker_yahoo(codigo: str) -> str:
    codigo = codigo.upper().strip()
    if eh_cripto(codigo):
        return codigo
    if codigo.endswith('.SA') or codigo.endswith('=X'):
        return codigo
    return f'{codigo}.SA'

def achatar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def normalizar_timestamp(ts: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        try:
            ts = ts.tz_convert(TIMEZONE_LOCAL).tz_localize(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts

def baixar_historico_diario(codigo: str) -> pd.DataFrame:
    ticker = ticker_yahoo(codigo)
    df = yf.download(ticker, period=f'{HISTORICO_ANOS}y', interval='1d', auto_adjust=False, actions=False, progress=False, threads=False)
    df = achatar_colunas(df)
    if df.empty:
        raise ValueError(f'Yahoo Finance não retornou histórico diário para {ticker}.')
    obrigatorias = ['Open', 'High', 'Low', 'Close']
    faltando = [coluna for coluna in obrigatorias if coluna not in df.columns]
    if faltando:
        raise ValueError(f'Colunas ausentes em {ticker}: {faltando}')
    colunas = obrigatorias.copy()
    if 'Adj Close' in df.columns:
        colunas.append('Adj Close')
    df = df[colunas].copy()
    indice = pd.to_datetime(df.index)
    try:
        indice = indice.tz_localize(None)
    except TypeError:
        pass
    df.index = indice
    df = df.dropna(subset=obrigatorias).sort_index()
    if len(df) < MIN_OBSERVACOES:
        raise ValueError(f'Histórico insuficiente para {codigo}: {len(df)} observações.')
    return df

def baixar_preco_mais_recente(codigo: str, historico: pd.DataFrame) -> tuple[float, str, pd.Timestamp]:
    ticker = ticker_yahoo(codigo)
    tentativas = [('1d', '1m'), ('5d', '5m')]
    for periodo, intervalo in tentativas:
        try:
            df = yf.download(ticker, period=periodo, interval=intervalo, auto_adjust=False, actions=False, prepost=False, progress=False, threads=False)
            df = achatar_colunas(df)
            if not df.empty and 'Close' in df.columns:
                serie = pd.to_numeric(df['Close'], errors='coerce').dropna()
                if not serie.empty:
                    return (float(serie.iloc[-1]), intervalo, normalizar_timestamp(serie.index[-1]))
        except Exception:
            pass
    preco = float(historico['Close'].iloc[-1])
    momento = normalizar_timestamp(historico.index[-1])
    return (preco, '1d', momento)

def inicio_periodo(tipo: str, data: Any=None) -> pd.Timestamp:
    if data is None:
        data = agora_local()
    else:
        data = pd.Timestamp(data)
        if data.tzinfo is not None:
            data = data.tz_localize(None)
    tipo = tipo.upper()
    if tipo == 'MENSAL':
        return pd.Timestamp(year=data.year, month=data.month, day=1)
    if tipo == 'SEMESTRAL':
        mes = 1 if data.month <= 6 else 7
        return pd.Timestamp(year=data.year, month=mes, day=1)
    if tipo == 'ANUAL':
        return pd.Timestamp(year=data.year, month=1, day=1)
    raise ValueError(f'Período desconhecido: {tipo}')

def fim_periodo(tipo: str, inicio: Any) -> pd.Timestamp:
    tipo = tipo.upper()
    inicio = inicio_periodo(tipo, inicio)
    if tipo == 'MENSAL':
        return inicio + pd.offsets.MonthEnd(0)
    if tipo == 'SEMESTRAL':
        if inicio.month == 1:
            return pd.Timestamp(year=inicio.year, month=6, day=30)
        return pd.Timestamp(year=inicio.year, month=12, day=31)
    if tipo == 'ANUAL':
        return pd.Timestamp(year=inicio.year, month=12, day=31)
    raise ValueError(f'Período desconhecido: {tipo}')

def rotulo_periodo(tipo: str, inicio: Any) -> str:
    tipo = tipo.upper()
    inicio = inicio_periodo(tipo, inicio)
    if tipo == 'MENSAL':
        return inicio.strftime('%m/%Y')
    if tipo == 'SEMESTRAL':
        numero = 1 if inicio.month == 1 else 2
        return f'{numero}º semestre/{inicio.year}'
    if tipo == 'ANUAL':
        return str(inicio.year)
    raise ValueError(f'Período desconhecido: {tipo}')

def sessoes_b3_no_periodo(tipo: str, inicio: Any) -> int:
    inicio = inicio_periodo(tipo, inicio).normalize()
    fim = fim_periodo(tipo, inicio).normalize()
    if CAL_B3 is not None:
        try:
            sessoes = CAL_B3.sessions_in_range(inicio, fim)
            if len(sessoes) > 0:
                return int(len(sessoes))
        except Exception:
            pass
    return max(1, len(pd.bdate_range(inicio, fim)))

def dias_corridos_no_periodo(tipo: str, inicio: Any) -> int:
    inicio = inicio_periodo(tipo, inicio).normalize()
    fim = fim_periodo(tipo, inicio).normalize()
    return int((fim - inicio).days + 1)

def horizonte_periodo(codigo: str, tipo: str, inicio: Any) -> tuple[int, str]:
    if eh_cripto(codigo):
        return (dias_corridos_no_periodo(tipo, inicio), 'dias')
    return (sessoes_b3_no_periodo(tipo, inicio), 'pregões')

def obter_ajuste_garch(codigo: str, historico: pd.DataFrame, data_corte: Any, ajustes_cache: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    data_corte = pd.Timestamp(data_corte).normalize()
    chave = (codigo, data_corte.strftime('%Y-%m-%d'), HISTORICO_ANOS, DISTRIBUICAO_GARCH)
    if chave in ajustes_cache:
        return ajustes_cache[chave]
    treino = historico.loc[historico.index < data_corte].copy()
    if len(treino) < MIN_OBSERVACOES:
        raise ValueError(f"Histórico insuficiente antes de {data_corte.strftime('%d/%m/%Y')}: {len(treino)} observações.")
    raw_close = treino['Close'].astype(float).dropna()
    preco_base = float(raw_close.iloc[-1])
    data_base = raw_close.index[-1]
    if 'Adj Close' in treino.columns and treino['Adj Close'].notna().sum() > 0:
        serie_retorno = treino['Adj Close'].astype(float).dropna()
    else:
        serie_retorno = raw_close
    retornos = 100.0 * np.log(serie_retorno / serie_retorno.shift(1))
    retornos = retornos.replace([np.inf, -np.inf], np.nan).dropna()
    if len(retornos) < MIN_OBSERVACOES:
        raise ValueError(f'Retornos insuficientes para GARCH(2,2): {len(retornos)}.')
    modelo = arch_model(retornos, mean='Constant', vol='GARCH', p=2, o=0, q=2, dist=DISTRIBUICAO_GARCH, rescale=False)
    ajuste = modelo.fit(disp='off', show_warning=False, options={'maxiter': 1000})
    resultado = {'ajuste': ajuste, 'preco_base': preco_base, 'data_base': data_base, 'data_corte': data_corte, 'convergencia': int(getattr(ajuste, 'convergence_flag', 0))}
    ajustes_cache[chave] = resultado
    return resultado

def calcular_bandas_periodo(codigo: str, historico: pd.DataFrame, tipo: str, periodo_inicio: Any, ajustes_cache: dict[tuple[Any, ...], dict[str, Any]], bandas_cache: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    tipo = tipo.upper()
    periodo_inicio = inicio_periodo(tipo, periodo_inicio)
    chave = (tipo, codigo, periodo_inicio.strftime('%Y-%m-%d'), HISTORICO_ANOS, DISTRIBUICAO_GARCH)
    if chave in bandas_cache:
        return bandas_cache[chave]
    base = obter_ajuste_garch(codigo, historico, periodo_inicio, ajustes_cache)
    horizonte, unidade = horizonte_periodo(codigo, tipo, periodo_inicio)
    previsao = base['ajuste'].forecast(horizon=horizonte, method='analytic', reindex=False)
    variancias = previsao.variance.iloc[-1].to_numpy(dtype=float)
    if len(variancias) != horizonte or not np.all(np.isfinite(variancias)) or np.any(variancias < 0):
        raise ValueError(f'Previsão de variância inválida para {tipo.lower()}.')
    variancia_periodo = float(np.sum(variancias))
    sigma_periodo = math.sqrt(variancia_periodo) / 100.0
    preco_base = base['preco_base']
    resultado = {'codigo': codigo, 'tipo': tipo, 'rotulo': rotulo_periodo(tipo, periodo_inicio), 'periodo_inicio': periodo_inicio, 'periodo_fim': fim_periodo(tipo, periodo_inicio), 'data_base': base['data_base'], 'preco_base': preco_base, 'sigma': sigma_periodo, 'vol_pct': sigma_periodo * 100.0, 'menos_2': preco_base * math.exp(-2.0 * sigma_periodo), 'menos_15': preco_base * math.exp(-1.5 * sigma_periodo), 'mais_15': preco_base * math.exp(1.5 * sigma_periodo), 'mais_2': preco_base * math.exp(2.0 * sigma_periodo), 'horizonte': horizonte, 'unidade_horizonte': unidade, 'convergencia': base['convergencia']}
    bandas_cache[chave] = resultado
    return resultado

def analisar_status(preco: float, bandas: dict[str, Any]) -> tuple[str, str, float, int]:
    if preco >= bandas['mais_2']:
        return ('🚨 ACIMA +2σ', '+2σ', 0.0, 0)
    if preco >= bandas['mais_15']:
        distancia = abs(bandas['mais_2'] / preco - 1.0) * 100.0
        return ('🚨 ACIMA +1,5σ', '+2σ', distancia, 1)
    if preco <= bandas['menos_2']:
        return ('🚨 ABAIXO -2σ', '-2σ', 0.0, 0)
    if preco <= bandas['menos_15']:
        distancia = abs(bandas['menos_2'] / preco - 1.0) * 100.0
        return ('🚨 ABAIXO -1,5σ', '-2σ', distancia, 1)
    distancia_superior = abs(bandas['mais_15'] / preco - 1.0) * 100.0
    distancia_inferior = abs(bandas['menos_15'] / preco - 1.0) * 100.0
    if distancia_superior <= distancia_inferior:
        proxima = '+1,5σ'
        distancia = distancia_superior
    else:
        proxima = '-1,5σ'
        distancia = distancia_inferior
    if distancia <= LIMIAR_PROXIMO_PCT:
        return (f'🟠 PRÓXIMO {proxima}', proxima, distancia, 2)
    return ('🟢 NORMAL', proxima, distancia, 3)
