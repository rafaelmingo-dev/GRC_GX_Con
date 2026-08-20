# GARCH × GEX — Painel de Confluência V3

Terceiro painel independente. Não altera os projetos GARCH Radar ou GEX Radar Brasil.

## Regras
- GARCH Mensal × GEX 30D
- GARCH Semestral × GEX 90D
- GARCH Semestral × GEX 180D
- GARCH Anual sozinho
- GEX 60D fora do cruzamento
- Principal = W1
- Secundária = W2/W3
- Sem score composto ou classificação forte/fraca

## Executar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
Crie um novo repositório e envie:
- app.py
- confluence_core.py
- garch_core.py
- gex_core.py
- requirements.txt
- .streamlit/config.toml

Main file: `app.py`.
