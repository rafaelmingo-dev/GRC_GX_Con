GARCH × GEX — CONFLUÊNCIA V3 — CLOUD ROBUSTA FINAL
Terceiro painel independente. Não substitui nem altera o GARCH Radar ou o GEX Radar Brasil.
Regras preservadas
GARCH Mensal × GEX 30D
GARCH Semestral × GEX 90D
GARCH Semestral × GEX 180D
GARCH Anual sozinho
GEX 60D fora do cruzamento
Principal = W1
Secundária = W2/W3
sem score composto
sem classificação Forte/Moderada/Fraca
sem sinal de compra/venda
Mudança arquitetural para o Streamlit Cloud
O `app.py` abre imediatamente e não dispara o pipeline pesado da B3 no startup.
Na primeira execução:
abra o app;
clique em `Preparar painel agora`;
`panel_worker.py` executa B3/GEX/GARCH em processo separado;
o cache final só é substituído se tudo terminar com sucesso.
Nas próximas execuções, o botão `Atualizar` repete o processo e preserva o último cache válido em caso de falha.
O painel conjunto não carrega o COTAHIST do GEX porque esse histórico não participa dos
cálculos de IV/Gamma/GEX/Walls/confluência utilizados neste painel.
Arquivos
`app.py`
`panel_worker.py`
`confluence_core.py`
`garch_core.py`
`gex_core.py`
`requirements.txt`
`.streamlit/config.toml`
`.gitignore`
Streamlit Community Cloud
Main file: `app.py`
No deploy, selecione Python 3.12 em Advanced settings.
Execução local
```bash
pip install -r requirements.txt
streamlit run app.py
```
