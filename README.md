# AZIONARIO

Dashboard tecnica su universo doppio **STOXX 600** (Europa, 535 titoli) + **S&P 500** (USA, 503 titoli).

Sostituisce l'architettura del vecchio repo `stoxx600` (fetch client-side via proxy) con lo
stesso schema di `raptor-leva`: backend Python via GitHub Actions che scrive JSON statici,
frontend che legge solo quei JSON (nessuna chiamata a Yahoo Finance/proxy dal browser).

## Struttura

- `index.html` — dashboard (tab Europa/USA, tabella, grafico, export Excel, link regole PDF/TradingView)
- `fetch_azionario.py` — scarica storico Yahoo Finance, calcola gli indicatori, scrive i JSON
- `azionario.json` — riepilogo di tutti i titoli (generato da fetch_azionario.py)
- `data/charts/TICKER.json` — serie storica + indicatori per il grafico di ogni titolo
- `regole/TICKER_Regole.html` — scheda regole operative per titolo (generata dal template)
- `regole_template.html` — template con placeholder `{{...}}` usato per generare le schede
- `tickers_stoxx600.json` / `tickers_sp500.json` — universo titoli
- `.github/workflows/update.yml` — Action che esegue lo script periodicamente

## Indicatori

ER, KAMA Fast/Slow, KAMA gap%, baff (barre sopra/sotto KAMA), SAR con flip tracking,
AO (EMA3−EMA13), RVI, RSI14/RSI5 + cross, ADX, momentum 1M/3M/6M, Volume Ratio.

Zone: `LONG_CONF`, `LONG_EARLY`, `USCITA`, `STOP`, `NEUTRA`.
Segnali: `BUY3`, `BUY2`, `SELL`, `STOP`, `HOLD`, più flag indipendente `Super Best Buy`
(basato su SAR-flip: primo pallino entro 2 barre + AO>0 in miglioramento + volume ratio ≥1.5x
+ movimento giornaliero entro ±4%).

## Nota

Lo script non è stato testato con dati Yahoo Finance live: l'ambiente di sviluppo non ha
accesso in uscita a Yahoo Finance. È stato validato con dati OHLC sintetici (nessun crash,
output coerente). **Verificare il primo run della Action** e segnalare eventuali eccezioni
o ticker che falliscono sistematicamente.
