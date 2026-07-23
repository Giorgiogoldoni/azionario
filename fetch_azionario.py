#!/usr/bin/env python3
"""
fetch_azionario.py
Scarica storico Yahoo Finance per l'universo STOXX600 + S&P500, calcola gli
indicatori tecnici standard (ER, KAMA fast/slow, baff, SAR con flip tracking,
AO, RVI, RSI14/RSI5, ADX, momentum 1M/3M/6M), determina zone/segnali
(BUY3/BUY2/SELL/STOP, RSI cross, Super Best Buy basato su SAR-flip) e scrive:
  - azionario.json                  riepilogo di tutti i titoli (per la tabella)
  - data/charts/TICKER.json         serie storica + indicatori (per il grafico)
  - data/charts/index.json          mappa ticker -> file
  - regole/TICKER_Regole.html       scheda regole operative per titolo

Pensato per girare via GitHub Actions (accesso libero a Yahoo Finance).
NOTA: non è stato possibile testarlo con dati Yahoo live in questo ambiente
(rete sandbox senza accesso a query1.finance.yahoo.com) — verificare il primo
run in Actions e segnalare eventuali eccezioni.
"""

import json
import math
import time
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
REGOLE_DIR = ROOT / "regole"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REGOLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parametri indicatori (allineati al template regole EEI_Regole_RAPTOR.html)
# ---------------------------------------------------------------------------
KAMA_FAST_N = 10
KAMA_SLOW_N = 20
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30
ER_N = 10
AO_FAST = 3
AO_SLOW = 13
RVI_N = 4
RSI_FAST_N = 5
RSI_SLOW_N = 14
ADX_N = 14
SAR_STEP = 0.02
SAR_MAX = 0.2
VOL_AVG_N = 20

BATCH_SIZE = 40          # ticker per batch yfinance
SLEEP_BETWEEN_BATCH = 3  # secondi, per non farsi rate-limitare da Yahoo
HISTORY_PERIOD = "18mo"

# Mappa suffisso ticker -> prefisso TradingView (standard richiesto)
TV_SUFFIX_MAP = {
    ".MI": "MIL",
    ".DE": "XETR",
    ".PA": "EURONEXT",
    ".L": "LSE",
    ".AS": "EURONEXT",
    ".BR": "EURONEXT",
    ".LS": "EURONEXT",
    ".MC": "BME",
    ".SW": "SIX",
    ".VX": "SIX",
    ".ST": "OMXSTO",
    ".CO": "OMXCOP",
    ".OL": "OSE",
    ".HE": "OMXHEX",
    ".VI": "VIE",
    ".IR": "ISE",
    ".PR": "PSE",
    ".WA": "GPW",
    ".AT": "ASE",
    ".BUD": "BET",
}


def tv_symbol(ticker: str, exchange_hint: str | None = None) -> str:
    """Costruisce il simbolo TradingView (BORSA:TICKER) da un ticker Yahoo."""
    for suf, tv_ex in TV_SUFFIX_MAP.items():
        if ticker.endswith(suf):
            base = ticker[: -len(suf)].replace("-", ".")
            return f"{tv_ex}:{base}"
    # Nessun suffisso -> titolo USA: serve l'exchange passato esplicitamente
    if exchange_hint:
        return f"{exchange_hint}:{ticker.replace('-', '.')}"
    return ticker


# ---------------------------------------------------------------------------
# Indicatori
# ---------------------------------------------------------------------------

def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.fillna(0)


def kama(close: pd.Series, n: int, fast_sc: int, slow_sc: int) -> pd.Series:
    er = efficiency_ratio(close, n)
    fast_alpha = 2 / (fast_sc + 1)
    slow_alpha = 2 / (slow_sc + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    out = np.full(len(close), np.nan)
    first_valid = n
    if len(close) <= first_valid:
        return pd.Series(out, index=close.index)
    out[first_valid] = close.iloc[first_valid]
    for i in range(first_valid + 1, len(close)):
        prev = out[i - 1]
        if np.isnan(prev):
            prev = close.iloc[i - 1]
        out[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
    return pd.Series(out, index=close.index)


def awesome_oscillator(close: pd.Series, fast=AO_FAST, slow=AO_SLOW) -> pd.Series:
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, n=RVI_N) -> pd.Series:
    num = (close - open_).rolling(n).mean()
    den = (high - low).rolling(n).mean()
    return (num / den.replace(0, np.nan)).fillna(0)


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n=ADX_N) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, step=SAR_STEP, max_af=SAR_MAX):
    n = len(close)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)   # 1 = rialzista, -1 = ribassista
    flip = np.zeros(n, dtype=bool)
    if n < 2:
        return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index), pd.Series(flip, index=close.index)

    trend[0] = 1 if close.iloc[1] >= close.iloc[0] else -1
    sar[0] = low.iloc[0] if trend[0] == 1 else high.iloc[0]
    af = step
    ep = high.iloc[0] if trend[0] == 1 else low.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low.iloc[i - 1], low.iloc[i - 2] if i >= 2 else low.iloc[i - 1])
            if low.iloc[i] < new_sar:
                trend[i] = -1
                flip[i] = True
                sar[i] = ep
                ep = low.iloc[i]
                af = step
            else:
                trend[i] = 1
                sar[i] = new_sar
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_af)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i - 1], high.iloc[i - 2] if i >= 2 else high.iloc[i - 1])
            if high.iloc[i] > new_sar:
                trend[i] = 1
                flip[i] = True
                sar[i] = ep
                ep = high.iloc[i]
                af = step
            else:
                trend[i] = -1
                sar[i] = new_sar
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_af)
    return (pd.Series(sar, index=close.index),
            pd.Series(trend, index=close.index),
            pd.Series(flip, index=close.index))


def bars_since(flag_series: pd.Series) -> int:
    """Numero di barre trascorse dall'ultimo True (0 = oggi)."""
    idx = np.where(flag_series.values)[0]
    if len(idx) == 0:
        return len(flag_series)
    return len(flag_series) - 1 - idx[-1]


def baff_count(price_above_kama: pd.Series) -> int:
    """Barre consecutive con lo stesso stato (sopra/sotto KAMA) fino a oggi."""
    vals = price_above_kama.values
    if len(vals) == 0:
        return 0
    last = vals[-1]
    cnt = 0
    for v in vals[::-1]:
        if v == last:
            cnt += 1
        else:
            break
    return cnt


# ---------------------------------------------------------------------------
# Calcolo indicatori + segnali per un singolo titolo
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < KAMA_SLOW_N + 5:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    close, open_, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]

    er = efficiency_ratio(close, ER_N)
    kama_fast = kama(close, KAMA_FAST_N, KAMA_FAST_SC, KAMA_SLOW_N * 1)
    kama_slow = kama(close, KAMA_SLOW_N, KAMA_FAST_SC, 30)
    ao = awesome_oscillator(close)
    rvi_v = rvi(open_, high, low, close)
    rsi5 = rsi(close, RSI_FAST_N)
    rsi14 = rsi(close, RSI_SLOW_N)
    adx_v = adx(high, low, close)
    sar, sar_trend, sar_flip = parabolic_sar(high, low, close)
    vol_avg = vol.rolling(VOL_AVG_N).mean()
    vol_ratio = (vol / vol_avg.replace(0, np.nan)).fillna(0)

    price_above_kf = close > kama_fast

    i = -1  # ultima barra
    price = float(close.iloc[i])
    kf = float(kama_fast.iloc[i]) if not math.isnan(kama_fast.iloc[i]) else None
    ks = float(kama_slow.iloc[i]) if not math.isnan(kama_slow.iloc[i]) else None
    if kf is None or ks is None:
        return None

    er_v = float(er.iloc[i])
    ao_v = float(ao.iloc[i])
    ao_prev3 = ao.iloc[i - 3:i]
    ao_improving = bool(len(ao_prev3) == 3 and all(np.diff(ao_prev3.values) > 0))
    rvi_val = float(rvi_v.iloc[i])
    rsi5_v = float(rsi5.iloc[i])
    rsi14_v = float(rsi14.iloc[i])
    adx_val = float(adx_v.iloc[i])
    volr = float(vol_ratio.iloc[i])
    baff = baff_count(price_above_kf)
    sar_v = float(sar.iloc[i])
    sar_bullish = price > sar_v
    bars_since_flip = bars_since(sar_flip)

    gap_pct = (kf - ks) / ks * 100 if ks else 0
    perf_oggi = float((close.iloc[i] / close.iloc[i - 1] - 1) * 100) if len(close) > 1 else 0
    perf_1m = float((close.iloc[i] / close.iloc[max(i - 21, -len(close))] - 1) * 100) if len(close) > 21 else None
    perf_3m = float((close.iloc[i] / close.iloc[max(i - 63, -len(close))] - 1) * 100) if len(close) > 63 else None
    perf_6m = float((close.iloc[i] / close.iloc[max(i - 126, -len(close))] - 1) * 100) if len(close) > 126 else None

    # Zona
    if price > kf > ks:
        zona = "LONG_CONF"
    elif price > kf and price <= ks:
        zona = "LONG_EARLY"
    elif price < ks * 0.98:
        zona = "STOP"
    elif price < ks:
        zona = "USCITA"
    else:
        zona = "NEUTRA"

    # RSI cross (bull/bear) sull'ultima barra
    rsi_cross = 0
    if len(rsi5) > 1 and len(rsi14) > 1:
        prev5, prev14 = rsi5.iloc[i - 1], rsi14.iloc[i - 1]
        if prev5 <= prev14 and rsi5_v > rsi14_v:
            rsi_cross = 1
        elif prev5 >= prev14 and rsi5_v < rsi14_v:
            rsi_cross = -1

    buy3 = (zona == "LONG_CONF" and ao_v > 0 and volr >= 2.0 and baff >= 3
            and er_v >= 0.35 and gap_pct >= 0.3 and sar_bullish)
    buy2 = (zona == "LONG_EARLY" and ao_v > 0 and volr >= 1.5 and baff >= 3 and er_v >= 0.35)
    sell_stop = zona == "STOP"
    sell_exit = zona == "USCITA"

    super_best_buy = (
        sar_bullish and bars_since_flip <= 2
        and ao_v > 0 and ao_improving
        and volr >= 1.5
        and abs(perf_oggi) <= 4.0
    )

    # Score tecnico composito 0-100 (allineamento trend + momentum + forza)
    score = 50.0
    score += 15 if zona == "LONG_CONF" else (7 if zona == "LONG_EARLY" else (-15 if zona == "STOP" else -7 if zona == "USCITA" else 0))
    score += min(max(ao_v, -10), 10) * 1.0
    score += (adx_val - 20) * 0.3
    score += (er_v - 0.3) * 20
    score += 10 if sar_bullish else -10
    score += 5 if rsi_cross == 1 else (-5 if rsi_cross == -1 else 0)
    score = max(0, min(100, score))
    if score >= 75:
        rating = "Forte Buy"
    elif score >= 60:
        rating = "Buy"
    elif score >= 40:
        rating = "Neutro"
    elif score >= 25:
        rating = "Sell"
    else:
        rating = "Forte Sell"

    if buy3:
        segnale = "BUY3"
    elif buy2:
        segnale = "BUY2"
    elif sell_stop:
        segnale = "STOP"
    elif sell_exit:
        segnale = "SELL"
    else:
        segnale = "HOLD"

    # data ultimo flip SAR
    flip_idx = np.where(sar_flip.values)[0]
    sar_since_date = str(df.index[flip_idx[-1]].date()) if len(flip_idx) else None

    # data da cui vige il segnale attuale (ultima variazione di `segnale` — approssimato
    # tornando indietro finché baff resta coerente con lo stato attuale sopra/sotto KAMA)
    segnale_dal = str(df.index[max(len(df) - baff, 0)].date())

    return {
        "prezzo": round(price, 4),
        "kama_fast": round(kf, 4),
        "kama_slow": round(ks, 4),
        "kama_gap_pct": round(gap_pct, 2),
        "er": round(er_v, 3),
        "ao": round(ao_v, 4),
        "ao_improving": ao_improving,
        "rvi": round(rvi_val, 3),
        "rsi5": round(rsi5_v, 1),
        "rsi14": round(rsi14_v, 1),
        "rsi_cross": rsi_cross,
        "adx": round(adx_val, 1),
        "volume_ratio": round(volr, 2),
        "baff": baff,
        "sar": round(sar_v, 4),
        "sar_bullish": sar_bullish,
        "sar_since": sar_since_date,
        "bars_since_flip": int(bars_since_flip),
        "zona": zona,
        "segnale": segnale,
        "score": round(score, 1),
        "rating": rating,
        "segnale_dal": segnale_dal,
        "buy3": buy3,
        "buy2": buy2,
        "super_best_buy": super_best_buy,
        "perf_oggi": round(perf_oggi, 2),
        "perf_1m": round(perf_1m, 2) if perf_1m is not None else None,
        "perf_3m": round(perf_3m, 2) if perf_3m is not None else None,
        "perf_6m": round(perf_6m, 2) if perf_6m is not None else None,
        "ultimo_aggiornamento": str(df.index[-1].date()),
    }, {
        # serie storiche per il grafico (stile scannerv2)
        "date": [str(d.date()) for d in df.index],
        "open": [round(float(v), 4) for v in open_],
        "high": [round(float(v), 4) for v in high],
        "low": [round(float(v), 4) for v in low],
        "close": [round(float(v), 4) for v in close],
        "volume": [int(v) for v in vol.fillna(0)],
        "kama_fast": [None if math.isnan(v) else round(float(v), 4) for v in kama_fast],
        "kama_slow": [None if math.isnan(v) else round(float(v), 4) for v in kama_slow],
        "sar": [round(float(v), 4) for v in sar],
        "sar_trend": [int(v) for v in sar_trend],
        "ao": [round(float(v), 4) for v in ao],
        "rsi14": [round(float(v), 2) for v in rsi14],
    }


# ---------------------------------------------------------------------------
# Generazione pagina regole (dal template EEI_Regole_RAPTOR.html)
# ---------------------------------------------------------------------------

REGOLE_TEMPLATE = (ROOT / "regole_template.html").read_text(encoding="utf-8")


def build_regole_html(nome: str, ticker: str, ind: dict) -> str:
    now = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    html = REGOLE_TEMPLATE
    repl = {
        "{{NOME}}": nome,
        "{{TICKER}}": ticker,
        "{{GENERATO}}": now,
        "{{AGGIORNATO}}": ind["ultimo_aggiornamento"],
        "{{PREZZO}}": f"{ind['prezzo']:.4f}",
        "{{KAMA_FAST}}": f"{ind['kama_fast']:.4f}",
        "{{KAMA_SLOW}}": f"{ind['kama_slow']:.4f}",
        "{{RSI14}}": f"{ind['rsi14']:.1f}",
        "{{RSI5}}": f"{ind['rsi5']:.1f}",
        "{{AO}}": f"{ind['ao']:.4f}",
        "{{ZONA}}": ind["zona"],
        "{{SEGNALE}}": ind["segnale"],
        "{{RATING}}": ind["rating"],
        "{{SCORE}}": f"{ind['score']:.0f}",
    }
    for k, v in repl.items():
        html = html.replace(k, str(v))
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_universe():
    stoxx = json.loads((ROOT / "tickers_stoxx600.json").read_text(encoding="utf-8"))
    sp500 = json.loads((ROOT / "tickers_sp500.json").read_text(encoding="utf-8"))
    universe = []
    for nome, ticker, settore, paese_o_exch, _ in stoxx:
        universe.append({"regione": "EU", "nome": nome, "ticker": ticker,
                          "settore": settore, "paese": paese_o_exch, "exchange": None})
    for nome, ticker, settore, exch, _ in sp500:
        universe.append({"regione": "US", "nome": nome, "ticker": ticker,
                          "settore": settore, "paese": "US", "exchange": exch})
    return universe


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    universe = load_universe()
    print(f"Universo totale: {len(universe)} titoli")

    results = []
    chart_index = {}
    errors = []

    for batch in chunked(universe, BATCH_SIZE):
        tickers = [u["ticker"] for u in batch]
        try:
            data = yf.download(tickers=tickers, period=HISTORY_PERIOD, interval="1d",
                                group_by="ticker", threads=True, progress=False,
                                auto_adjust=True)
        except Exception as e:
            print(f"Errore batch {tickers[:3]}...: {e}", file=sys.stderr)
            errors.extend(tickers)
            time.sleep(SLEEP_BETWEEN_BATCH)
            continue

        for u in batch:
            t = u["ticker"]
            try:
                df = data[t] if len(tickers) > 1 else data
                out = compute_indicators(df)
            except Exception as e:
                print(f"Errore indicatori {t}: {e}", file=sys.stderr)
                errors.append(t)
                continue
            if out is None:
                errors.append(t)
                continue
            summary, chart = out

            tv = tv_symbol(t, u["exchange"])
            row = {**u, "tv_symbol": tv, **summary}
            results.append(row)

            chart_file = f"{t.replace('.', '_').replace('-', '_')}.json"
            (CHARTS_DIR / chart_file).write_text(json.dumps(chart), encoding="utf-8")
            chart_index[t] = chart_file

            regole_file = f"{t.replace('.', '_').replace('-', '_')}_Regole.html"
            (REGOLE_DIR / regole_file).write_text(
                build_regole_html(u["nome"], t, summary), encoding="utf-8")

        time.sleep(SLEEP_BETWEEN_BATCH)

    (DATA_DIR / "charts" / "index.json").write_text(json.dumps(chart_index), encoding="utf-8")
    (ROOT / "azionario.json").write_text(
        json.dumps({
            "generato": datetime.datetime.now().isoformat(),
            "totale": len(results),
            "errori": errors,
            "titoli": results,
        }, ensure_ascii=False), encoding="utf-8")

    print(f"Completato: {len(results)} titoli ok, {len(errors)} errori")
    if errors:
        print("Errori:", errors[:30], "..." if len(errors) > 30 else "")


if __name__ == "__main__":
    main()
