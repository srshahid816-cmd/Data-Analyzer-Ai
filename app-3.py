"""
Crypto Multi-Factor Market Scanner - v1 (Streamlit Cloud build)

Same backend logic as the Flask version (indicators, structure, regime,
signal engine, universe discovery) - only the UI layer changed, because
Streamlit Cloud runs Streamlit apps, not raw Flask/WSGI apps.

No fabricated data: every field traces to Binance's public REST API, or
shows DATA UNAVAILABLE / NOT_ENOUGH_DATA.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import requests
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # api.binance.com blocks requests from US-based cloud IPs (Streamlit Cloud,
    # Render, Railway all run there) with HTTP 451. data-api.binance.vision is
    # Binance's own official public market-data-only endpoint - same data,
    # not geo-blocked. Docs: developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only
    "binance_base_url": "https://data-api.binance.vision",
    "request_timeout_seconds": 10,
    "max_retries": 3,
    "backoff_base_seconds": 1.5,

    "quote_asset": "USDT",
    "min_24h_quote_volume_usdt": 5_000_000,
    "min_price_usdt": 0.0000001,
    "exclude_stablecoins": True,
    "stablecoin_symbols": {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT", "BUSDUSDT", "USDPUSDT"},
    "exclude_leveraged_tokens": True,
    "leveraged_token_suffixes": ("UP", "DOWN", "BULL", "BEAR"),
    "top_n_gainers": 10,
    "top_n_losers": 10,

    "primary_timeframe": "1h",
    "regime_timeframe_high": "4h",
    "klines_limit": 200,

    "btc_correlated_watchlist": ["ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT",
                                  "SUIUSDT", "LTCUSDT", "NEARUSDT", "SNXUSDT"],

    "coingecko_base_url": "https://api.coingecko.com/api/v3",
    "liquidity_tolerance_pct": 0.15,   # how close two swing prices must be to count as "equal"
    "exhaustion_lookback": 10,          # candles used for parabolic-move / climax checks

    "signal_weights": {
        "btc_regime_alignment": 20,
        "structure_state": 25,
        "indicator_confluence": 20,
        "volume_confirmation": 15,
        "relative_strength_vs_btc": 10,
    },
    "min_score_buy_sell": 65,
    "min_data_quality_for_trade": 60,
    "scan_cache_ttl_seconds": 55,
}

# ============================================================
# DATA LAYER
# ============================================================

class MarketDataError(Exception):
    pass


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


class BinanceMarketData:
    def __init__(self):
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None):
        url = f"{CONFIG['binance_base_url']}{path}"
        last_err = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                resp = self.session.get(url, params=params, timeout=CONFIG["request_timeout_seconds"])
                if resp.status_code in (429, 418):
                    time.sleep(CONFIG["backoff_base_seconds"] * (2 ** attempt))
                    last_err = MarketDataError(f"Rate limited ({resp.status_code}) on {path}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                time.sleep(CONFIG["backoff_base_seconds"] * attempt)
        raise MarketDataError(f"Failed to fetch {path}: {last_err}")

    def get_exchange_info(self):
        return self._get("/api/v3/exchangeInfo")

    def get_ticker_24hr(self):
        return self._get("/api/v3/ticker/24hr")

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        raw = self._get("/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
        if not raw:
            raise MarketDataError(f"Empty klines for {symbol} {interval}")
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        for col in ["open", "high", "low", "close", "volume",
                    "taker_buy_base_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        return df


class CoinGeckoError(Exception):
    pass


class CoinGeckoData:
    """Public CoinGecko API - no key required for basic access (rate-limited).
    Used only for tokenomics/supply data, which Binance doesn't provide."""

    def __init__(self):
        self.session = requests.Session()
        self._id_cache: dict[str, Optional[str]] = {}

    def _get(self, path: str, params: dict | None = None):
        url = f"{CONFIG['coingecko_base_url']}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                raise CoinGeckoError("Rate limited by CoinGecko free tier")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise CoinGeckoError(str(e))

    def find_coin_id(self, base_symbol: str) -> Optional[str]:
        """base_symbol e.g. 'BTC', 'NEAR'. Returns CoinGecko's internal id, or
        None if no confident match - NEVER guesses an id from memory."""
        key = base_symbol.upper()
        if key in self._id_cache:
            return self._id_cache[key]
        try:
            result = self._get("/search", params={"query": base_symbol})
            coins = [c for c in result.get("coins", []) if c.get("symbol", "").upper() == key]
            if not coins:
                self._id_cache[key] = None
                return None
            # Multiple projects can share a ticker - pick the one CoinGecko itself
            # ranks highest (lowest market_cap_rank; None ranks last), not a guess.
            coins.sort(key=lambda c: (c.get("market_cap_rank") is None, c.get("market_cap_rank") or 0))
            best = coins[0]["id"]
            self._id_cache[key] = best
            return best
        except CoinGeckoError:
            return None

    def get_tokenomics(self, base_symbol: str) -> dict:
        coin_id = self.find_coin_id(base_symbol)
        if coin_id is None:
            return {"status": "DATA UNAVAILABLE", "reason": "No confident CoinGecko match for this symbol"}
        try:
            data = self._get(f"/coins/{coin_id}", params={
                "localization": "false", "tickers": "false", "market_data": "true",
                "community_data": "false", "developer_data": "false",
            })
            md = data.get("market_data", {})
            circ = md.get("circulating_supply")
            total = md.get("total_supply")
            maxs = md.get("max_supply")
            mcap = (md.get("market_cap") or {}).get("usd")
            fdv = (md.get("fully_diluted_valuation") or {}).get("usd")

            supply_flag = "UNKNOWN"
            if circ and total:
                pct_circulating = circ / total * 100
                supply_flag = "SUPPLY_MOSTLY_UNLOCKED" if pct_circulating > 90 else "SUPPLY_STILL_UNLOCKING"

            return {
                "status": "OK",
                "coingecko_id": coin_id,
                "circulating_supply": circ,
                "total_supply": total,
                "max_supply": maxs,
                "market_cap_usd": mcap,
                "fdv_usd": fdv,
                "supply_flag": supply_flag,
                "note": "Unlock schedule / vesting dates need a dedicated tokenomics "
                        "provider - not available from this free source, not guessed.",
            }
        except CoinGeckoError as e:
            return {"status": "DATA UNAVAILABLE", "reason": str(e)}


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100.0)


def macd(closes: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(closes, fast) - ema(closes, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(highs, lows, closes, period=14) -> pd.Series:
    prev_close = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def last_valid(series: pd.Series):
    if series is None or len(series) == 0 or pd.isna(series.iloc[-1]):
        return None, False
    return float(series.iloc[-1]), True


# ============================================================
# STRUCTURE
# ============================================================

class StructureState(Enum):
    BULLISH_STRUCTURE = "BULLISH_STRUCTURE"
    BEARISH_STRUCTURE = "BEARISH_STRUCTURE"
    RANGE = "RANGE"
    TRANSITION = "TRANSITION"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str
    label: Optional[str] = None


def find_swing_points(highs: pd.Series, lows: pd.Series, lookback: int = 2) -> List[SwingPoint]:
    points: List[SwingPoint] = []
    n = len(highs)
    if n < (2 * lookback + 1):
        return points
    for i in range(lookback, n - lookback):
        window_h = highs.iloc[i - lookback: i + lookback + 1]
        window_l = lows.iloc[i - lookback: i + lookback + 1]
        prior_h = highs.iloc[i - lookback:i]
        prior_h_max = prior_h.max() if len(prior_h) else float("-inf")
        prior_l = lows.iloc[i - lookback:i]
        prior_l_min = prior_l.min() if len(prior_l) else float("inf")
        if highs.iloc[i] == window_h.max() and highs.iloc[i] != prior_h_max:
            points.append(SwingPoint(i, float(highs.iloc[i]), "HIGH"))
        if lows.iloc[i] == window_l.min() and lows.iloc[i] != prior_l_min:
            points.append(SwingPoint(i, float(lows.iloc[i]), "LOW"))
    points.sort(key=lambda p: p.index)
    return points


def label_swings(points: List[SwingPoint]) -> List[SwingPoint]:
    last_high = last_low = None
    for p in points:
        if p.kind == "HIGH":
            if last_high is not None:
                p.label = "HH" if p.price > last_high.price else "LH"
            last_high = p
        else:
            if last_low is not None:
                p.label = "HL" if p.price > last_low.price else "LL"
            last_low = p
    return points


def detect_structure(highs: pd.Series, lows: pd.Series, lookback: int = 2) -> dict:
    raw_points = find_swing_points(highs, lows, lookback)
    if len(raw_points) < 4:
        return {"state": StructureState.UNCERTAIN, "reason": "NOT_ENOUGH_DATA"}
    points = label_swings(raw_points)
    labels = [p.label for p in points if p.label is not None]
    recent = labels[-4:] if len(labels) >= 4 else labels
    bull = sum(1 for l in recent if l in ("HH", "HL"))
    bear = sum(1 for l in recent if l in ("LH", "LL"))
    if bull >= 3:
        state = StructureState.BULLISH_STRUCTURE
    elif bear >= 3:
        state = StructureState.BEARISH_STRUCTURE
    elif bull == bear:
        state = StructureState.RANGE
    else:
        state = StructureState.TRANSITION
    return {"state": state, "reason": f"last swings: {recent}", "points": points}


def nearest_support_resistance(highs, lows, current_price, lookback=2):
    points = label_swings(find_swing_points(highs, lows, lookback))
    highs_above = [p.price for p in points if p.kind == "HIGH" and p.price > current_price]
    lows_below = [p.price for p in points if p.kind == "LOW" and p.price < current_price]
    return (max(lows_below) if lows_below else None,
            min(highs_above) if highs_above else None)


# ============================================================
# LIQUIDITY ANALYSIS - equal highs/lows, sweeps vs breakouts
# ============================================================

def detect_liquidity_events(klines: pd.DataFrame, lookback: int = 2, tol_pct: float = 0.15) -> list:
    """Finds clusters of equal highs/lows (liquidity pools) and classifies what
    happened when price later traded through them: a SWEEP_AND_RECLAIM (wick
    through, close back inside - stop-run) or a TRUE_BREAKOUT (closed beyond
    and stayed). Returns a list of dicts, most recent first."""
    highs, lows, closes = klines["high"], klines["low"], klines["close"]
    points = label_swings(find_swing_points(highs, lows, lookback))
    events = []

    for kind, arr in (("HIGH", [p for p in points if p.kind == "HIGH"]),
                       ("LOW", [p for p in points if p.kind == "LOW"])):
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            if a.price == 0:
                continue
            if abs(a.price - b.price) / a.price * 100 <= tol_pct:
                # Equal high/low pool found between index a and b. Check what
                # happened after the second point formed.
                level = max(a.price, b.price) if kind == "HIGH" else min(a.price, b.price)
                after = klines.iloc[b.index + 1:]
                if len(after) == 0:
                    continue
                if kind == "HIGH":
                    swept = (after["high"] > level).any()
                    closed_above = (after["close"] > level).any()
                    reaction = "SWEEP_AND_RECLAIM" if (swept and not closed_above) else \
                               ("TRUE_BREAKOUT" if closed_above else "UNTESTED")
                else:
                    swept = (after["low"] < level).any()
                    closed_below = (after["close"] < level).any()
                    reaction = "SWEEP_AND_RECLAIM" if (swept and not closed_below) else \
                               ("TRUE_BREAKOUT" if closed_below else "UNTESTED")
                events.append({
                    "type": f"EQUAL_{kind}S",
                    "level": round(level, 8),
                    "formed_at_index": b.index,
                    "reaction": reaction,
                })
    events.sort(key=lambda e: e["formed_at_index"], reverse=True)
    return events[:5]


# ============================================================
# SPOT BUY/SELL FLOW - from Binance's taker buy volume field
# ============================================================

def spot_flow(klines: pd.DataFrame, lookback: int = 20) -> dict:
    if len(klines) < lookback + 1 or "taker_buy_base_volume" not in klines.columns:
        return {"status": "DATA UNAVAILABLE"}

    recent = klines.iloc[-lookback:]
    total_vol = recent["volume"].sum()
    if total_vol == 0:
        return {"status": "DATA UNAVAILABLE"}

    taker_buy_vol = recent["taker_buy_base_volume"].sum()
    taker_sell_vol = total_vol - taker_buy_vol
    buy_ratio = taker_buy_vol / total_vol

    price_change_pct = (klines["close"].iloc[-1] / klines["close"].iloc[-lookback] - 1) * 100

    if price_change_pct > 0.5 and buy_ratio > 0.55:
        interpretation = "Price up + spot buying dominant -> stronger confirmation of the move"
    elif price_change_pct > 0.5 and buy_ratio <= 0.52:
        interpretation = "Price up but spot buying is NOT dominant -> possible fragile / leverage-driven move"
    elif price_change_pct < -0.5 and buy_ratio < 0.45:
        interpretation = "Price down + spot selling dominant -> confirms the down move"
    elif price_change_pct < -0.5 and buy_ratio >= 0.48:
        interpretation = "Price down but spot selling is fading -> possible selling exhaustion (not proof)"
    else:
        interpretation = "No strong spot-flow signal either way"

    return {
        "status": "OK",
        "spot_buy_pressure_pct": round(buy_ratio * 100, 1),
        "spot_sell_pressure_pct": round((1 - buy_ratio) * 100, 1),
        "price_change_pct": round(price_change_pct, 2),
        "interpretation": interpretation,
    }


# ============================================================
# GAINER EXHAUSTION / LOSER REVERSAL ANALYSIS
# (same function, direction-aware - spec sections 7 & 8)
# ============================================================

def exhaustion_or_reversal_analysis(klines: pd.DataFrame, direction: str) -> dict:
    """direction: 'up' (check for gainer exhaustion) or 'down' (check for loser
    reversal/selling-exhaustion). Every signal here is derived from the same
    candles already fetched - no invented data, no certainty claims."""
    n = CONFIG["exhaustion_lookback"]
    if len(klines) < n + 30:
        return {"classification": "NO_CLEAR_SIGNAL", "evidence": ["NOT_ENOUGH_DATA"]}

    recent = klines.iloc[-n:]
    closes = klines["close"]
    evidence = []
    score = 0  # + supports the exhaustion/reversal call, - argues against it

    move_pct = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100
    if direction == "up":
        if move_pct > 15:
            evidence.append(f"Parabolic expansion: +{move_pct:.1f}% in last {n} candles")
            score += 1
    else:
        if move_pct < -15:
            evidence.append(f"Parabolic decline: {move_pct:.1f}% in last {n} candles")
            score += 1

    avg_vol = klines["volume"].iloc[-(n + 20):-n].mean()
    peak_vol = recent["volume"].max()
    if avg_vol > 0 and peak_vol / avg_vol > 2.5:
        evidence.append(f"Volume climax: peak volume {peak_vol/avg_vol:.1f}x the prior average")
        score += 1

    rng = (recent["high"] - recent["low"]).replace(0, np.nan)
    if direction == "up":
        upper_wick_ratio = ((recent["high"] - recent[["open", "close"]].max(axis=1)) / rng).mean()
        if pd.notna(upper_wick_ratio) and upper_wick_ratio > 0.35:
            evidence.append(f"Long upper wicks forming (avg {upper_wick_ratio*100:.0f}% of candle range) - buyers failing to hold highs")
            score += 1
    else:
        lower_wick_ratio = ((recent[["open", "close"]].min(axis=1) - recent["low"]) / rng).mean()
        if pd.notna(lower_wick_ratio) and lower_wick_ratio > 0.35:
            evidence.append(f"Long lower wicks forming (avg {lower_wick_ratio*100:.0f}% of candle range) - sellers failing to hold lows")
            score += 1

    rsi_series = rsi(closes)
    struct = detect_structure(klines["high"], klines["low"])
    points = struct.get("points", [])
    if direction == "up":
        highs = [p for p in points if p.kind == "HIGH"]
        if len(highs) >= 2 and highs[-1].price > highs[-2].price:
            rsi_at_a, ok_a = last_valid(rsi_series.iloc[:highs[-2].index + 1])
            rsi_at_b, ok_b = last_valid(rsi_series.iloc[:highs[-1].index + 1])
            if ok_a and ok_b and rsi_at_b < rsi_at_a:
                evidence.append("Bearish RSI divergence: price made a higher high, RSI made a lower high")
                score += 1
        if len(highs) >= 2 and highs[-1].label == "LH":
            evidence.append("Lower High formed after the prior Higher High - momentum stalling")
            score += 1
    else:
        lows = [p for p in points if p.kind == "LOW"]
        if len(lows) >= 2 and lows[-1].price < lows[-2].price:
            rsi_at_a, ok_a = last_valid(rsi_series.iloc[:lows[-2].index + 1])
            rsi_at_b, ok_b = last_valid(rsi_series.iloc[:lows[-1].index + 1])
            if ok_a and ok_b and rsi_at_b > rsi_at_a:
                evidence.append("Bullish RSI divergence: price made a lower low, RSI made a higher low")
                score += 1
        if len(lows) >= 2 and lows[-1].label == "HL":
            evidence.append("Higher Low formed after the prior Lower Low - selling pressure stalling")
            score += 1

    if not evidence:
        return {"classification": "NO_CLEAR_SIGNAL", "evidence": ["No exhaustion/reversal evidence found in current data"]}

    if direction == "up":
        classification = "BEARISH_CONFIRMATION" if score >= 3 else "REVERSAL_WATCH" if score == 2 else "EXHAUSTION_WARNING"
    else:
        classification = "BULLISH_CONFIRMATION" if score >= 3 else "REVERSAL_WATCH" if score == 2 else "SELLING_EXHAUSTION"

    return {"classification": classification, "evidence": evidence, "score": score}


# ============================================================
# MARKET REGIME
# ============================================================

class Regime(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    UNCERTAIN = "UNCERTAIN"


def classify_regime(klines_1h: pd.DataFrame, klines_4h: pd.DataFrame) -> dict:
    if klines_1h is None or klines_4h is None or len(klines_1h) < 50 or len(klines_4h) < 50:
        return {"regime": Regime.UNCERTAIN, "reason": "NOT_ENOUGH_DATA"}

    closes_1h = klines_1h["close"]
    ema50_1h, ok50 = last_valid(ema(closes_1h, 50))
    ema200_1h, ok200 = last_valid(ema(closes_1h, 200))
    price, _ = last_valid(closes_1h)
    if not (ok50 and price is not None):
        return {"regime": Regime.UNCERTAIN, "reason": "NOT_ENOUGH_DATA for EMA"}

    struct_4h = detect_structure(klines_4h["high"], klines_4h["low"])
    struct_1h = detect_structure(klines_1h["high"], klines_1h["low"])

    score = 0
    reasons = []
    if price > ema50_1h:
        score += 1; reasons.append("price>EMA50")
    else:
        score -= 1; reasons.append("price<EMA50")
    if ok200:
        if price > ema200_1h:
            score += 1; reasons.append("price>EMA200")
        else:
            score -= 1; reasons.append("price<EMA200")
    if struct_4h["state"] == StructureState.BULLISH_STRUCTURE:
        score += 2; reasons.append("4h bullish")
    elif struct_4h["state"] == StructureState.BEARISH_STRUCTURE:
        score -= 2; reasons.append("4h bearish")
    if struct_1h["state"] == StructureState.BULLISH_STRUCTURE:
        score += 1; reasons.append("1h bullish")
    elif struct_1h["state"] == StructureState.BEARISH_STRUCTURE:
        score -= 1; reasons.append("1h bearish")

    if score >= 4:
        regime = Regime.STRONG_BULLISH
    elif score >= 2:
        regime = Regime.BULLISH
    elif score <= -4:
        regime = Regime.STRONG_BEARISH
    elif score <= -2:
        regime = Regime.BEARISH
    else:
        regime = Regime.NEUTRAL
    return {"regime": regime, "reason": "; ".join(reasons)}


def relative_strength_vs_btc(sym_klines, btc_klines, lookback=24):
    if sym_klines is None or btc_klines is None or len(sym_klines) < lookback + 1 or len(btc_klines) < lookback + 1:
        return None
    sym_chg = (sym_klines["close"].iloc[-1] / sym_klines["close"].iloc[-lookback] - 1) * 100
    btc_chg = (btc_klines["close"].iloc[-1] / btc_klines["close"].iloc[-lookback] - 1) * 100
    return float(sym_chg - btc_chg)


def backtest_regime(closes_1h: pd.Series, struct_1h_state: "StructureState") -> "Regime":
    """Simplified regime call used ONLY inside the backtest loop - 1h data only
    (no 4h confirmation, since resampling 1h into synthetic 4h bars for every
    backtest step adds complexity without changing the honesty of the result).
    This is disclosed to the user as an approximation of the live regime logic,
    not claimed to be identical."""
    ema50, ok50 = last_valid(ema(closes_1h, 50))
    ema200, ok200 = last_valid(ema(closes_1h, 200))
    price, price_ok = last_valid(closes_1h)
    if not (ok50 and price_ok):
        return Regime.UNCERTAIN
    score = 0
    score += 1 if price > ema50 else -1
    if ok200:
        score += 1 if price > ema200 else -1
    if struct_1h_state == StructureState.BULLISH_STRUCTURE:
        score += 2
    elif struct_1h_state == StructureState.BEARISH_STRUCTURE:
        score -= 2
    if score >= 3:
        return Regime.STRONG_BULLISH
    elif score >= 1:
        return Regime.BULLISH
    elif score <= -3:
        return Regime.STRONG_BEARISH
    elif score <= -1:
        return Regime.BEARISH
    return Regime.NEUTRAL


# ============================================================
# SIGNAL ENGINE
# ============================================================

class Decision(Enum):
    BUY_SETUP = "BUY SETUP"
    SELL_SETUP = "SELL SETUP"
    HOLD_WAIT = "HOLD / WAIT"
    NO_TRADE = "NO TRADE"


@dataclass
class FactorScore:
    name: str
    value: Optional[str]
    score: float
    data_quality: str


def score_symbol(symbol: str, klines: pd.DataFrame, btc_regime: Regime, rel_strength: Optional[float]) -> dict:
    struct = detect_structure(klines["high"], klines["low"])
    state = struct["state"]
    factors = []

    bullish_regimes = {Regime.STRONG_BULLISH, Regime.BULLISH}
    bearish_regimes = {Regime.STRONG_BEARISH, Regime.BEARISH}
    if btc_regime == Regime.UNCERTAIN:
        factors.append(FactorScore("btc_regime_alignment", btc_regime.value, 0.0, "UNAVAILABLE"))
    else:
        if btc_regime in bullish_regimes and state == StructureState.BULLISH_STRUCTURE:
            s = 1.0 if btc_regime == Regime.STRONG_BULLISH else 0.6
        elif btc_regime in bearish_regimes and state == StructureState.BEARISH_STRUCTURE:
            s = -1.0 if btc_regime == Regime.STRONG_BEARISH else -0.6
        else:
            s = 0.0
        factors.append(FactorScore("btc_regime_alignment", f"{btc_regime.value} vs {state.value}", s, "GOOD"))

    struct_map = {StructureState.BULLISH_STRUCTURE: 1.0, StructureState.BEARISH_STRUCTURE: -1.0,
                  StructureState.RANGE: 0.0, StructureState.TRANSITION: 0.0}
    if state == StructureState.UNCERTAIN:
        factors.append(FactorScore("structure_state", state.value, 0.0, "PARTIAL"))
    else:
        factors.append(FactorScore("structure_state", state.value, struct_map[state], "GOOD"))

    closes = klines["close"]
    if len(closes) < 30:
        factors.append(FactorScore("indicator_confluence", None, 0.0, "UNAVAILABLE"))
    else:
        rsi_val, rsi_ok = last_valid(rsi(closes))
        macd_line, signal_line = macd(closes)
        macd_val, macd_ok = last_valid(macd_line)
        sig_val, sig_ok = last_valid(signal_line)
        ema20_val, e20_ok = last_valid(ema(closes, 20))
        ema50_val, e50_ok = last_valid(ema(closes, 50))
        price, price_ok = last_valid(closes)
        if not all([rsi_ok, macd_ok, sig_ok, e20_ok, e50_ok, price_ok]):
            factors.append(FactorScore("indicator_confluence", None, 0.0, "PARTIAL"))
        else:
            votes, total = 0, 0
            total += 1
            votes += 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
            total += 1
            votes += 1 if macd_val > sig_val else -1
            total += 1
            if price > ema20_val > ema50_val:
                votes += 1
            elif price < ema20_val < ema50_val:
                votes -= 1
            factors.append(FactorScore("indicator_confluence", f"RSI={rsi_val:.1f}", votes / total, "GOOD"))

    vol = klines["volume"]
    if len(vol) < 21:
        factors.append(FactorScore("volume_confirmation", None, 0.0, "UNAVAILABLE"))
    else:
        recent, avg = vol.iloc[-1], vol.iloc[-21:-1].mean()
        if avg == 0 or pd.isna(avg):
            factors.append(FactorScore("volume_confirmation", None, 0.0, "PARTIAL"))
        else:
            ratio = recent / avg
            price_up = klines["close"].iloc[-1] > klines["close"].iloc[-2]
            s = (0.7 if price_up else -0.7) if ratio > 1.3 else (-0.2 if ratio < 0.6 else 0.0)
            factors.append(FactorScore("volume_confirmation", f"vol/avg={ratio:.2f}", s, "GOOD"))

    if rel_strength is None:
        factors.append(FactorScore("relative_strength_vs_btc", None, 0.0, "UNAVAILABLE"))
    else:
        s = max(-1.0, min(1.0, rel_strength / 5.0))
        factors.append(FactorScore("relative_strength_vs_btc", f"{rel_strength:+.2f}% vs BTC", s, "GOOD"))

    unavailable = sum(1 for f in factors if f.data_quality == "UNAVAILABLE")
    partial = sum(1 for f in factors if f.data_quality == "PARTIAL")
    quality_ratio = 1 - (unavailable + 0.5 * partial) / max(len(factors), 1)

    weights = CONFIG["signal_weights"]
    total_weight = sum(weights.values()) or 1
    weighted = sum(f.score * weights.get(f.name, 0) for f in factors) / total_weight
    confidence = abs(weighted) * 100 * quality_ratio
    direction = "BUY" if weighted > 0 else "SELL" if weighted < 0 else "NONE"
    data_quality_status = "GOOD" if quality_ratio >= 0.8 else "PARTIAL" if quality_ratio >= 0.5 else "STALE"

    decision = Decision.HOLD_WAIT
    entry_zone = invalidation = targets = risk_reward = None
    meets_confidence_bar = confidence >= CONFIG["min_score_buy_sell"]

    if quality_ratio * 100 < CONFIG["min_data_quality_for_trade"]:
        decision = Decision.NO_TRADE
    elif direction != "NONE":
        # Levels are computed for whichever direction the score leans, REGARDLESS
        # of whether it clears the confidence bar - the official BUY/SELL SETUP
        # label still requires the bar, but the levels themselves are always shown
        # (labelled "indicative" below the bar) so the user always sees them.
        price, price_ok = last_valid(closes)
        atr_val, atr_ok = last_valid(atr(klines["high"], klines["low"], closes))
        support, resistance = nearest_support_resistance(klines["high"], klines["low"], price or 0)
        if price_ok and atr_ok:
            if direction == "BUY":
                ref = support if support else price - atr_val
                if price - ref > 2.5 * atr_val:
                    entry_zone = "NO ENTRY — WAIT FOR RETEST (price extended beyond support)"
                else:
                    entry_zone = f"{ref:.6g} - {price:.6g}"
                    invalidation = ref - atr_val
                    targets = [round(price + atr_val * m, 8) for m in (1.5, 2.5, 4.0)]
                    risk = price - invalidation
                    risk_reward = round((targets[0] - price) / risk, 2) if risk > 0 else None
                    if meets_confidence_bar:
                        decision = Decision.BUY_SETUP
            else:
                ref = resistance if resistance else price + atr_val
                if ref - price > 2.5 * atr_val:
                    entry_zone = "NO ENTRY — WAIT FOR RETEST (price extended beyond resistance)"
                else:
                    entry_zone = f"{price:.6g} - {ref:.6g}"
                    invalidation = ref + atr_val
                    targets = [round(price - atr_val * m, 8) for m in (1.5, 2.5, 4.0)]
                    risk = invalidation - price
                    risk_reward = round((price - targets[0]) / risk, 2) if risk > 0 else None
                    if meets_confidence_bar:
                        decision = Decision.SELL_SETUP
        else:
            decision = Decision.NO_TRADE

    return {
        "symbol": symbol,
        "decision": decision.value,
        "direction": direction,
        "meets_confidence_bar": meets_confidence_bar,
        "confidence": round(confidence, 1),
        "historical_win_probability": None,
        "historical_note": "See the backtest panel for this symbol's real historical win rate.",
        "entry_zone": entry_zone,
        "invalidation": round(invalidation, 8) if invalidation else None,
        "targets": targets,
        "risk_reward": risk_reward,
        "data_quality": data_quality_status,
    }


# ============================================================
# UNIVERSE + SCAN
# ============================================================

def get_eligible_universe(client: BinanceMarketData) -> List[dict]:
    info = client.get_exchange_info()
    tradable = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING" and s["quoteAsset"] == CONFIG["quote_asset"]}
    tickers = client.get_ticker_24hr()
    eligible = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in tradable:
            continue
        if CONFIG["exclude_stablecoins"] and symbol in CONFIG["stablecoin_symbols"]:
            continue
        if CONFIG["exclude_leveraged_tokens"] and any(symbol.replace(CONFIG["quote_asset"], "").endswith(s) for s in CONFIG["leveraged_token_suffixes"]):
            continue
        try:
            qv, price, pct = float(t["quoteVolume"]), float(t["lastPrice"]), float(t["priceChangePercent"])
        except (KeyError, ValueError):
            continue
        if qv < CONFIG["min_24h_quote_volume_usdt"] or price < CONFIG["min_price_usdt"]:
            continue
        eligible.append({"symbol": symbol, "last_price": price, "quote_volume_24h": qv, "pct_change_24h": pct})
    return eligible


def _safe_klines(client, symbol, interval, limit):
    try:
        return client.get_klines(symbol, interval, limit)
    except MarketDataError as e:
        logger.warning("Klines fetch failed for %s %s: %s", symbol, interval, e)
        return None


def analyze_symbol(client, symbol, btc_klines_1h, btc_regime):
    klines = _safe_klines(client, symbol, CONFIG["primary_timeframe"], CONFIG["klines_limit"])
    if klines is None or len(klines) < 30:
        return {"symbol": symbol, "status": "DATA UNAVAILABLE"}
    rel = relative_strength_vs_btc(klines, btc_klines_1h) if btc_klines_1h is not None else None
    result = score_symbol(symbol, klines, btc_regime, rel)
    result["status"] = "OK"
    result["price"] = float(klines["close"].iloc[-1])

    # Liquidity + spot flow reuse the SAME already-fetched candles - no extra
    # network calls, so these are safe to compute for every scanned symbol.
    result["liquidity_events"] = detect_liquidity_events(klines)
    result["spot_flow"] = spot_flow(klines)

    look = CONFIG["exhaustion_lookback"]
    if len(klines) > look:
        recent_move = klines["close"].iloc[-1] / klines["close"].iloc[-look] - 1
        direction = "up" if recent_move >= 0 else "down"
    else:
        direction = "up"
    result["exhaustion_analysis"] = exhaustion_or_reversal_analysis(klines, direction)
    return result


@st.cache_data(ttl=CONFIG["scan_cache_ttl_seconds"], show_spinner=False)
def run_scan_cached() -> dict:
    client = BinanceMarketData()
    limit = CONFIG["klines_limit"]
    btc_1h = _safe_klines(client, "BTCUSDT", "1h", limit)
    btc_4h = _safe_klines(client, "BTCUSDT", CONFIG["regime_timeframe_high"], limit)

    if btc_1h is not None and btc_4h is not None:
        regime_result = classify_regime(btc_1h, btc_4h)
    else:
        regime_result = {"regime": Regime.UNCERTAIN, "reason": "DATA UNAVAILABLE"}
    btc_regime = regime_result["regime"]

    try:
        eligible = get_eligible_universe(client)
    except MarketDataError as e:
        logger.error("Universe discovery failed: %s", e)
        eligible = []

    gainers = sorted(eligible, key=lambda x: x["pct_change_24h"], reverse=True)[:CONFIG["top_n_gainers"]]
    losers = sorted(eligible, key=lambda x: x["pct_change_24h"])[:CONFIG["top_n_losers"]]
    eligible_by_symbol = {e["symbol"]: e for e in eligible}

    def with_pct(items):
        out = []
        for it in items:
            r = analyze_symbol(client, it["symbol"], btc_1h, btc_regime)
            r["pct_change_24h"] = it["pct_change_24h"]
            out.append(r)
        return out

    # BTC-correlated watchlist: fixed large-cap coins that tend to move with
    # BTC. Reuses the same analyze_symbol pipeline; pct_change comes from the
    # universe scan when available, else DATA UNAVAILABLE (never guessed).
    correlated = []
    for sym in CONFIG["btc_correlated_watchlist"]:
        r = analyze_symbol(client, sym, btc_1h, btc_regime)
        r["pct_change_24h"] = eligible_by_symbol.get(sym, {}).get("pct_change_24h")
        correlated.append(r)

    return {
        "btc_regime": btc_regime.value,
        "btc_regime_reason": regime_result.get("reason"),
        "gainers": with_pct(gainers),
        "losers": with_pct(losers),
        "correlated": correlated,
        "universe_size": len(eligible),
    }


@st.cache_data(ttl=120, show_spinner=False)
def get_symbol_deep_dive(symbol: str) -> dict:
    """On-demand, per-symbol detail: multi-timeframe structure + tokenomics.
    Only called for the ONE symbol the user selects - keeps CoinGecko calls
    (rate-limited free tier) and extra klines calls low."""
    client = BinanceMarketData()
    timeframes = ["15m", "1h", "4h", "1d"]
    structure_by_tf = {}
    for tf in timeframes:
        k = _safe_klines(client, symbol, tf, CONFIG["klines_limit"])
        if k is None or len(k) < 10:
            structure_by_tf[tf] = {"state": "DATA UNAVAILABLE"}
        else:
            s = detect_structure(k["high"], k["low"])
            structure_by_tf[tf] = {"state": s["state"].value, "reason": s["reason"]}

    base_symbol = symbol.replace(CONFIG["quote_asset"], "")
    try:
        cg = CoinGeckoData()
        tokenomics = cg.get_tokenomics(base_symbol)
    except Exception as e:
        tokenomics = {"status": "DATA UNAVAILABLE", "reason": str(e)}

    return {"structure_by_timeframe": structure_by_tf, "tokenomics": tokenomics}


# ============================================================
# BACKTEST ENGINE - real historical win rate, computed on demand
# per symbol. Walks forward through real past candles using the
# SAME score_symbol() logic live scanning uses - no separate,
# possibly-inconsistent scoring path. No look-ahead: at each step
# i, only klines[:i+1] is visible to the scorer.
#
# Known, disclosed simplifications (shown to the user, not hidden):
#  - Regime uses 1h data only during backtest (see backtest_regime)
#  - No fees/slippage modeled
#  - In-sample replay, not a formal train/test walk-forward split
#  - Trades are taken as non-overlapping (skip ahead after each
#    trade resolves) to avoid trivially correlated duplicate counts
# ============================================================

BACKTEST_KLINES_LIMIT = 500       # ~20 days of 1h candles
BACKTEST_MIN_HISTORY = 210        # need >=200 for EMA200 to be valid
BACKTEST_MAX_HOLDING_BARS = 48    # max 48h to hit TP1 or SL before "timeout"


def _run_backtest_core(symbol_klines: pd.DataFrame, btc_klines: pd.DataFrame) -> dict:
    n = len(symbol_klines)
    if n < BACKTEST_MIN_HISTORY + BACKTEST_MAX_HOLDING_BARS + 10:
        return {"status": "NOT_ENOUGH_DATA", "trades": []}

    closes = symbol_klines["close"]
    highs = symbol_klines["high"]
    lows = symbol_klines["low"]
    btc_closes = btc_klines["close"] if btc_klines is not None else None

    trades = []
    i = BACKTEST_MIN_HISTORY
    upper_bound = n - BACKTEST_MAX_HOLDING_BARS - 1

    while i <= upper_bound:
        sym_slice = symbol_klines.iloc[:i + 1]

        if btc_closes is not None and i < len(btc_klines):
            btc_slice = btc_klines.iloc[:i + 1]
            btc_struct = detect_structure(btc_slice["high"], btc_slice["low"])
            regime = backtest_regime(btc_slice["close"], btc_struct["state"])
            rel_strength = relative_strength_vs_btc(sym_slice, btc_slice)
        else:
            regime = Regime.UNCERTAIN
            rel_strength = None

        result = score_symbol(f"BT_{i}", sym_slice, regime, rel_strength)

        is_actionable = (
            result["direction"] in ("BUY", "SELL")
            and result["meets_confidence_bar"]
            and result["entry_zone"] is not None
            and "NO ENTRY" not in result["entry_zone"]
            and result["targets"] is not None
        )

        if not is_actionable:
            i += 1
            continue

        direction = result["direction"]
        entry_price = float(closes.iloc[i])
        stop = result["invalidation"]
        tp1 = result["targets"][0]
        outcome = "TIMEOUT"
        resolved_at = min(i + BACKTEST_MAX_HOLDING_BARS, n - 1)

        for j in range(i + 1, min(i + BACKTEST_MAX_HOLDING_BARS + 1, n)):
            if direction == "BUY":
                if lows.iloc[j] <= stop:
                    outcome, resolved_at = "LOSS", j
                    break
                if highs.iloc[j] >= tp1:
                    outcome, resolved_at = "WIN", j
                    break
            else:
                if highs.iloc[j] >= stop:
                    outcome, resolved_at = "LOSS", j
                    break
                if lows.iloc[j] <= tp1:
                    outcome, resolved_at = "WIN", j
                    break

        return_pct = ((closes.iloc[resolved_at] / entry_price - 1) * 100 if direction == "BUY"
                      else (entry_price / closes.iloc[resolved_at] - 1) * 100)

        trades.append({
            "entry_index": i, "direction": direction, "confidence": result["confidence"],
            "entry_price": entry_price, "stop": stop, "tp1": tp1,
            "outcome": outcome, "bars_held": resolved_at - i, "return_pct": round(return_pct, 2),
        })
        i = resolved_at + 1

    return {"status": "OK", "trades": trades}


@st.cache_data(ttl=600, show_spinner=False)
def run_backtest(symbol: str) -> dict:
    client = BinanceMarketData()
    sym_klines = _safe_klines(client, symbol, CONFIG["primary_timeframe"], BACKTEST_KLINES_LIMIT)
    btc_klines = _safe_klines(client, "BTCUSDT", CONFIG["primary_timeframe"], BACKTEST_KLINES_LIMIT)

    if sym_klines is None:
        return {"status": "DATA UNAVAILABLE", "trades": []}

    core = _run_backtest_core(sym_klines, btc_klines)
    if core["status"] != "OK":
        return core

    trades = core["trades"]
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    timeouts = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    resolved = wins + losses
    win_rate = round(wins / resolved * 100, 1) if resolved > 0 else None
    avg_return = round(sum(t["return_pct"] for t in trades) / len(trades), 2) if trades else None

    date_start = date_end = None
    if "open_time" in sym_klines.columns:
        try:
            date_start = pd.to_datetime(sym_klines["open_time"].iloc[BACKTEST_MIN_HISTORY], unit="ms").strftime("%Y-%m-%d")
            date_end = pd.to_datetime(sym_klines["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
        except Exception:
            pass

    return {
        "status": "OK",
        "sample_size": len(trades),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_return,
        "period": f"{date_start} to {date_end}" if date_start else "unknown",
        "trades": trades,
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Crypto Scanner", page_icon="📊", layout="wide")


st.title("📊 Crypto Multi-Factor Scanner")
st.caption("Research view only — not trading advice. Confidence is a transparent weighted score, NOT a win probability.")
st.caption("Not included yet (blocked/needs another account): Futures flow, Open Interest, Funding rate "
           "(Binance futures API is geo-blocked with no public bypass found) and a News engine (needs a paid/free-key news API).")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Refresh now"):
        run_scan_cached.clear()

with st.spinner("Fetching live data from Binance..."):
    data = run_scan_cached()

regime_colors = {
    "STRONG_BULLISH": "🟢", "BULLISH": "🟢", "NEUTRAL": "🟡",
    "BEARISH": "🔴", "STRONG_BEARISH": "🔴", "UNCERTAIN": "⚪",
}
st.subheader(f"{regime_colors.get(data['btc_regime'], '⚪')} BTC Regime: {data['btc_regime']}")
st.caption(data.get("btc_regime_reason") or "")
st.caption(f"Eligible USDT pairs scanned: {data['universe_size']}")


def risk_label(r: dict) -> str:
    if r.get("status") != "OK":
        return "-"
    cls = r.get("exhaustion_analysis", {}).get("classification", "NO_CLEAR_SIGNAL")
    icons = {
        "EXHAUSTION_WARNING": "🟠 Exhaustion warning", "REVERSAL_WATCH": "🟡 Reversal watch",
        "BEARISH_CONFIRMATION": "🔴 Bearish confirmation", "SELLING_EXHAUSTION": "🟠 Selling exhaustion",
        "BULLISH_CONFIRMATION": "🟢 Bullish confirmation", "NO_CLEAR_SIGNAL": "⚪ No clear signal",
    }
    return icons.get(cls, cls)


def to_dataframe(rows):
    display_rows = []
    for r in rows:
        if r["status"] != "OK":
            display_rows.append({"Symbol": r["symbol"], "24h %": None, "Signal": r["status"],
                                  "Confidence": "-", "Data Quality": "-", "Entry Zone": "-", "Risk/Reversal": "-"})
        else:
            pct = r.get("pct_change_24h")
            display_rows.append({
                "Symbol": r["symbol"],
                "24h %": pct,  # kept numeric here so it can be colour-styled; formatted for display below
                "Signal": r["decision"],
                "Confidence": r["confidence"],
                "Data Quality": r["data_quality"],
                "Entry Zone": r.get("entry_zone") or "-",
                "Risk/Reversal": risk_label(r),
            })
    return pd.DataFrame(display_rows)


def _color_pct(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: #2ea043; font-weight: 700"
        elif val < 0:
            return "color: #f85149; font-weight: 700"
    return ""


def render_table(rows):
    df = to_dataframe(rows)
    styled = (
        df.style
        .applymap(_color_pct, subset=["24h %"])
        .format({"24h %": lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else "-"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


st.markdown("### Top Gainers")
render_table(data["gainers"])

st.markdown("### Top Losers")
render_table(data["losers"])

st.markdown("### BTC-Correlated Coins")
st.caption("Fixed watchlist of large-caps that tend to move with BTC — not filtered by today's % change.")
render_table(data["correlated"])


# ---- Deep-dive detail panel ----
st.markdown("---")
st.markdown("## 🔍 Deep-dive a coin")

all_rows = {r["symbol"]: r for r in (data["gainers"] + data["losers"] + data["correlated"]) if r["status"] == "OK"}
if not all_rows:
    st.info("No symbols with usable data yet — try Refresh.")
else:
    selected = st.selectbox("Choose a coin to see the full breakdown", sorted(all_rows.keys()))
    r = all_rows[selected]

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"{r['price']:.6g}")
    c2.metric("Signal", r["decision"], f"{r['confidence']} confidence")
    c3.metric("Risk/Reversal", risk_label(r))

    st.markdown("#### Why this signal")
    for f in r.get("factors", []):
        st.write(f"- **{f['name']}**: {f['value']} (score {f['score']}, {f['data_quality']})")

    st.markdown("#### Trade levels")
    if r.get("entry_zone") and "NO ENTRY" not in r["entry_zone"]:
        bar_note = "✅ meets this system's confidence bar" if r.get("meets_confidence_bar") else \
                   "⚠️ below this system's confidence bar — informational only, not an official setup"
        st.write(f"Direction: **{r.get('direction')}** ({bar_note})")
        st.write(f"- Entry zone: **{r['entry_zone']}**")
        st.write(f"- Invalidation / SL: **{r['invalidation']}**")
        st.write(f"- Targets (TP1/TP2/TP3): **{r['targets']}**")
        st.write(f"- Risk/Reward to TP1: **{r['risk_reward']}**")
    elif r.get("entry_zone"):
        st.write(r["entry_zone"])
    else:
        st.write("No directional bias — decision is " + r["decision"])


    st.markdown("#### Exhaustion / reversal evidence")
    exh = r.get("exhaustion_analysis", {})
    st.write(f"Classification: **{exh.get('classification')}**")
    for ev in exh.get("evidence", []):
        st.write(f"- {ev}")

    st.markdown("#### Liquidity events (equal highs/lows)")
    events = r.get("liquidity_events", [])
    if events:
        for e in events:
            st.write(f"- {e['type']} at {e['level']} → **{e['reaction']}**")
    else:
        st.write("No clear equal-high/low liquidity pools detected in the current window.")

    st.markdown("#### Spot buy/sell flow")
    sf = r.get("spot_flow", {})
    if sf.get("status") == "OK":
        st.write(f"Buy pressure **{sf['spot_buy_pressure_pct']}%** / Sell pressure **{sf['spot_sell_pressure_pct']}%** "
                 f"(price {sf['price_change_pct']:+.2f}% over the window)")
        st.write(sf["interpretation"])
    else:
        st.write("DATA UNAVAILABLE")

    with st.spinner("Fetching multi-timeframe structure + tokenomics..."):
        deep = get_symbol_deep_dive(selected)

    st.markdown("#### Multi-timeframe structure")
    tf_cols = st.columns(4)
    for col, tf in zip(tf_cols, ["15m", "1h", "4h", "1d"]):
        s = deep["structure_by_timeframe"].get(tf, {})
        col.write(f"**{tf}**")
        col.write(s.get("state", "DATA UNAVAILABLE"))

    st.markdown("#### Tokenomics / Supply")
    tok = deep["tokenomics"]
    if tok.get("status") == "OK":
        st.write(f"- Circulating supply: {tok['circulating_supply']}")
        st.write(f"- Total supply: {tok['total_supply']}")
        st.write(f"- Max supply: {tok['max_supply']}")
        st.write(f"- Market cap: ${tok['market_cap_usd']:,}" if tok['market_cap_usd'] else "- Market cap: DATA UNAVAILABLE")
        st.write(f"- FDV: ${tok['fdv_usd']:,}" if tok['fdv_usd'] else "- FDV: DATA UNAVAILABLE")
        st.write(f"- Supply status: {tok['supply_flag']}")
        st.caption(tok["note"])
    else:
        st.write(f"DATA UNAVAILABLE — {tok.get('reason', 'no match found')}")

    st.markdown("#### 📈 Real historical win rate (backtest)")
    st.caption("Computed from real past candles for this exact symbol — walked forward with no look-ahead. "
               "Simplifications: regime uses 1h data only during backtest, no fees/slippage modeled, "
               "in-sample replay (not a formal train/test split), trades taken non-overlapping.")
    if st.button(f"Run backtest for {selected}"):
        with st.spinner(f"Backtesting {selected} on real historical candles..."):
            bt = run_backtest(selected)
        if bt["status"] != "OK":
            st.write(f"{bt['status']} — not enough historical data to backtest this symbol yet.")
        elif bt["sample_size"] == 0:
            st.write("No qualifying signals were found in the backtest window — nothing to report "
                     "(this is a valid outcome, not an error).")
        else:
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Sample size", bt["sample_size"])
            bc2.metric("Win rate", f"{bt['win_rate_pct']}%" if bt["win_rate_pct"] is not None else "N/A")
            bc3.metric("Wins / Losses", f"{bt['wins']} / {bt['losses']}")
            bc4.metric("Timeouts", bt["timeouts"])
            st.caption(f"Period covered: {bt['period']} | Avg return per trade: {bt['avg_return_pct']}%")
            if bt["sample_size"] < 20:
                st.warning(f"Only {bt['sample_size']} historical trades found — too small a sample to "
                           "trust this win rate. Treat it as a rough indicator, not a real edge.")
            with st.expander("See individual backtested trades"):
                st.dataframe(pd.DataFrame(bt["trades"]), use_container_width=True, hide_index=True)

st.caption("Tap 'Run backtest' on a selected coin above for its real historical win rate. "
           "Data auto-refreshes every ~55s on reload; use the Refresh button for an immediate update.")
