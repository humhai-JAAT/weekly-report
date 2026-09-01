"""Entry logic — copied from unified-trading-engine/engine/strategy.py
(2026-08-20) into this standalone "weekly report" project, per the user's
established project-separation convention (each bot/tool is its own folder,
never importing another project's code — see the sibling bots' rules.md).
This is a SNAPSHOT: if the source project's strategy.py changes later
(e.g. a new bug fix), it will NOT automatically propagate here — re-copy by
hand if this project needs to stay in lockstep.

Port of the user's Pine Script "EMA-MACD V2.1.2" — entry logic only,
unchanged from the sibling bots (intraday-trading-bot v1/v2, bot-v3,
unified-trading-engine). Timeframe-agnostic: works on whatever OHLC bars are
passed in (this project evaluates it on WEEKLY candles).

Pine reference (entry side):
    fastEMA=9  slowEMA=30  trendEMA=100  trendSMA=9 (SMA of EMA100)
    MACD(12,26,9)
    emaSepMinPct = 0.6
    setupArmed: latched True on a bullish EMA9/EMA30 crossover, latched False on
                a bearish crossover, and consumed (set False) the instant an
                entry fires — so only one entry per arm cycle.
    entryCondition = setupArmed and emaFast>emaSlow and macd>signal and
                     ema100>sma(ema100,9) and emaSepPct>=0.6 and flat
"""

from dataclasses import dataclass

import pandas as pd

from indicators import ema, macd, sma

FAST_EMA = 9
SLOW_EMA = 30
TREND_EMA = 100
TREND_SMA = 9
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_SEP_MIN_PCT = 0.6

MIN_BARS_REQUIRED = TREND_EMA + TREND_SMA  # EMA100 + its SMA need this much warm-up

# How many calendar days old the ARM CYCLE (the EMA9/30 crossover bar) is
# allowed to be, relative to the bar being evaluated. Calibrated for
# intraday/daily use in the source project (survives a Friday->Monday gap) —
# NOT meaningful on the WEEKLY timeframe this project runs on, since
# consecutive weekly bars are already 7 days apart. This project's own
# weekly_buy_scan.py deliberately never passes `today=` to decide_entry() for
# exactly this reason — see that file's module docstring.
MAX_ARM_CYCLE_AGE_DAYS = 3


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["ema_fast"] = ema(close, FAST_EMA)
    out["ema_slow"] = ema(close, SLOW_EMA)
    out["ema_trend"] = ema(close, TREND_EMA)
    out["ema_trend_sma"] = sma(out["ema_trend"], TREND_SMA)

    macd_df = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["macd_signal"]

    out["ema_sep_pct"] = (out["ema_fast"] - out["ema_slow"]) / out["ema_slow"] * 100

    bull_cross = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    bear_cross = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))

    arm_signal = pd.Series(float("nan"), index=out.index, dtype="float64")
    arm_signal[bull_cross] = 1.0
    arm_signal[bear_cross] = 0.0
    out["armed"] = arm_signal.ffill().fillna(0.0).astype(bool)

    # Timestamp of the bull-cross bar that produced the *current* armed state —
    # used to make sure we only take one entry per arm cycle (mirrors Pine's
    # setupArmed being consumed the instant an entry fires).
    bull_cross_time = pd.Series(pd.NaT, index=out.index, dtype=out.index.dtype)
    bull_cross_time[bull_cross] = out.index[bull_cross]
    out["arm_cycle_id"] = bull_cross_time.ffill()

    out["macd_bullish"] = out["macd"] > out["macd_signal"]
    out["trend_bullish"] = out["ema_trend"] > out["ema_trend_sma"]
    out["ema_sep_ok"] = out["ema_sep_pct"] >= EMA_SEP_MIN_PCT

    out["entry_signal"] = (
        out["armed"] & (out["ema_fast"] > out["ema_slow"]) & out["macd_bullish"]
        & out["trend_bullish"] & out["ema_sep_ok"]
    )
    return out


@dataclass
class EntryCheck:
    signal: bool
    arm_cycle_id: "pd.Timestamp | None"
    close: float
    reason: str


def decide_entry(enriched: pd.DataFrame, used_arm_cycles: set[str] = frozenset(),
                  today: "pd.Timestamp | None" = None) -> EntryCheck:
    """See MAX_ARM_CYCLE_AGE_DAYS's comment above — `today` is only meaningful
    on intraday/daily bars in the source project; this project's own caller
    never passes it."""
    if len(enriched) < MIN_BARS_REQUIRED:
        return EntryCheck(False, None, float(enriched["Close"].iloc[-1]) if len(enriched) else 0.0,
                           "insufficient_history")

    last = enriched.iloc[-1]
    close = float(last["Close"])

    if not bool(last["entry_signal"]):
        return EntryCheck(False, last["arm_cycle_id"], close, "no_signal")

    # Only take the trade if entry_signal *just* turned true on this exact bar (false
    # on the previous bar). Still only looks 1 bar back — a signal flickering
    # True->False->True within one arm cycle isn't caught by this alone; see
    # weekly_buy_scan.py's fired_before_in_same_cycle() for the broader check
    # this project adds on top.
    if len(enriched) > 1 and bool(enriched.iloc[-2]["entry_signal"]):
        return EntryCheck(False, last["arm_cycle_id"], close, "signal_not_fresh")

    if today is not None and pd.notna(last["arm_cycle_id"]):
        age_days = (today.date() - pd.Timestamp(last["arm_cycle_id"]).date()).days
        if age_days > MAX_ARM_CYCLE_AGE_DAYS:
            return EntryCheck(False, last["arm_cycle_id"], close, "arm_cycle_stale")

    arm_id_str = str(last["arm_cycle_id"]) if pd.notna(last["arm_cycle_id"]) else None
    if arm_id_str is not None and arm_id_str in used_arm_cycles:
        return EntryCheck(False, last["arm_cycle_id"], close, "arm_cycle_already_used")

    return EntryCheck(True, last["arm_cycle_id"], close, "entry")
