"""Scanner: checks the SAME buy (entry) condition the unified-trading-engine
project uses (strategy.py, copied 2026-08-20 — see that file's docstring) but
on the WEEKLY timeframe, across the Nifty 500 universe (500 stocks), using
free yfinance data (no broker accounts needed).

INTENTIONAL: reproduces the pre-2026-08-13 arm_cycle_stale bug (see
strategy.decide_entry's docstring) — a bullish EMA9/30 crossover that armed
weeks ago and never got a bearish crossunder can still fire an entry today,
however stale. This is done by simply NOT passing `today` to decide_entry():
the staleness guard is gated behind `if today is not None`, so omitting it
skips the guard entirely without touching strategy.py itself. Requested as-is
(2026-08-20) for research/comparison purposes — do not "fix" this by adding
`today=` unless that's actually what's wanted. It's also not meaningful on
weekly bars anyway — see strategy.py's MAX_ARM_CYCLE_AGE_DAYS comment.

Usage:
    venv/Scripts/python.exe weekly_buy_scan.py
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy import build_indicators, decide_entry, MIN_BARS_REQUIRED

PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "data" / "nifty500_list.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / f"weekly_buy_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"

# 6 calendar years of daily bars resamples down to ~300+ weekly bars — comfortably
# above MIN_BARS_REQUIRED (109 weekly bars for EMA100 + its 9-period SMA to warm up).
DAILY_LOOKBACK_PERIOD = "6y"
CHUNK_SIZE = 50          # yfinance multi-ticker batch size per call
CHUNK_PAUSE_SECONDS = 1.5  # be polite to Yahoo between batches


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df["yf_symbol"] = df["Symbol"].astype(str).str.strip() + ".NS"
    return df


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the sibling backtest scripts' W-FRI convention (weekly candle
    closes Friday, matching NSE's trading week).

    pandas labels each weekly bucket with its Friday date regardless of
    whether that Friday's data actually exists yet — e.g. if the daily feed's
    last bar is Thursday, the current (still-forming, only Mon-Thu so far)
    week still gets labeled with THIS WEEK'S Friday, a date that hasn't
    traded yet. Live-caught 2026-08-20: a scan run midweek showed a
    "buy_signal_date" one day in the FUTURE, on an incomplete candle - not a
    real closed weekly bar. Drop that last bucket whenever its Friday label
    is after the latest real daily bar, so decide_entry() only ever sees
    fully-closed weekly candles."""
    weekly = daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    weekly = weekly.dropna(subset=["Close"])
    if len(weekly) and weekly.index[-1].date() > daily.index[-1].date():
        weekly = weekly.iloc[:-1]
    return weekly


def find_last_fire(enriched: pd.DataFrame) -> "dict | None":
    """Finds the most recent week the FULL buy condition freshly fired
    (entry_signal transitioning False->True) across the stock's WHOLE
    available weekly history — not just whether it's true on the latest bar.
    decide_entry() only ever looks at the last row, so on its own it can only
    answer "is the signal live THIS week"; this answers "which week did it
    last actually happen"."""
    fresh = enriched["entry_signal"] & ~enriched["entry_signal"].shift(1).fillna(False).astype(bool)
    fired = enriched[fresh]
    if fired.empty:
        return None
    fire_date = fired.index[-1]
    return {
        "signal_date": fire_date.date(),
        "close_at_signal": float(fired.iloc[-1]["Close"]),
        "weeks_ago": int((enriched.index[-1] - fire_date).days // 7),
    }


def fired_before_in_same_cycle(enriched: pd.DataFrame) -> bool:
    """True if entry_signal was ALREADY True at some earlier bar within the
    SAME arm cycle as the current (last) bar — i.e. this week's "fresh" fire
    is really a REPEAT within a cycle that already fired once before.

    Matters because a real live bot only takes ONE entry per arm cycle
    (`decide_entry`'s `used_arm_cycles` dedup) — it would have entered on the
    EARLIER fire and then rejected this one with reason="arm_cycle_already_used".
    This scan doesn't track trade history (`used_arm_cycles` is always
    empty), so it can't reproduce that rejection on its own; this check
    answers the same question after the fact by scanning the cycle's own
    bars. Live-caught 2026-08-20: 7 of 19 "fresh" hits (AUBANK, BEML,
    BERGEPAINT, CARBORUNIV, FINCABLES, KEI, NETWEB) were actually repeats —
    a real bot would never have seen these as new trades."""
    if len(enriched) < 2:
        return False
    current_arm_id = enriched.iloc[-1]["arm_cycle_id"]
    if pd.isna(current_arm_id):
        return False
    same_cycle_prior = enriched[enriched["arm_cycle_id"] == current_arm_id].iloc[:-1]
    return bool(same_cycle_prior["entry_signal"].any()) if len(same_cycle_prior) else False


def fetch_daily_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=symbols, period=DAILY_LOOKBACK_PERIOD, interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = raw[symbol] if len(symbols) > 1 else raw
        except (KeyError, TypeError):
            continue
        df = df.dropna(subset=["Close"]) if "Close" in df.columns else pd.DataFrame()
        if not df.empty:
            out[symbol] = df
    return out


def scan() -> pd.DataFrame:
    universe = load_universe()
    symbols = universe["yf_symbol"].tolist()
    name_by_symbol = dict(zip(universe["yf_symbol"], universe["Company Name"]))
    print(f"Universe: {len(symbols)} symbols")

    rows = []
    fetched = 0
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i + CHUNK_SIZE]
        print(f"Fetching {i + 1}-{min(i + CHUNK_SIZE, len(symbols))}/{len(symbols)} ...")
        try:
            daily_by_symbol = fetch_daily_batch(chunk)
        except Exception as e:
            print(f"  batch fetch failed: {e}")
            daily_by_symbol = {}

        for symbol in chunk:
            daily = daily_by_symbol.get(symbol)
            if daily is None or daily.empty:
                rows.append({"symbol": symbol, "company": name_by_symbol.get(symbol, ""),
                              "signal": False, "reason": "no_data", "close": None})
                continue

            weekly = resample_weekly(daily)
            if len(weekly) < MIN_BARS_REQUIRED:
                rows.append({"symbol": symbol, "company": name_by_symbol.get(symbol, ""),
                              "signal": False, "reason": "insufficient_history", "close": None})
                continue

            enriched = build_indicators(weekly)
            # No `today=` passed on purpose — keeps the arm_cycle staleness guard
            # OFF, i.e. reproduces the pre-fix production bug (see module docstring).
            check = decide_entry(enriched)
            last = enriched.iloc[-1]
            arm_age_weeks = None
            if pd.notna(last["arm_cycle_id"]):
                arm_age_weeks = (weekly.index[-1] - pd.Timestamp(last["arm_cycle_id"])).days // 7

            last_fire = find_last_fire(enriched)

            # Latest actual traded price — the daily feed's own last row, which
            # is fresher than the weekly bar's close (that's pinned to Friday;
            # this reflects the most recent daily close/live price fetched).
            current_price = float(daily["Close"].iloc[-1])
            pct_change_since_signal = None
            if last_fire and last_fire["close_at_signal"]:
                pct_change_since_signal = round(
                    (current_price - last_fire["close_at_signal"]) / last_fire["close_at_signal"] * 100, 2
                )

            rows.append({
                "symbol": symbol, "company": name_by_symbol.get(symbol, ""),
                "signal": check.signal, "reason": check.reason, "close": check.close,
                "ema_sep_pct": round(float(last["ema_sep_pct"]), 3) if pd.notna(last["ema_sep_pct"]) else None,
                "macd_bullish": bool(last["macd_bullish"]),
                "trend_bullish": bool(last["trend_bullish"]),
                "arm_cycle_age_weeks": arm_age_weeks,
                "last_weekly_bar": weekly.index[-1].date(),
                "last_signal_date": last_fire["signal_date"] if last_fire else None,
                "last_signal_close": last_fire["close_at_signal"] if last_fire else None,
                "signal_weeks_ago": last_fire["weeks_ago"] if last_fire else None,
                "current_price": current_price,
                "pct_change_since_signal": pct_change_since_signal,
                # True => a real live bot would have ALREADY entered on an
                # earlier bar in this same arm cycle and blocked this one via
                # used_arm_cycles — see fired_before_in_same_cycle's docstring.
                "repeat_within_arm_cycle": fired_before_in_same_cycle(enriched) if check.signal else None,
            })
            fetched += 1

        time.sleep(CHUNK_PAUSE_SECONDS)

    result = pd.DataFrame(rows)
    print(f"\nFetched & evaluated: {fetched}/{len(symbols)}")
    return result


def clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Shared symbol/company/price/date presentation used by both the CLI
    output below and the WhatsApp PDF report (weekly_whatsapp_report.py) —
    keeps the two outputs' columns/formatting identical by construction."""
    out = df[["symbol", "company", "last_signal_close", "last_signal_date",
              "current_price", "pct_change_since_signal"]].copy()
    out["symbol"] = out["symbol"].str.replace(".NS", "", regex=False)
    out = out.rename(columns={"last_signal_close": "price_at_signal", "last_signal_date": "buy_signal_date",
                               "current_price": "ltp", "pct_change_since_signal": "pct_change"})
    return out.sort_values("symbol")


def split_hits(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits scan() output into (fresh, repeats) — see the PRIMARY/SEPARATE
    comments below for what each means."""
    hits = result[result["signal"] == True].copy()  # noqa: E712
    fresh = clean_view(hits[hits["repeat_within_arm_cycle"] == False])  # noqa: E712
    repeats = clean_view(hits[hits["repeat_within_arm_cycle"] == True])  # noqa: E712
    return fresh, repeats


def main() -> None:
    result = scan()
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Full results saved: {OUTPUT_CSV}")

    fresh, repeats = split_hits(result)

    # PRIMARY: genuinely fresh — first entry_signal fire of this arm cycle.
    # This is what a real live bot (which dedupes via used_arm_cycles) would
    # actually treat as a new trade opportunity.
    print(f"\n=== GENUINELY FRESH (first fire this arm cycle): {len(fresh)} stocks ===")
    if not fresh.empty:
        fresh_path = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_fresh.csv")
        fresh.to_csv(fresh_path, index=False)
        print(fresh.to_string(index=False))
        print(f"\nSaved: {fresh_path}")
    else:
        print("(none)")

    # SEPARATE, NOT hidden: live on the latest bar but a REPEAT within the
    # same arm cycle — a real bot would have entered earlier and rejected
    # this one via used_arm_cycles (arm_cycle_already_used). Shown on its own
    # so it's never silently mixed into the "fresh opportunities" list.
    print(f"\n=== LIVE BUT A REPEAT WITHIN THE SAME ARM CYCLE (a real bot would "
          f"NOT have taken these — already used): {len(repeats)} stocks ===")
    if not repeats.empty:
        repeats_path = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_repeats.csv")
        repeats.to_csv(repeats_path, index=False)
        print(repeats.to_string(index=False))
        print(f"\nSaved: {repeats_path}")
    else:
        print("(none)")

    stale = result[(result["signal"] != True) & result["last_signal_date"].notna()]  # noqa: E712
    print(f"\n({len(stale)} more stocks had a buy signal in the past but it's not live this week — "
          f"see the full CSV's last_signal_date/signal_weeks_ago columns.)")


if __name__ == "__main__":
    main()
