"""
Nifty 500 — 10%+ Move Alert Scanner (NSE)
=========================================
Scans the Nifty 500 universe for stocks that look primed for a 10%+ upside
move, using the classic "coiled spring" evidence stack:

  LEVELS     * price coiled within a few % of its prior 20-day high
             * within striking distance of the 52-week high (no overhead supply)
  VOLATILITY * Bollinger-Band-width squeeze — today's 20-day band width sits in
               the bottom quartile of its own 1-year range (energy stored)
             * tight 5-day closing range (hands have stopped changing)
  VOLUME     * accumulation — up-day volume outweighs down-day volume over 20d
             * volume dry-up in the base and/or an RVOL surge today
               (pace-adjusted for how much of the 09:15–15:30 session is done)
  TREND      * price above a rising 50-DMA, 50-DMA above 200-DMA
  FEASIBILITY* ATR% high enough that a 10% move within ~10 sessions is
               realistic for this name (a 0.8%-a-day mega cap can't do it)

Each stock gets a MOVE-READINESS SCORE (0–100) plus the list of reasons that
fired. Tiers:

  🔥 READY   score >= 70  — setup complete, trigger = break of the 20-day high
  🌱 SETUP   score 55–69  — base forming, needs a squeeze/volume confirmation
  👀 WATCH   score 45–54  — early, keep on the radar

The scanner also self-validates: it replays the same core setup over the past
year across the whole universe and reports how often it actually resolved into
a >= 10% gain within the next 10 sessions, versus the base rate of any random
day doing so. That hit-rate is printed in the report header so the "high
probability" claim is measured, not asserted.

Output: move_alerts.md (Markdown) and move_alerts.json (for the dashboard).

Usage:  pip install yfinance pandas requests && python move_scanner.py
"""

import datetime as dt
import io
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

FWD_WINDOW = 10      # sessions ahead in which the 10% move must happen
TARGET_MOVE = 0.10   # +10%
MIN_ATR_PCT = 1.2    # below this daily range, a 10% burst is unrealistic
MIN_PRICE = 30.0     # skip micro-priced names where 10% is noise

# NSE macro-industry names → familiar F&O-style sector labels
SECTOR_RENAME = {
    "Financial Services": "Financial Services",
    "Information Technology": "IT",
    "Oil Gas & Consumable Fuels": "Oil & Gas",
    "Fast Moving Consumer Goods": "FMCG",
    "Automobile and Auto Components": "Auto",
    "Healthcare": "Pharma & Healthcare",
    "Metals & Mining": "Metals & Mining",
    "Consumer Services": "Consumer Services",
    "Power": "Power",
    "Capital Goods": "Capital Goods",
    "Construction": "Infrastructure",
    "Construction Materials": "Cement",
    "Telecommunication": "Telecom",
    "Chemicals": "Chemicals",
    "Consumer Durables": "Consumer Durables",
    "Realty": "Realty",
    "Media Entertainment & Publication": "Media",
    "Services": "Services",
    "Forest Materials": "Forest Materials",
    "Diversified": "Diversified",
    "Textiles": "Textiles",
}

# Embedded fallback universe (liquid NSE large/mid caps) if the NSE file fails.
FALLBACK_UNIVERSE = """
RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL SBIN ITC HINDUNILVR LT
BAJFINANCE HCLTECH MARUTI SUNPHARMA KOTAKBANK TITAN ULTRACEMCO AXISBANK NTPC ONGC
ADANIENT ADANIPORTS POWERGRID M&M TATAMOTORS TATASTEEL WIPRO COALINDIA BAJAJFINSV NESTLEIND
ASIANPAINT JSWSTEEL HINDALCO DLF GRASIM SBILIFE HDFCLIFE TECHM DIVISLAB DRREDDY
CIPLA APOLLOHOSP BRITANNIA EICHERMOT BPCL IOC HEROMOTOCO BAJAJ-AUTO TATAPOWER TATACONSUM
GODREJCP PIDILITIND SIEMENS ABB HAVELLS AMBUJACEM SHREECEM VEDL INDUSINDBK TRENT
ETERNAL DMART IRCTC HAL BEL VBL LTIM NAUKRI JUBLFOOD PERSISTENT
POLYCAB DIXON ASTRAL CUMMINSIND BHEL BHARATFORG MOTHERSON TVSMOTOR ASHOKLEY EXIDEIND
CANBK PNB BANKBARODA UNIONBANK IDFCFIRSTB FEDERALBNK AUBANK BANDHANBNK CHOLAFIN MUTHOOTFIN
SHRIRAMFIN LICHSGFIN RECLTD PFC IRFC INDIGO CONCOR GMRAIRPORT ADANIGREEN ADANIPOWER
JSWENERGY NHPC SJVN TORNTPHARM LUPIN AUROPHARMA ALKEM ZYDUSLIFE BIOCON GLENMARK
LAURUSLABS MANKIND MAXHEALTH FORTIS SYNGENE PIIND SRF UPL DEEPAKNTR TATACHEM
COLPAL DABUR MARICO EMAMILTD PGHH UBL RADICO PAGEIND ABFRL BATAINDIA
OBEROIRLTY GODREJPROP LODHA PRESTIGE PHOENIXLTD NCC KEC RVNL IRCON NBCC
JINDALSTEL SAIL NMDC NATIONALUM HINDZINC APLAPOLLO RATNAMANI OFSS COFORGE MPHASIS
TATAELXSI KPITTECH CYIENT ZENSARTECH BSOFT TATACOMM IDEA INDUSTOWER HFCL TEJASNET
""".split()


def load_universe():
    """Nifty 500 symbols + sector map from NSE archives, with graceful fallbacks."""
    import requests

    for url, label in ((NIFTY500_URL, "Nifty 500"), (NIFTY200_URL, "Nifty 200")):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            symbols = df["Symbol"].astype(str).str.strip().tolist()
            industries = df.get("Industry", pd.Series(dtype=str)).astype(str).str.strip()
            sectors = {
                s: SECTOR_RENAME.get(i, i or "Other")
                for s, i in zip(symbols, industries)
            }
            if len(symbols) >= 150:
                return symbols, sectors, f"{label} (NSE constituents file)"
        except Exception as exc:
            print(f"  ! {label} download failed ({exc})")
    return (
        FALLBACK_UNIVERSE,
        {},
        f"embedded liquid-stock list ({len(FALLBACK_UNIVERSE)} names)",
    )


def session_elapsed_fraction(now_ist: dt.datetime) -> float:
    """Fraction of the 09:15-15:30 IST session elapsed; 1.0 outside market hours."""
    open_t = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    if now_ist <= open_t or now_ist >= close_t or now_ist.weekday() >= 5:
        return 1.0
    frac = (now_ist - open_t).total_seconds() / (close_t - open_t).total_seconds()
    return max(frac, 0.10)  # avoid silly RVOL inflation right at the open


def download_history(symbols, chunk_size=50):
    """Batch-download 1y of daily bars. Returns {symbol: OHLCV DataFrame}."""
    tickers = [s + ".NS" for s in symbols]
    out = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        data = yf.download(
            chunk,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        for t in chunk:
            try:
                hist = data[t].dropna(how="all") if len(chunk) > 1 else data.dropna(how="all")
                if hist is not None and len(hist.dropna()) >= 60:
                    out[t[:-3]] = hist
            except Exception:
                pass
    return out


def feature_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-day feature series for one stock (used for both today's score and
    the historical validation replay)."""
    c, h, l, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    f = pd.DataFrame(index=hist.index)
    f["close"], f["high"], f["volume"] = c, h, v

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c * 100
    # typical volatility for the 10%-feasibility gate — current ATR is
    # deliberately LOW in a squeeze, so gate on the name's normal energy level
    f["atr_typ"] = f["atr_pct"].expanding(min_periods=30).median()

    sma20 = c.rolling(20).mean()
    f["bbw"] = (4 * c.rolling(20).std()) / sma20 * 100          # band width, % of price
    f["bbw_rank"] = f["bbw"].rank(pct=True)                     # 1y percentile of squeeze
    f["tight5"] = (c.rolling(5).max() - c.rolling(5).min()) / c * 100

    prior_20d_high = h.shift(1).rolling(20).max()
    f["dist_20d_high"] = (prior_20d_high - c) / c * 100         # <=0 means broken out
    f["dist_52w_high"] = (h.shift(1).expanding().max() - c) / c * 100
    f["prior_20d_high"] = prior_20d_high

    up = (c > prev_c).astype(float)
    f["updown_vol"] = (v * up).rolling(20).sum() / (v * (1 - up)).rolling(20).sum().clip(lower=1)
    f["vol_dryup"] = v.rolling(5).mean() / v.rolling(20).mean()
    f["rvol"] = v / v.shift(1).rolling(20).mean()               # vs 20d avg (yesterday back)

    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["uptrend"] = (c > sma50) & (sma50 > sma50.shift(5))
    f["strong_trend"] = f["uptrend"] & (sma50 > sma200)
    return f


def core_setup_mask(f: pd.DataFrame) -> pd.Series:
    """The single boolean 'coiled spring' condition used for validation:
    squeeze + coiled under the 20d high + accumulation + uptrend + enough ATR."""
    return (
        (f["bbw_rank"] <= 0.25)
        & (f["dist_20d_high"].between(-1.0, 4.0))
        & (f["updown_vol"] >= 1.2)
        & f["uptrend"]
        & (f["atr_typ"] >= MIN_ATR_PCT)
        & (f["close"] >= MIN_PRICE)
    )


def validate(features_by_symbol) -> dict:
    """Replay the setup over the past year: how often did a >=10% gain follow
    within FWD_WINDOW sessions? The base rate is MATCHED to the same
    feasibility gates (price, typical ATR), so the comparison is apples to
    apples — an unmatched base would be dominated by wild small-cap days the
    scanner never trades. Two variants are measured:
      coil          — the setup alone (waiting under the 20-day high)
      coil+breakout — setup AND price clears the prior 20-day high that day
                      (the actual entry trigger the report recommends)"""
    setup_days = setup_hits = 0
    conf_days = conf_hits = 0
    base_days = base_hits = 0
    for f in features_by_symbol.values():
        fwd_high = f["high"][::-1].rolling(FWD_WINDOW).max()[::-1].shift(-1)
        fwd_ret = fwd_high / f["close"] - 1
        valid = (
            fwd_ret.notna()
            & f["bbw_rank"].notna()
            & (f["close"] >= MIN_PRICE)
            & (f["atr_typ"] >= MIN_ATR_PCT)
        )
        hit = fwd_ret >= TARGET_MOVE
        setup = core_setup_mask(f) & valid
        confirmed = setup & (f["close"] > f["prior_20d_high"])
        setup_days += int(setup.sum())
        setup_hits += int((setup & hit).sum())
        conf_days += int(confirmed.sum())
        conf_hits += int((confirmed & hit).sum())
        base_days += int(valid.sum())
        base_hits += int((valid & hit).sum())
    setup_rate = setup_hits / setup_days if setup_days else 0.0
    conf_rate = conf_hits / conf_days if conf_days else 0.0
    base_rate = base_hits / base_days if base_days else 0.0
    return {
        "setup_days": setup_days,
        "setup_hits": setup_hits,
        "setup_hit_rate": round(setup_rate, 4),
        "confirm_days": conf_days,
        "confirm_hits": conf_hits,
        "confirm_hit_rate": round(conf_rate, 4),
        "base_days": base_days,
        "base_hit_rate": round(base_rate, 4),
        "lift": round(setup_rate / base_rate, 2) if base_rate else None,
        "confirm_lift": round(conf_rate / base_rate, 2) if base_rate else None,
        "definition": (
            f"hit = high reaches +{TARGET_MOVE:.0%} above the signal close "
            f"within the next {FWD_WINDOW} sessions (past 1y; base rate is "
            f"matched to the same price/ATR gates)"
        ),
    }


def score_today(f: pd.DataFrame, elapsed_frac: float):
    """Score the most recent bar. Returns (score, reasons, stats) or None."""
    r = f.iloc[-1]
    if not np.isfinite(r["bbw_rank"]) or r["close"] < MIN_PRICE:
        return None
    if not np.isfinite(r["atr_typ"]) or r["atr_typ"] < MIN_ATR_PCT:
        return None  # this name can't plausibly travel 10% in ~2 weeks

    # live RVOL, pace-adjusted for the fraction of the session elapsed
    avg20 = f["volume"].iloc[-21:-1].mean()
    rvol = r["volume"] / (avg20 * elapsed_frac) if avg20 > 0 else 0.0

    score, reasons = 0.0, []

    if r["bbw_rank"] <= 0.10:
        score += 20; reasons.append("extreme volatility squeeze (bottom 10% of 1y)")
    elif r["bbw_rank"] <= 0.25:
        score += 14; reasons.append("volatility squeeze (bottom 25% of 1y)")
    elif r["bbw_rank"] <= 0.40:
        score += 7

    if r["tight5"] <= 3.0:
        score += 10; reasons.append(f"very tight 5-day range ({r['tight5']:.1f}%)")
    elif r["tight5"] <= 5.0:
        score += 6

    d20 = r["dist_20d_high"]
    if d20 <= 0:
        score += 15; reasons.append("breaking the 20-day high NOW")
    elif d20 <= 2.0:
        score += 12; reasons.append(f"coiled {d20:.1f}% under the 20-day high")
    elif d20 <= 4.0:
        score += 8; reasons.append(f"{d20:.1f}% below the 20-day high")

    d52 = r["dist_52w_high"]
    if d52 <= 5.0:
        score += 10; reasons.append("within 5% of 52-week high (no overhead supply)")
    elif d52 <= 12.0:
        score += 6

    if rvol >= 3.0:
        score += 15; reasons.append(f"volume surge {rvol:.1f}x average")
    elif rvol >= 2.0:
        score += 11; reasons.append(f"volume {rvol:.1f}x average")
    elif rvol >= 1.5:
        score += 6
    elif r["vol_dryup"] <= 0.65:
        score += 5; reasons.append("volume dry-up in the base (supply exhausted)")

    if r["updown_vol"] >= 1.6:
        score += 12; reasons.append(f"heavy accumulation (up/down volume {r['updown_vol']:.1f})")
    elif r["updown_vol"] >= 1.2:
        score += 7; reasons.append("up-day volume outweighs down-day volume")

    if r["strong_trend"]:
        score += 11; reasons.append("uptrend: price > rising 50-DMA > 200-DMA")
    elif r["uptrend"]:
        score += 6

    if r["atr_typ"] >= 2.5:
        score += 5; reasons.append(f"high-energy name (typical ATR {r['atr_typ']:.1f}%/day)")

    stats = {
        "rvol": rvol,
        "atr_pct": float(r["atr_typ"]),
        "bbw_rank": float(r["bbw_rank"]),
        "tight5": float(r["tight5"]),
        "dist_20d_high": float(d20),
        "dist_52w_high": float(d52),
        "updown_vol": float(r["updown_vol"]),
        "trigger": float(r["prior_20d_high"]) if np.isfinite(r["prior_20d_high"]) else None,
        "target": float(r["close"]) * (1 + TARGET_MOVE),
        "stop": float(r["close"]) * (1 - 0.04),
    }
    return min(round(score), 100), reasons, stats


def tier_of(score: int) -> str:
    if score >= 70:
        return "READY"
    if score >= 55:
        return "SETUP"
    return "WATCH"


def scan(hist_by_symbol, sectors, elapsed_frac):
    rows, features_by_symbol = [], {}
    for symbol, hist in hist_by_symbol.items():
        try:
            f = feature_frame(hist)
            features_by_symbol[symbol] = f
            scored = score_today(f, elapsed_frac)
            if scored is None:
                continue
            score, reasons, s = scored
            if score < 45:
                continue
            closes = hist["Close"].dropna()
            prev = float(closes.iloc[-2])
            rows.append(
                {
                    "Stock": symbol,
                    "Sector": sectors.get(symbol, "Other"),
                    "Price": float(closes.iloc[-1]),
                    "% Chg": (float(closes.iloc[-1]) - prev) / prev * 100,
                    "Score": score,
                    "Tier": tier_of(score),
                    "RVOL": round(s["rvol"], 2),
                    "ATR %": round(s["atr_pct"], 2),
                    "Squeeze %ile": round(s["bbw_rank"] * 100),
                    "To 20D High %": round(s["dist_20d_high"], 2),
                    "To 52W High %": round(s["dist_52w_high"], 2),
                    "Up/Down Vol": round(s["updown_vol"], 2),
                    "Trigger": round(s["trigger"], 2) if s["trigger"] else None,
                    "Target (+10%)": round(s["target"], 2),
                    "Stop (-4%)": round(s["stop"], 2),
                    "Reasons": reasons,
                    "Spark": [round(float(c), 2) for c in closes.iloc[-30:]],
                }
            )
        except Exception as exc:
            print(f"  ! {symbol}: {exc}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Score", "RVOL"], ascending=False).reset_index(drop=True)
    return df, features_by_symbol


def fmt_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Stock | Price ₹ | Score | RVOL | Squeeze | To 20D Hi | Trigger ₹ | Why |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| **{r['Stock']}** | {r['Price']:,.1f} | **{r['Score']}** "
            f"| {r['RVOL']:.1f}x | p{r['Squeeze %ile']:.0f} | {r['To 20D High %']:+.1f}% "
            f"| {r['Trigger'] or 0:,.1f} | {'; '.join(r['Reasons'][:3])} |"
        )
    return lines


def render(df, validation, universe_note, now_ist, elapsed_frac, scanned):
    in_session = elapsed_frac < 1.0
    status = (
        f"market OPEN — {elapsed_frac:.0%} of session elapsed (RVOL pace-adjusted)"
        if in_session
        else "market CLOSED — full-day data from the last session"
    )
    v = validation
    lines = [
        "# ⚡ Nifty 500 — 10%+ Move Alerts",
        "",
        f"**Run time:** {now_ist.strftime('%A, %B %d, %Y %I:%M %p IST')}  ",
        f"**Market status:** {status}  ",
        f"**Universe:** {universe_note} · {scanned} stocks with usable data  ",
        "",
        "> **Strategy:** volatility squeeze + tight base coiled under the 20-day high",
        "> + volume accumulation + uptrend, on names volatile enough to travel 10%.",
        f"> **Self-check (past 1y, this universe, matched base rate {v['base_hit_rate']:.0%}):** "
        f"coil alone hit +10% within {FWD_WINDOW} sessions {v['setup_hit_rate']:.0%} of the time "
        f"({v['setup_hits']}/{v['setup_days']}, {v['lift']}x); waiting for the **breakout trigger** "
        f"raised it to **{v['confirm_hit_rate']:.0%}** ({v['confirm_hits']}/{v['confirm_days']}, "
        f"**{v['confirm_lift']}x lift**).",
        "> Entry idea: buy the break of the Trigger (prior 20-day high) with volume;",
        "> stop ≈ −4%; target +10%. Not investment advice.",
        "",
    ]
    for tier, emoji, blurb in (
        ("READY", "🔥", "setup complete — watch the trigger"),
        ("SETUP", "🌱", "base forming — needs confirmation"),
        ("WATCH", "👀", "early — on the radar"),
    ):
        sub = df[df["Tier"] == tier] if not df.empty else df
        lines += [f"## {emoji} {tier} ({len(sub)}) — {blurb}", ""]
        lines += fmt_table(sub.head(25)) if len(sub) else ["_None right now._"]
        if len(sub) > 25:
            lines.append(f"\n_...and {len(sub) - 25} more not shown._")
        lines.append("")
    return "\n".join(lines)


def main():
    now_ist = dt.datetime.now(IST)
    elapsed_frac = session_elapsed_fraction(now_ist)
    symbols, sectors, universe_note = load_universe()
    print(f"Universe: {universe_note} — downloading history for {len(symbols)} symbols...")
    hist = download_history(symbols)
    if not hist:
        raise SystemExit("No data retrieved — check network access to Yahoo Finance.")
    print(f"Got usable history for {len(hist)} symbols. Scoring...")
    df, features = scan(hist, sectors, elapsed_frac)
    validation = validate(features)
    md = render(df, validation, universe_note, now_ist, elapsed_frac, len(hist))
    with open("move_alerts.md", "w") as fh:
        fh.write(md + "\n")
    payload = {
        "run_time_ist": now_ist.strftime("%Y-%m-%d %I:%M %p IST"),
        "elapsed_frac": round(elapsed_frac, 3),
        "market_open": elapsed_frac < 1.0,
        "universe": universe_note,
        "scanned": len(hist),
        "params": {
            "target_move": TARGET_MOVE,
            "fwd_window_sessions": FWD_WINDOW,
            "min_atr_pct": MIN_ATR_PCT,
        },
        "validation": validation,
        "rows": df.to_dict(orient="records") if not df.empty else [],
    }
    with open("move_alerts.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(md)


if __name__ == "__main__":
    main()
