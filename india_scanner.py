"""
Indian Stocks Dashboard — High RVOL & Breakouts (NSE)
=====================================================
Scans the Nifty 200 universe (falls back to an embedded liquid-stock list if
the NSE constituents file is unreachable) and flags:

  * High RVOL   — today's volume vs its 20-day average, adjusted for how much
                  of the 09:15–15:30 IST session has elapsed, so a 2.45pm run
                  is comparable to a full day
  * Breakouts   — price taking out the prior 20-day high, and fresh 52-week highs

Dashboard sections:
  1. High RVOL + Breakout (RVOL >= 1.5 and a breakout)  — the A-list
  2. Breakouts (any RVOL)
  3. High RVOL (>= 2.0) without a breakout — unusual activity to watch

Output: india_dashboard.md (Markdown), also printed to stdout.

Usage:  pip install yfinance pandas requests && python india_scanner.py
"""

import datetime as dt
import io
import json
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

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
}


def load_universe():
    """Nifty 200 symbols + sector map from NSE archives, else the embedded fallback."""
    try:
        import requests

        resp = requests.get(
            NIFTY200_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        industries = df.get("Industry", pd.Series(dtype=str)).astype(str).str.strip()
        sectors = {
            s: SECTOR_RENAME.get(i, i or "Other")
            for s, i in zip(symbols, industries)
        }
        if len(symbols) >= 150:
            return symbols, sectors, "Nifty 200 (NSE constituents file)"
    except Exception as exc:
        print(f"  ! Nifty 200 download failed ({exc}); using embedded fallback list")
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


def scan(symbols, elapsed_frac, sectors):
    """Batch-download 1y of daily bars and compute RVOL/breakout stats."""
    tickers = [s + ".NS" for s in symbols]
    rows, failures = [], []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
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
            symbol = t[:-3]
            try:
                hist = data[t].dropna(how="all") if len(chunk) > 1 else data.dropna(how="all")
                closes = hist["Close"].dropna()
                if len(closes) < 30:
                    failures.append(symbol)
                    continue
                highs, vols = hist["High"].dropna(), hist["Volume"].dropna()
                price = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                chg = (price - prev) / prev * 100.0

                vol_today = float(vols.iloc[-1])
                avg20 = float(vols.iloc[-21:-1].mean())
                rvol = vol_today / (avg20 * elapsed_frac) if avg20 > 0 else 0.0

                high_today = float(highs.iloc[-1])
                prior_20d_high = float(highs.iloc[-21:-1].max())
                prior_52w_high = float(highs.iloc[:-1].max())
                bo_20d = high_today > prior_20d_high
                bo_52w = high_today > prior_52w_high
                breakout = "52W High" if bo_52w else ("20D High" if bo_20d else "")

                low_52w = float(hist["Low"].dropna().min())
                rows.append(
                    {
                        "Stock": symbol,
                        "Sector": sectors.get(symbol, "Other"),
                        "Price": price,
                        "% Chg": chg,
                        "RVOL": rvol,
                        "Breakout": breakout,
                        "Off 52W Low": (price - low_52w) / low_52w * 100.0,
                        "Spark": [round(float(c), 2) for c in closes.iloc[-30:]],
                    }
                )
            except Exception:
                failures.append(symbol)
    return pd.DataFrame(rows), failures


def fmt_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Stock | Price ₹ | % Chg | RVOL | Breakout | Off 52W Low |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for _, r in df.iterrows():
        dot = "🟢" if r["% Chg"] > 0 else ("🔴" if r["% Chg"] < 0 else "⚪")
        lines.append(
            f"| {r['Stock']} | {r['Price']:,.1f} | {dot} {r['% Chg']:+.2f}% "
            f"| **{r['RVOL']:.1f}x** | {r['Breakout'] or '—'} | {r['Off 52W Low']:.0f}% |"
        )
    return lines


def render(df, failures, universe_note, now_ist, elapsed_frac):
    in_session = elapsed_frac < 1.0
    status = (
        f"market OPEN — intraday scan, {elapsed_frac:.0%} of session elapsed "
        "(RVOL is pace-adjusted)"
        if in_session
        else "market CLOSED — full-day data from the last session"
    )
    a_list = df[(df["RVOL"] >= 1.5) & (df["Breakout"] != "")].sort_values("RVOL", ascending=False)
    breakouts = df[(df["Breakout"] != "") & ~df.index.isin(a_list.index)].sort_values(
        ["Breakout", "RVOL"], ascending=[False, False]
    )
    rvol_only = df[(df["RVOL"] >= 2.0) & (df["Breakout"] == "")].sort_values(
        "RVOL", ascending=False
    )

    lines = [
        "# 🇮🇳 Indian Stocks Dashboard — High RVOL & Breakouts",
        "",
        f"**Run time:** {now_ist.strftime('%A, %B %d, %Y %I:%M %p IST')}  ",
        f"**Market status:** {status}  ",
        f"**Universe:** {universe_note} · {len(df)} scanned  ",
        "**RVOL** = today's volume ÷ (20-day avg volume × fraction of session elapsed)",
        "",
        f"## 🚀 High RVOL + Breakout ({len(a_list)})",
        "",
    ]
    lines += fmt_table(a_list) if len(a_list) else ["_None right now._"]
    lines += ["", f"## 📈 Other Breakouts ({len(breakouts)})", ""]
    lines += fmt_table(breakouts.head(25)) if len(breakouts) else ["_None right now._"]
    if len(breakouts) > 25:
        lines.append(f"\n_...and {len(breakouts) - 25} more breakouts not shown._")
    lines += ["", f"## 🔊 High RVOL (≥2x), No Breakout Yet ({len(rvol_only)})", ""]
    lines += fmt_table(rvol_only.head(15)) if len(rvol_only) else ["_None right now._"]
    if failures:
        lines += ["", f"**No data ({len(failures)}):** " + ", ".join(sorted(failures))]
    return "\n".join(lines)


def main():
    now_ist = dt.datetime.now(IST)
    elapsed_frac = session_elapsed_fraction(now_ist)
    symbols, sectors, universe_note = load_universe()
    df, failures = scan(symbols, elapsed_frac, sectors)
    if df.empty:
        raise SystemExit("No data retrieved — check network access to Yahoo Finance.")
    md = render(df, failures, universe_note, now_ist, elapsed_frac)
    with open("india_dashboard.md", "w") as fh:
        fh.write(md + "\n")
    payload = {
        "run_time_ist": now_ist.strftime("%Y-%m-%d %I:%M %p IST"),
        "elapsed_frac": round(elapsed_frac, 3),
        "market_open": elapsed_frac < 1.0,
        "universe": universe_note,
        "scanned": len(df),
        "rows": df.sort_values("RVOL", ascending=False).to_dict(orient="records"),
        "failures": sorted(failures),
    }
    with open("india_dashboard.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(md)


if __name__ == "__main__":
    main()
