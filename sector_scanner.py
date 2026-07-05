"""
Equal-Weight Sector / Industry Group Performance Table
=======================================================
Fetches market data via yfinance and builds a table showing, for each
equal-weight sector / industry-group ETF:

  * 52-Weeks  — how far the current price sits ABOVE its 52-week low
                ((current - 52w_low) / 52w_low * 100), a strength-from-the-low measure
  * % Daily Chg — most recent trading day's percentage change

Sorted by 52-week strength, strongest first. Output is a Markdown table
written to sector_table.md and printed to stdout.

Usage:  pip install yfinance pandas && python sector_scanner.py
"""

import datetime as dt

import pandas as pd
import yfinance as yf

# (Equal-weight industry group name, primary ticker)
GROUPS = [
    ("Semiconductor", "XSD"),
    ("Technology", "RSPT"),
    ("Real Estate", "RSPR"),  # equal-weight preferred; falls back to XLRE
    ("QQQ", "QQQ"),
    ("China", "GXC"),
    ("Health Care Services", "XHS"),
    ("Biotech", "XBI"),
    ("Homebuilders", "XHB"),
    ("Industrial", "RSPN"),
    ("Insurance", "KIE"),
    ("Healthcare Equipments", "XHE"),
    ("Financial", "RSPF"),
    ("Regional Banks", "KRE"),
    ("S&P 500", "RSP"),
    ("Transportation", "XTN"),
    ("Aerospace & Defence", "XAR"),
    ("Materials", "RSPM"),
    ("Consumer Discretionary", "RSPD"),
    ("US Banks", "KBE"),
    ("IWM (Russell 2000)", "IWM"),
    ("Communication", "RSPC"),
    ("Utilities", "RSPU"),
    ("Consumer Staples", "RSPS"),
    ("Energy", "RSPG"),
    ("Oil & Gas Exploration & Production", "XOP"),
    ("Pharmaceuticals", "XPH"),
    ("Metal & Minings", "XME"),
    ("Natural Resources", "GNR"),
    ("Healthcare", "RSPH"),
    ("Retail", "XRT"),
    ("Software & Services", "XSW"),
    ("Oil & Gas Equipment & Services", "XES"),
    ("Telecom", "XTL"),
]

# If a primary ticker returns no data, try these instead (in order).
FALLBACKS = {
    "RSPR": ["XLRE"],
    "RSPT": ["XLK"],
    "RSPF": ["XLF"],
    "RSPN": ["XLI"],
    "RSPM": ["XLB"],
    "RSPD": ["XLY"],
    "RSPC": ["XLC"],
    "RSPU": ["XLU"],
    "RSPS": ["XLP"],
    "RSPG": ["XLE"],
    "RSPH": ["XLV"],
}


def fetch_row(name: str, ticker: str):
    """Return a result dict for one ticker, or None if no usable data."""
    hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
    if hist is None or hist.empty or len(hist) < 2:
        return None
    closes = hist["Close"].dropna()
    lows = hist["Low"].dropna()
    current = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    low_52w = float(lows.min())
    return {
        "Equal Weight Industry Group": name,
        "Ticker": ticker,
        "Price": current,
        "52-Weeks": (current - low_52w) / low_52w * 100.0,
        "% Daily Chg": (current - prev) / prev * 100.0,
        "Last Bar": closes.index[-1].strftime("%Y-%m-%d"),
    }


def build_table():
    rows, failures = [], []
    for name, ticker in GROUPS:
        row = None
        for t in [ticker] + FALLBACKS.get(ticker, []):
            try:
                row = fetch_row(name, t)
            except Exception as exc:  # network hiccup, bad symbol, etc.
                print(f"  ! {t}: {exc}")
                row = None
            if row:
                if t != ticker:
                    row["Ticker"] = f"{t}*"
                    row["Fallback"] = f"{t} used as fallback for {ticker}"
                break
        if row:
            rows.append(row)
        else:
            failures.append((name, ticker))
    return rows, failures


def render_markdown(rows, failures) -> str:
    df = (
        pd.DataFrame(rows)
        .sort_values("52-Weeks", ascending=False)
        .reset_index(drop=True)
    )
    today = dt.date.today()
    last_bar = max(r["Last Bar"] for r in rows)
    is_live = last_bar == today.strftime("%Y-%m-%d")
    status = (
        "market session in progress — intraday prices"
        if is_live
        else f"market CLOSED — data as of last close ({last_bar})"
    )

    lines = [
        "# Equal-Weight Sector / Industry Group Performance",
        "",
        f"**Report date:** {today.strftime('%A, %B %d, %Y')}  ",
        f"**Market status:** {status}  ",
        "**52-Weeks** = % the current price sits above its 52-week low "
        "(strength from the low)",
        "",
        "| # | Equal Weight Industry Group | Ticker | 52-Weeks | % Daily Chg |",
        "|---:|---|---|---:|---:|",
    ]
    for i, r in df.iterrows():
        chg = r["% Daily Chg"]
        dot = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")
        lines.append(
            f"| {i + 1} | {r['Equal Weight Industry Group']} | {r['Ticker']} "
            f"| {r['52-Weeks']:.0f}% | {dot} {chg:+.2f}% |"
        )
    notes = [r["Fallback"] for r in rows if r.get("Fallback")]
    if notes:
        lines += ["", "\\* " + "; ".join(notes)]
    if failures:
        lines += ["", "**Failed tickers (no data):** "
                  + ", ".join(f"{n} ({t})" for n, t in failures)]
    return "\n".join(lines)


def main():
    rows, failures = build_table()
    if not rows:
        raise SystemExit(
            "No data retrieved for any ticker — check network access to "
            "Yahoo Finance and try again."
        )
    md = render_markdown(rows, failures)
    with open("sector_table.md", "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
