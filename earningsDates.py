import yfinance as yf
import pandas as pd
import numpy as np


# ======================================================
# 1. Fetch earnings + short-dated option chains
# ======================================================

def load_option_data(ticker, n_exp=4):
    """
    Downloads:
      - earnings dates (best effort)
      - first n_exp option expirations (short maturities)
      - all calls for those expirations

    Returns:
      df        : concatenated calls across expirations
      earnings  : DataFrame of earnings dates (may be None)
      exps      : list of expiration strings used
    """
    tk = yf.Ticker(ticker)

    # Earnings
    try:
        earnings = tk.get_earnings_dates(limit=24)
        if earnings is not None and not earnings.empty:
            start_date = pd.Timestamp('2023-01-01').tz_localize('UTC')
            end_date = pd.Timestamp.now().tz_localize('UTC')
            earnings = earnings[(earnings.index >= start_date) & (earnings.index <= end_date)]
    except Exception as e:
        print(f"[WARN] Could not fetch earnings dates for {ticker}: {e}")
        earnings = None

    # Option expirations
    exps_all = tk.options
    if not exps_all:
        raise RuntimeError(f"No option expirations found for {ticker}")
    exps = exps_all[:n_exp]  # first n_exp maturities

    chains = []
    for exp in exps:
        try:
            raw = tk.option_chain(exp)
        except Exception as e:
            print(f"[WARN] Skipping expiration {exp} due to error: {e}")
            continue

        calls = raw.calls.copy()
        calls["expiration"] = exp
        chains.append(calls)

    if not chains:
        raise RuntimeError(f"No valid option chains downloaded for {ticker}")

    df = pd.concat(chains).reset_index(drop=True)
    return df, earnings, exps


# ======================================================
# 2. Clean option quotes
# ======================================================

def clean_option_quotes(df):
    """
    Removes bad/missing quotes:
      - zero volume
      - zero or missing IV
      - extreme bid/ask spreads
    """
    df = df.copy()

    # Remove no-volume strikes (if volume column exists)
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    # Remove missing IV
    df = df[~df["impliedVolatility"].isna()]

    # Remove IV ≤ 0
    df = df[df["impliedVolatility"] > 0]

    # Spread filter: ask/bid not crazy
    df = df[df["bid"] > 0]
    df = df[df["ask"] > df["bid"]]
    df = df[(df["ask"] - df["bid"]) / df["bid"] < 0.5]  # 50% spread cap

    return df.reset_index(drop=True)


# ======================================================
# 3. Extract ATM IV for each expiration
# ======================================================

def extract_atm(df, spot):
    """
    Picks ATM strike for each expiration (closest strike to spot).
    Returns one row per expiration.
    """
    df = df.copy()
    df["atm_diff"] = (df["strike"] - spot).abs()

    # Pick the ATM row within each expiration
    atm = df.loc[df.groupby("expiration")["atm_diff"].idxmin()]

    return atm.reset_index(drop=True)


# ======================================================
# 4. Main preprocessing pipeline
# ======================================================

def preprocess(ticker, n_exp=4):
    """
    Full pipeline:
      - loads option chains
      - cleans quotes
      - extracts ATM IV for chosen expirations

    Returns:
      df_clean : cleaned option chain DataFrame
      atm      : ATM row per expiration
      earnings : earnings date DataFrame or None
      exps     : list of expirations used
    """
    print(f"Fetching data for {ticker}...")
    df, earnings, exps = load_option_data(ticker, n_exp)

    # fetch spot price (last close)
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"No price history available for {ticker}")
    spot = hist["Close"].iloc[-1]
    print(f"Spot price = {spot:.2f}")

    print("Cleaning quotes...")
    df_clean = clean_option_quotes(df)

    print("Extracting ATM IV...")
    atm = extract_atm(df_clean, spot)

    print("Done.")
    return df_clean, atm, earnings, exps


# ======================================================
# 5. Helper: next earnings date
# ======================================================

def get_next_earnings_date(earnings_df):
    """
    Returns the date of the next earnings event (as a date),
    or None if not available.
    """
    if earnings_df is None or earnings_df.empty:
        return None
    # earnings_df index is a DatetimeIndex of earnings dates
    return earnings_df.index[0].date()

if __name__ == "__main__":
    tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet Class A
    "GOOG",  # Alphabet Class C
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "META",  # Meta Platforms (Facebook)
    "NFLX",  # Netflix
    "AMD",   # Advanced Micro Devices
    "INTC",  # Intel
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "PFE",   # Pfizer
    "BAC",   # Bank of America
    "WMT",   # Walmart
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "JNJ",   # Johnson & Johnson
    "DIS",   # Disney
    "CRM",   # Salesforce
    "T",     # AT&T
    "VZ",    # Verizon
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "PG",    # Procter & Gamble
    "TSM",   # Taiwan Semiconductor
    "ADBE",  # Adobe
    "NFLX",  # (already included — duplicate, but shows frequent liquidity)
    # ETFs
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq-100 ETF
    "IWM",   # Russell 2000 ETF
    "EEM",   # Emerging Markets ETF
    "GDX",   # Gold Miners ETF
    "XLF",   # Financials Select Sector SPDR
    "GLD",   # SPDR Gold Shares
    "HYG",   # iShares High-Yield Corp Bond ETF
    "TLT",   # iShares 20+ Year Treasury Bond ETF
    "SLV",   # iShares Silver Trust
    "SOXL",  # Direxion Semiconductor Bull 3x ETF
    "TQQQ",  # ProShares UltraPro QQQ 3x ETF
    "SQQQ",  # ProShares UltraPro Short QQQ 3x ETF
    "SPXL",  # Direxion S&P 500 Bull 3x Shares ETF
    "XLE",   # Energy Select Sector SPDR ETF
    "XLK",   # Technology Select Sector SPDR ETF
    "XLY"    # Consumer Discretionary Select Sector SPDR ETF
]

    for ticker in tickers:
        df_clean, atm_df, earnings, exps = preprocess(ticker, n_exp=4)
        print("\nCleaned data sample:")
        print(df_clean.head())

        print("\nATM IV per expiration:")
        print(atm_df[["expiration", "strike", "impliedVolatility"]])

        print("\nEarnings Dates:")
        print(earnings)

        next_evt = get_next_earnings_date(earnings)
        print("\nNext earnings date:", next_evt)
    

    
    
