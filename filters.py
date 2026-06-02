"""
filters.py
All filter / data-processing functions for the Ethereum dashboard.
Auto-generates data if ethereum.csv is missing (works on Streamlit Cloud).
"""

import pandas as pd
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "ethereum.csv")


# ─────────────────────────────────────────────
# Data generation (fallback if CSV missing)
# ─────────────────────────────────────────────

def _generate_and_save() -> pd.DataFrame:
    """Generate realistic synthetic ETH data and save to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)

    dates = pd.date_range("2015-08-07", "2025-05-01", freq="D")
    n = len(dates)

    log_returns = np.random.normal(0.001, 0.04, n)
    price = [0.70]
    for r in log_returns[1:]:
        price.append(max(price[-1] * np.exp(r), 0.01))
    price = np.array(price)

    scale = np.ones(n)
    def _ramp(start, end, v0, v1):
        m = (dates >= start) & (dates <= end)
        scale[m] = np.linspace(v0, v1, m.sum())

    _ramp("2016-01-01", "2017-01-01",   10,   12)
    _ramp("2017-01-01", "2018-01-15",   50, 1400)
    _ramp("2018-01-15", "2018-12-31", 1400,  100)
    _ramp("2019-01-01", "2020-09-30",  100,  350)
    _ramp("2020-10-01", "2021-11-09",  350, 4800)
    _ramp("2021-11-09", "2022-12-31", 4800, 1200)
    _ramp("2023-01-01", "2024-03-01", 1200, 3900)
    _ramp("2024-03-01", "2025-05-01", 3900, 3100)

    price = scale * (price / price.max())

    market_cap   = price * 120_000_000
    total_volume = market_cap * np.random.uniform(0.03, 0.15, n)

    df = pd.DataFrame({
        "date":         dates,
        "price":        price,
        "market_cap":   market_cap,
        "total_volume": total_volume,
    })

    df["price_change_pct"] = df["price"].pct_change() * 100
    df["market_cap_b"]     = df["market_cap"] / 1e9
    df["volume_b"]         = df["total_volume"] / 1e9
    df["year"]             = df["date"].dt.year
    df["month"]            = df["date"].dt.month
    df["month_name"]       = df["date"].dt.strftime("%b")
    df["quarter"]          = df["date"].dt.quarter
    df["day_of_week"]      = df["date"].dt.day_name()
    df["ma_7"]             = df["price"].rolling(7).mean().fillna(df["price"])
    df["ma_30"]            = df["price"].rolling(30).mean().fillna(df["price"])
    df["ma_90"]            = df["price"].rolling(90).mean().fillna(df["price"])
    df["volatility_30d"]   = df["price_change_pct"].rolling(30).std().fillna(0)

    df.to_csv(DATA_PATH, index=False)
    return df


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load Ethereum CSV; auto-generate it if the file does not exist."""
    if not os.path.exists(DATA_PATH):
        # Try live CoinGecko first, fall back to synthetic
        try:
            import requests
            url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            params = {"vs_currency": "usd", "days": "max", "interval": "daily"}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            raw = r.json()

            df_p = pd.DataFrame(raw["prices"],        columns=["ts", "price"])
            df_m = pd.DataFrame(raw["market_caps"],   columns=["ts", "market_cap"])
            df_v = pd.DataFrame(raw["total_volumes"], columns=["ts", "total_volume"])
            df = df_p.merge(df_m, on="ts").merge(df_v, on="ts")
            df["date"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.drop(columns=["ts"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

            df["price_change_pct"] = df["price"].pct_change() * 100
            df["market_cap_b"]     = df["market_cap"] / 1e9
            df["volume_b"]         = df["total_volume"] / 1e9
            df["year"]             = df["date"].dt.year
            df["month"]            = df["date"].dt.month
            df["month_name"]       = df["date"].dt.strftime("%b")
            df["quarter"]          = df["date"].dt.quarter
            df["day_of_week"]      = df["date"].dt.day_name()
            df["ma_7"]             = df["price"].rolling(7).mean().fillna(df["price"])
            df["ma_30"]            = df["price"].rolling(30).mean().fillna(df["price"])
            df["ma_90"]            = df["price"].rolling(90).mean().fillna(df["price"])
            df["volatility_30d"]   = df["price_change_pct"].rolling(30).std().fillna(0)

            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
        except Exception:
            df = _generate_and_save()
    else:
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])

    return df


# ─────────────────────────────────────────────
# Filter helpers
# ─────────────────────────────────────────────

def apply_date_filter(df, start_date, end_date):
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    return df[mask].copy()

def apply_year_filter(df, years):
    if not years:
        return df
    return df[df["year"].isin(years)].copy()

def apply_price_range_filter(df, min_price, max_price):
    return df[(df["price"] >= min_price) & (df["price"] <= max_price)].copy()

def apply_volume_range_filter(df, min_vol, max_vol):
    return df[(df["volume_b"] >= min_vol) & (df["volume_b"] <= max_vol)].copy()

def apply_quarter_filter(df, quarters):
    if not quarters:
        return df
    return df[df["quarter"].isin(quarters)].copy()

def apply_text_search(df, keyword):
    if not keyword.strip():
        return df
    mask = df["date"].astype(str).str.contains(keyword.strip(), case=False, na=False)
    return df[mask].copy()

def reset_filters(df_original):
    return df_original.copy()

# ─────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────

def get_kpis(df):
    if df.empty:
        return {k: 0 for k in ["total_records","avg_price","max_price","min_price",
                                "avg_volume_b","avg_market_cap_b","max_market_cap_b",
                                "best_day_return","worst_day_return","avg_volatility"]}
    return {
        "total_records":    len(df),
        "avg_price":        df["price"].mean(),
        "max_price":        df["price"].max(),
        "min_price":        df["price"].min(),
        "avg_volume_b":     df["volume_b"].mean(),
        "avg_market_cap_b": df["market_cap_b"].mean(),
        "max_market_cap_b": df["market_cap_b"].max(),
        "best_day_return":  df["price_change_pct"].max(),
        "worst_day_return": df["price_change_pct"].min(),
        "avg_volatility":   df["volatility_30d"].mean(),
    }
