"""Technical indicator calculations. All functions take pandas Series/DataFrame and return pandas Series."""
import numpy as np
import pandas as pd


def compute_ma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period).mean()


def compute_ema(close: pd.Series, period: int = 20) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_kd(df: pd.DataFrame, k_period: int = 9, d_period: int = 3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=d_period - 1, min_periods=1).mean()
    d = k.ewm(com=d_period - 1, min_periods=1).mean()
    return k, d


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
    avg = volume.rolling(window=period).mean()
    return volume / avg.replace(0, np.nan)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume * direction).cumsum()


def compute_all(df: pd.DataFrame) -> dict:
    close = df["Close"]
    macd, macd_signal, macd_hist = compute_macd(close)
    k, d = compute_kd(df)
    bb_upper, bb_mid, bb_lower = compute_bollinger(close)
    return {
        "ma_5": compute_ma(close, 5), "ma_10": compute_ma(close, 10),
        "ma_20": compute_ma(close, 20), "ma_60": compute_ma(close, 60),
        "ma_120": compute_ma(close, 120),
        "ema_12": compute_ema(close, 12), "ema_26": compute_ema(close, 26),
        "rsi": compute_rsi(close),
        "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "k": k, "d": d,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "atr": compute_atr(df),
        "vol_ratio_5d": compute_volume_ratio(df["Volume"], 5),
        "vol_ratio_20d": compute_volume_ratio(df["Volume"], 20),
        "obv": compute_obv(close, df["Volume"]),
    }
