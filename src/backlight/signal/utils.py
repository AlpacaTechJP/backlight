import pandas as pd
from typing import Optional

from backlight.query import query
from backlight.signal.signal import BinarySignal, Signal, TernarySignal
from backlight.asset.currency import Currency


def load_signal(
    symbol: str,
    url: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    currency_unit: Currency,
    threshold: float = 0.0,
) -> Signal:
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(df, symbol, currency_unit, col_mapping=None)


def from_dataframe(
    df: pd.DataFrame,
    symbol: str,
    currency_unit: Currency,
    col_mapping: Optional[dict] = None,
) -> Signal:
    """Create a MarketData instance out of a DataFrame object

    Args:
        df (pd.DataFrame):  DataFrame
        symbol (str): symbol to query
        col_mapping (dict):  A dictionary to map columns.
        currency_unit: currency unit of the dataframe

    Returns:
        Signal
    """

    df = df.copy()

    if col_mapping is not None:
        df = df.rename(columns=col_mapping)

    sig = None

    if ("up" in df.columns) and ("neutral" in df.columns) and ("down" in df.columns):
        sig = TernarySignal(df)

    elif ("up" in df.columns) and ("down" in df.columns):
        sig = BinarySignal(df)

    if sig is None:
        raise ValueError("Unsupported signal")

    sig.symbol = symbol
    sig.currency_unit = currency_unit
    sig.reset_cols()
    sig.reset_pred()

    return sig
