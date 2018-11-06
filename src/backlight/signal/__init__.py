from backlight.query import query


def load_signal(symbol, url, start_dt, end_dt):
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(df, symbol, col_mapping=None)


def from_dataframe(df, symbol, col_mapping=None):
    """Create a MarketData instance out of a DataFrame object

    Args:
        df (pd.DataFrame):  DataFrame
        symbol (str): symbol to query
        col_mapping (dict):  A dictionary to map columns.

    Returns:
        Signal
    """

    df = df.copy()

    if col_mapping is not None:
        df = df.rename(columns=col_mapping)

    sig = None

    if ("up" in df.columns) and ("neutral" in df.columns) and ("down" in df.columns):
        from backlight.signal.signal import TernarySignal

        sig = TernarySignal(df)

    elif ("up" in df.columns) and ("down" in df.columns):
        from backlight.signal.signal import BinarySignal

        sig = BinarySignal(df)

    if sig is None:
        raise ValueError("Unsupported signal")

    sig.symbol = symbol
    sig.reset_cols()
    sig.reset_pred()

    return sig
