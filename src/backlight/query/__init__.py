import pandas as pd

from backlight.query.common import adapter_factory


def query(
    symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, url: str, **kwargs: str
) -> pd.DataFrame:
    adapter = adapter_factory(url, **kwargs)
    return adapter.query(symbol, start_dt, end_dt)
