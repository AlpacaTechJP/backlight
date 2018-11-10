import pandas as pd
from backlight.query.adapter import DataSourceAdapter


class RDSAdapter(DataSourceAdapter):
    """Data source adapter for RDS"""

    def __init__(self, url: str) -> None:
        self._url = url

    def query(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        raise NotImplementedError()
