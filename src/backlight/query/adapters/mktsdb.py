import logging
import pandas as pd
from libalpaca.datasource.market import MarketData

from backlight.query.adapter import DataSourceAdapter

logger = logging.getLogger(__name__)
DEFAULT_RETRIES = 100


class MarketstoreAdapter(DataSourceAdapter):
    """Data source adapter for Marketstore"""

    def __init__(
        self,
        url: str,
        mktdt: MarketData = MarketData(timeframe="1Min", source="tick_mktsdb"),
    ) -> None:
        self._url = url
        self._mktdt = mktdt

    def query(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        ret = self._mktdt.query([symbol], start_dt=start_dt, end_dt=end_dt)[symbol]
        return ret.sort_index()
