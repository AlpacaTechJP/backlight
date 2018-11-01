import logging
from libalpaca.datasource.market import MarketData

from .adapter import DataSourceAdapter

logger = logging.getLogger(__name__)
DEFAULT_RETRIES = 100


class MarketstoreAdapter(DataSourceAdapter):
    """Data source adapter for Marketstore"""

    def __init__(self, url, mktdt=MarketData(timeframe="1Min", source="tick_mktsdb")):
        self._url = url
        self._mktdt = mktdt

    def query(self, symbol, start_dt, end_dt):
        ret = self._mktdt.query([symbol], start_dt=start_dt, end_dt=end_dt)[symbol]
        return ret.sort_index()
