import logging
import os
import pandas as pd
import urllib.parse

from libalpaca.marketstore import client

from backlight.query.adapter import DataSourceAdapter

logger = logging.getLogger(__name__)
DEFAULT_RETRIES = 100


class MarketstoreAdapter(DataSourceAdapter):
    """Data source adapter for Marketstore"""

    def __init__(self, url: str) -> None:
        if url is None:
            hostname = os.environ.get("MARKETSTORE_HOST")
            port = os.environ.get("MARKETSTORE_PORT")
        else:
            o = urllib.parse.urlparse(url)
            assert o.scheme == "mktsdb"
            hostname = o.hostname
            port = o.port
        self._cli = client.Client(host=hostname, port=port)

    def query(
        self,
        symbol: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:

        ret = self._cli.query(
            symbol=symbol, timeframe=timeframe, start_dt=start_dt, end_dt=end_dt
        )
        return ret.sort_index()
