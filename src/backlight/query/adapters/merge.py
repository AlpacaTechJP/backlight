import pandas as pd
from typing import Iterable

from backlight.query.adapter import DataSourceAdapter
from backlight.query.common import adapter_factory


class MergeAdapter(DataSourceAdapter):
    """Data source adapter for multiple data sources"""

    def __init__(self, urls: Iterable[str]) -> None:
        """Initializer.

        Args:
            urls    : Urls for multiple data sources. `urls` should
                      be castable into :class:`tuple`. Each element
                      should be implemented in
                      :class:`~backlight.query.common.adapter_factory`.
        """
        self._urls = tuple(urls)

    def query(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        """Query pandas dataframe.

        See also :class:`backlight.query.adapter`.
        """
        adapters = [adapter_factory(url) for url in self._urls]
        dfs = [a.query(symbol, start_dt, end_dt) for a in adapters]
        df = pd.concat(dfs, axis=0).sort_index()
        df = df[(start_dt <= df.index) & (df.index <= end_dt)]
        return df
