from .adapter import DataSourceAdapter


class KDBAdapter(DataSourceAdapter):
    """Data source adapter for KDB"""

    def __init__(self, url):
        self._url = url

    def query(self, symbol, start_dt, end_dt):
        pass
