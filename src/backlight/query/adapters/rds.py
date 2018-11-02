from backlight.query.adapter import DataSourceAdapter


class RDSAdapter(DataSourceAdapter):
    """Data source adapter for RDS"""

    def __init__(self, url):
        self._url = url

    def query(self, symbol, start_dt, end_dt):
        raise NotImplementedError()
