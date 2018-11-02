from backlight.query.common import adapter_factory


def query(symbol, start_dt, end_dt, url, **kwargs):
    adapter = adapter_factory(url, **kwargs)
    return adapter.query(symbol, start_dt, end_dt)
