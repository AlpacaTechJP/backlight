from backlight.datasource.marketdata import MarketData

from backlight.query import query
from backlight.labelizer.labelizer import Labelizer
from backlight.labelizer.common import simple_label_factory


def load_label(symbol, url, start_dt=None, end_dt=None, factory=simple_label_factory, mapping=None):
    df = query(symbol, start_dt, end_dt, url)
    return factory(df=df, symbol=symbol, start_dt=start_dt, end_dt=end_dt, mapping=mapping)


def generate_labels(mkt, labelizer):
    """Generate label with specified marketdata and labelizer

    Args:
        mkt (MarketData): market data to be used
        labelizer (Labelizer): labelzier instance
    """
    assert isinstance(mkt, MarketData)
    assert issubclass(labelizer.__class__, Labelizer)

    lbl = labelizer.generate(mkt)

    return lbl
