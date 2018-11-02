from backlight.datasource.marketdata import MarketData

from backlight.labelizer.labelizer import Labelizer
from backlight.labelizer.ternary.fixed_neutral import FixedNeutralLabelizer  # noqa
from backlight.labelizer.ternary.dynamic_neutral import DynamicNeutralLabelizer  # noqa


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
