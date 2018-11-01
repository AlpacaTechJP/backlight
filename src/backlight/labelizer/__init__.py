from pome.datasource.marketdata import MarketData

from .labelizer import Labelizer
from .ternary.fixed_neutral import FixedNeutralLabelizer  # noqa


def generate_labels(mkt, labelizer):

    assert isinstance(mkt, MarketData)
    assert issubclass(labelizer.__class__, Labelizer)

    lbl = labelizer.generate(mkt)

    return lbl
