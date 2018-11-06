from backlight.datasource.marketdata import MarketData

from backlight.query import query
from backlight.labelizer.common import LabelType
from backlight.labelizer.labelizer import Labelizer, Label


def load_label(symbol, url, start_dt=None, end_dt=None, mapping=None):
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(df=df, lbl_mapping=mapping)


def from_dataframe(df, col_mapping=None, lbl_mapping=None):
    """Create a Label instance out of a DataFrame object

    Args:
        df (pd.DataFrame):  DataFrame
        col_mapping (dict):  A dictionary to map columns.
        lbl_mapping (dict):  A dictionary to map labels.

        for Ternary classification:
            - "label_diff", "label", "neutral_range"

    Returns:
        Label
    """

    if "neutral_range" in df.columns:
        df = df.rename(
            columns={"label_up_neutral_down": "label", "label_up_down": "label"}
        )
        if mapping is not None:
            df.loc[:, "label"] = df.label.apply(lambda x: mapping[x])
        lbl = Label(df[["label_diff", "label", "neutral_range"]])
        lbl.label_type = LabelType.TERNARY
        return lbl

    raise ValueError("Unsupported label format")


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
