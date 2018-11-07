import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.labelizer.common import LabelType
from backlight.labelizer.labelizer import Labelizer, Label
from backlight.query import query


def load_label(
    symbol: str,
    url: str,
    start_dt: pd.Timestamp = None,
    end_dt: pd.Timestamp = None,
    mapping: dict = None,
) -> Label:
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(df=df, lbl_mapping=mapping)


def from_dataframe(
    df: pd.DataFrame, col_mapping: dict = None, lbl_mapping: dict = None
) -> Label:
    """Create a Label instance out of a DataFrame object

    Args:
        df :  DataFrame
        col_mapping :  A dictionary to map columns.
        lbl_mapping :  A dictionary to map labels.

        for Ternary classification:
            - "label_diff", "label", "neutral_range"

    Returns:
        Label
    """

    if "neutral_range" in df.columns:
        df = df.rename(
            columns={"label_up_neutral_down": "label", "label_up_down": "label"}
        )
        if lbl_mapping is not None:
            df.loc[:, "label"] = df.label.apply(lambda x: lbl_mapping[x])
        lbl = Label(df[["label_diff", "label", "neutral_range"]])
        lbl.label_type = LabelType.TERNARY
        return lbl

    raise ValueError("Unsupported label format")


def generate_labels(mkt: MarketData, labelizer: Labelizer) -> Label:
    """Generate label with specified marketdata and labelizer

    Args:
        mkt : market data to be used
        labelizer : labelzier instance
    """
    lbl = labelizer.generate(mkt)

    return lbl
