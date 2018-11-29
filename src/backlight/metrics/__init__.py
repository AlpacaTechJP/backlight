import pandas as pd

from typing import List

from backlight.datasource import MarketData
from backlight.labelizer.common import LabelType, Label
from backlight.metrics.ternary import calc_ternary_metrics
from backlight.signal import Signal
from backlight.trades import Trade, count, flatten, evaluate_pl


def calc_metrics(sig: Signal, lbl: Label, dropna: bool = True) -> pd.DataFrame:
    if lbl.label_type == LabelType.TERNARY:
        return calc_ternary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError


def calc_trade_peformance(trades: List[Trade], mkt: MarketData) -> pd.DataFrame:
    total_count, win_count, lose_count = count(trades, mkt)
    trade = flatten(trades)
    pl = evaluate_pl(trade, mkt)

    m = pd.DataFrame.from_records(
        [
            ("cnt_total", total_count),
            ("cnt_win", win_count),
            ("cnt_lose", lose_count),
            ("win_ratio", win_count / total_count),
            ("lose_ratio", lose_count / total_count),
            ("pl", pl),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
