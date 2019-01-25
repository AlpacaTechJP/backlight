import pandas as pd

from backlight.labelizer.common import Label
from backlight.labelizer.common import TernaryDirection as TD
from backlight.signal.signal import Signal


def _r(a: int, b: int) -> float:
    return a / b if b != 0 else 0


def calc_ternary_metrics(sig: Signal, lbl: Label) -> pd.DataFrame:
    """Compute metrics on ternary signal.

    Args:
        sig :  Signal to evaluate.
        lbl : Correct answer.

    Returns:
        DataFrame of metrics.
    """

    sig = sig.dropna()
    lbl = lbl.dropna()

    uu = ((sig.pred == TD.U.value) & (lbl.label == TD.U.value)).sum()
    un = ((sig.pred == TD.U.value) & (lbl.label == TD.N.value)).sum()
    ud = ((sig.pred == TD.U.value) & (lbl.label == TD.D.value)).sum()
    nu = ((sig.pred == TD.N.value) & (lbl.label == TD.U.value)).sum()
    nn = ((sig.pred == TD.N.value) & (lbl.label == TD.N.value)).sum()
    nd = ((sig.pred == TD.N.value) & (lbl.label == TD.D.value)).sum()
    du = ((sig.pred == TD.D.value) & (lbl.label == TD.U.value)).sum()
    dn = ((sig.pred == TD.D.value) & (lbl.label == TD.N.value)).sum()
    dd = ((sig.pred == TD.D.value) & (lbl.label == TD.D.value)).sum()
    total = len(sig)

    hit_ratio = _r(uu + dd, uu + ud + du + dd)
    hedge_ratio = _r(uu + un + dn + dd, uu + un + ud + du + dn + dd)
    neutral_ratio = _r(nu + nn + nd, total)
    coverage = _r(uu + un + ud + du + dn + dd, total)  # = 1.0 - neutral_ratio

    lbl = lbl.reindex(sig.index)

    pl = lbl[sig.pred != TD.N.value].label_diff.copy()
    pl.loc[sig.pred == TD.D.value] *= -1

    avg_pl = pl.mean()
    total_pl = pl.sum()

    m = pd.DataFrame.from_records(
        [
            ("cnt_uu", uu),
            ("cnt_un", un),
            ("cnt_ud", ud),
            ("cnt_nu", nu),
            ("cnt_nn", nn),
            ("cnt_nd", nd),
            ("cnt_du", du),
            ("cnt_dn", dn),
            ("cnt_dd", dd),
            ("cnt_total", total),
            ("hit_ratio", hit_ratio),
            ("hedge_ratio", hedge_ratio),
            ("neutral_ratio", neutral_ratio),
            ("coverage", coverage),
            ("avg_pl", avg_pl),
            ("total_pl", total_pl),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
