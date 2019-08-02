import pandas as pd

from backlight.labelizer.common import Label
from backlight.labelizer.common import TernaryDirection as TD
from backlight.signal.signal import Signal


def _r(a: int, b: int) -> float:
    return a / b if b != 0 else 0


def calculate_ternary_metrics(sig: Signal, lbl: Label) -> pd.DataFrame:
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

    cnt_u = uu + un + ud
    cnt_n = nu + nn + nd
    cnt_d = du + dn + dd

    hit_ratio = _r(uu + dd, uu + ud + du + dd)
    hit_ratio_u = _r(uu, uu + ud)
    hit_ratio_d = _r(dd, du + dd)

    hedge_ratio = _r(uu + un + dn + dd, uu + un + ud + du + dn + dd)
    hedge_ratio_u = _r(uu + un, uu + un + ud)
    hedge_ratio_d = _r(dn + dd, du + dn + dd)

    neutral_ratio = _r(cnt_n, total)
    coverage = _r(cnt_u + cnt_d, total)  # = 1.0 - neutral_ratio
    coverage_u = _r(cnt_u, total)
    coverage_d = _r(cnt_d, total)

    lbl = lbl.reindex(sig.index)

    pl = lbl[sig.pred != TD.N.value].label_diff.copy()
    pl.loc[sig.pred == TD.D.value] *= -1

    avg_pl = pl.mean()
    avg_win_pl = pl[pl > 0.0].mean()
    avg_lose_pl = pl[pl < 0.0].mean()
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
            ("hit_ratio_u", hit_ratio_u),
            ("hit_ratio_d", hit_ratio_d),
            ("hedge_ratio", hedge_ratio),
            ("hedge_ratio_u", hedge_ratio_u),
            ("hedge_ratio_d", hedge_ratio_d),
            ("neutral_ratio", neutral_ratio),
            ("coverage", coverage),
            ("coverage_u", coverage_u),
            ("coverage_d", coverage_d),
            ("avg_pl", avg_pl),
            ("avg_win_pl", avg_win_pl),
            ("avg_lose_pl", avg_lose_pl),
            ("risk_reward", -avg_win_pl / avg_lose_pl),
            ("total_pl", total_pl),
            ("sharpe", avg_pl / pl.std()),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
