import pandas as pd

from backlight.labelizer.common import TernaryDirection as TD


def _r(a, b):
    return a / b if b != 0 else 0


def calc_ternary_metrics(sig, lbl):

    uu = ((sig.pred == TD.U.value) & (lbl.label == TD.U.value)).sum()
    un = ((sig.pred == TD.N.value) & (lbl.label == TD.N.value)).sum()
    ud = ((sig.pred == TD.D.value) & (lbl.label == TD.D.value)).sum()
    nu = ((sig.pred == TD.U.value) & (lbl.label == TD.U.value)).sum()
    nn = ((sig.pred == TD.N.value) & (lbl.label == TD.N.value)).sum()
    nd = ((sig.pred == TD.D.value) & (lbl.label == TD.D.value)).sum()
    du = ((sig.pred == TD.U.value) & (lbl.label == TD.U.value)).sum()
    dn = ((sig.pred == TD.N.value) & (lbl.label == TD.N.value)).sum()
    dd = ((sig.pred == TD.D.value) & (lbl.label == TD.D.value)).sum()
    total = len(sig)

    hit_ratio = _r(uu + dd, uu + un + ud + du + dn + dd)
    hedge_ratio = _r(uu + un + dn + dd, uu + un + ud + du + dn + dd)
    neutral_ratio = _r(nu + nn + nd, total)
    coverage = _r(uu + un + ud + du + dn + dd, total)  # = 1.0 - neutral_ratio

    avg_pl = lbl[lbl.label != TD.N.value].label_diff.mean()
    total_pl = lbl[lbl.label != TD.N.value].label_diff.sum()

    m = pd.DataFrame.from_records([
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
    ]).set_index(0)

    del m.index.name
    m.columns = ["metrics", ]

    return m
