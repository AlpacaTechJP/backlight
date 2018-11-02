import numpy as np
import pandas as pd
from backlight.strategies.common import Action

from .binary import BinaryMetrics
from .common import metric_property, sync_index_and_get_list


def calc_binary_trade_performance(sig, lbl, trades):
    real_arr, pred_arr = sync_index_and_get_list(lbl, sig)
    return simple_binary_metrics_extend_factory(real_arr, pred_arr, trades)


def simple_binary_metrics_extend_factory(real_arr, pred_arr, trades):
    """Simple Factory method: Initialize correct object to compute extra
    metrics like PL according to the information inside trades. You can
    customized your logic here.
    """
    if "amount" in trades.keys():
        return AmountBasedPL(real_arr, pred_arr, trades)
    else:
        raise NotImplementedError


class BinaryMetricsExtend:
    def __init__(self, y_true: list, y_pred: list, trades):
        """
        Args:
            y_true     (list): Actual labels
            y_pred     (list): Predicted labels
            trades     (backlight.trades.Trades): contains price and amount
        """
        self._base_metrics = BinaryMetrics(y_true, y_pred)
        self._trades = trades
        self.metric_names = [
            attr
            for attr in dir(self)
            if not attr.startswith("__")
            and not attr.startswith("_")
            and attr not in ["get", "to_frame"]
        ]
        self._precompute()

    def _precompute():
        raise NotImplementedError


class AmountBasedPL(BinaryMetricsExtend):
    def _precompute(self):
        """In here, we assume the signal is today's n days later prediction,
        so we use tomrrow - today as diff and compute the PL. However, in real
        senario, it is impossible to treat with today's price. In that case,
        you can shift the signal to become the yesterday's prediction and trade
        on today's price. Actually, in some examples, all signal are
        yesterday's prediction, so can directly use this function.

        Here:
          today         tomrrow
            |--------------|
          prediction---------------------------------->| direction
            |----diff------|

        shifted case:
          yesterday         today        tomrrow
               |--------------|-------------|
            prediction--------------------------------->| direction
                              |----diff-----|

        """
        self._prices = self._trades[self._trades.target_column_name]
        self._initial_price = self._prices.values[0]
        self._final_price = self._prices.values[-1]
        self._diff1 = (self._prices.shift(-1) - self._prices.shift(0)).values

    @metric_property
    def total_pl(self):
        return sum(self._trades["amount"][:-1] * self._diff1[:-1])

    @metric_property
    def total_pl_percentage(self):
        return self.total_pl / self._initial_price

    @metric_property
    def accumulate_pl_percentage(self):
        rates = (self._trades["amount"][:-1] * self._diff1[:-1]) / self._prices[:-1]
        s = 1
        for i in range(0, len(rates)):
            s = s * (1 + rates[i])
        return s - 1

    @metric_property
    def baseline_pl(self):
        return self._final_price - self._initial_price

    @metric_property
    def baseline_pl_percentage(self):
        return self.baseline_pl / self._initial_price

    @metric_property
    def average_win(self):
        win_pl = []
        for each_pl in self._trades["amount"][:-1] * self._diff1[:-1]:
            if each_pl > 0.0:
                win_pl.append(each_pl)
        return sum(win_pl) / len(win_pl)

    @metric_property
    def average_loss(self):
        loss_pl = []
        for each_pl in self._trades["amount"][:-1] * self._diff1[:-1]:
            if each_pl < 0.0:
                loss_pl.append(each_pl)
        return sum(loss_pl) / len(loss_pl)

    @metric_property
    def average_pl(self):
        action_counts = 0
        action_pl = 0
        # Donothing can't not be count in average_pl
        for i, v in enumerate(self._trades[:-1]["amount"].iteritems()):
            if v[1] != Action.Donothing.value:
                action_counts += 1
                action_pl += self._trades["amount"][:-1].values[i] * self._diff1[:-1][i]
        return action_pl / action_counts

    def get(self, metric_names: list = []) -> dict:
        if metric_names is None or len(metric_names) == 0:
            metric_names = self.metric_names.copy()
            metric_names.extend(self._base_metrics.metric_names)
        ret = {}
        for metric_name in metric_names:
            if hasattr(self, metric_name):
                ret[metric_name] = getattr(self, metric_name)
            else:
                ret[metric_name] = getattr(self._base_metrics, metric_name)
        return ret

    def to_frame(self):
        dic = {}
        for metric_name in self.metric_names:
            dic[metric_name] = getattr(self, metric_name)
        for metric_name in self._base_metrics.metric_names:
            dic[metric_name] = getattr(self._base_metrics, metric_name)
        return pd.DataFrame(dic, index=[0]).astype(np.float64)
