import numpy as np
import pandas as pd
from typing import Callable
from collections import namedtuple
from functools import lru_cache
from backlight.labelizer.common import TernaryDirection


class InvalidTrade(Exception):
    pass


Trade = namedtuple("Trade", ["index", "value"])


def pick_random_tick(idx, x):
    return x.values[np.random.randint(len(x))]


def pick_exp_decay(idx, x, half_life=500):
    """Exponential decay, with half life in miliseconds"""
    delta = pd.Timedelta("%dms" % int(np.random.exponential(scale=half_life)))
    ret = x[idx + delta :]
    if len(ret) == 0:
        raise InvalidTrade("No values after index: {} + {}".format(idx, delta))
    return Trade(ret.index[0], ret.values[0])


class TickDataSimulator:
    def __init__(
        self,
        market_data: pd.DataFrame,
        transaction_cost: float = 0.0,
        disable_slippage: bool = False,
        ask_picker: Callable = lambda i, x: Trade(x.index[-1], x.values[-1]),
        bid_picker: Callable = lambda i, x: Trade(x.index[-1], x.values[-1]),
        max_ffill: str = "1H",
    ):
        """Trading simulation on tick data.

        Simulation of market using level 1 bid/ask tick data.
        Volume is not taken into account, orders are completely filled.

        Args:
            market_data          : Function to query data from marketstore,
                                   default query_usdjpy_ebs.
            transaction_cost     : Cost of each transaction (x2 for entry/exit pair),
                                   default 0.002.
            disable_slippage     : Disable calculation of slippage and use mid price
                                   instead, default False.
            ask_picker           : Function to choose which tick to use for computation
                                   on ask side. Take the index of the trade and
                                   a dataframe as input, return one value.
                                   Default to taking first tick (near immediate).
            bid_picker           : Reciprocal of ask_picker for bid.
            max_ffill            : How far to do the forward filling for tick prices,
                                   default 1H.
        """
        self.max_ffill = pd.Timedelta(max_ffill)
        self.transaction_cost = transaction_cost
        self.disable_slippage = disable_slippage

        self.market_data = market_data

        self.ask_picker = ask_picker
        self.bid_picker = bid_picker

    def _ticks_around(self, index):
        p = self.market_data.loc[index - self.max_ffill : index]
        if len(p) == 0:
            raise InvalidTrade("No market data after entry: {}".format(index))
        return p

    def _get_pair_no_slippage(self, entry, exit):
        # Compute mid price
        a = self._ticks_around(entry).copy()
        b = self._ticks_around(exit).copy()

        a["mid"] = (a["l1-askprice"] + a["l1-bidprice"]) / 2
        b["mid"] = (b["l1-askprice"] + b["l1-bidprice"]) / 2

        t_entry = self.ask_picker(entry, a["mid"])
        t_exit = self.ask_picker(exit, b["mid"])
        return t_entry, t_exit

    def _get_pair_long(self, entry, exit):
        if self.disable_slippage:
            return self._get_pair_no_slippage(entry, exit)

        a = self._ticks_around(entry)
        b = self._ticks_around(exit)

        t_entry = self.ask_picker(entry, a["l1-askprice"])
        t_exit = self.bid_picker(exit, b["l1-bidprice"])
        return t_entry, t_exit

    def _get_pair_short(self, entry, exit):
        if self.disable_slippage:
            return self._get_pair_no_slippage(entry, exit)

        a = self._ticks_around(entry)
        b = self._ticks_around(exit)

        t_entry = self.bid_picker(entry, a["l1-bidprice"])
        t_exit = self.ask_picker(exit, b["l1-askprice"])
        return t_entry, t_exit

    @lru_cache(maxsize=16)
    def get_entry_exit_pair(self, entry, exit, pred):
        """Get (entry, exit) prices, raise InvalidTrade if no data."""
        if pred == TernaryDirection.UP.value:
            pair = self._get_pair_long(entry, exit)
        elif pred == TernaryDirection.DOWN.value:
            pair = self._get_pair_short(entry, exit)
        else:
            raise ValueError("Unkown side for pred: {}".format(pred))
        return pair

    def get_pl(self, entry, exit, pred):
        """Get profit/loss for a completed trade, raise InvalidTrade if no data."""
        a, b = self.get_entry_exit_pair(entry, exit, pred)
        if pred == TernaryDirection.UP.value:
            gain = b.value - a.value
        elif pred == TernaryDirection.DOWN.value:
            gain = a.value - b.value
        else:
            raise ValueError("Unkown side for pred: {}".format(pred))
        cost = self.transaction_cost * 2
        return gain - cost

    def iter_on_pairs(self, pairs):
        for p in pairs:
            entry, exit, pred = p[0][0], p[1][0], p[0][1]
            dic = {"entry": entry, "exit": exit, "pred": pred}
            try:
                ex_entry, ex_exit = self.get_entry_exit_pair(entry, exit, pred)
                pl = self.get_pl(entry, exit, pred)
                dic.update(
                    {
                        "entry_preceding_tick": ex_entry.index,
                        "entry_value": ex_entry.value,
                        "exit_preceding_tick": ex_exit.index,
                        "exit_value": ex_exit.value,
                        "pl": pl,
                        "valid": True,
                    }
                )
                yield dic
            except InvalidTrade:
                dic.update({"valid": False})
                yield dic
