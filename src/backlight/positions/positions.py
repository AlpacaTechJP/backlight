import pandas as pd
import numpy as np
from typing import Type, Callable

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.trades.trades import Trades
from backlight.asset.currency import Currency


def _freq(idx: pd.Index) -> pd.Timedelta:
    if idx.freq is not None:
        return idx.freq
    if len(idx) > 1:
        return idx[1] - idx[0]
    return pd.Timedelta("1s")


class Positions(pd.DataFrame):
    """Positions container which inherits pd.DataFrame.
    
    They have following columns:
        - `amount`: Amount of the asset you are holding at that moment.
        - `price`: Price per unit of the asset at that moment.
        - `principal`: Principal in your bank account at that moment.
    """

    _metadata = ["symbol", "currency_unit"]

    _target_columns = ["amount", "price", "principal"]

    def reset_cols(self) -> None:
        """ Keep only _target_columns"""
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def value(self) -> pd.Series:
        """ Series of the position valuation"""
        return self.amount * self.price + self.principal

    @property
    def _constructor(self) -> Type["Positions"]:
        return Positions


def _pricer(trades: Trades, mkt: MarketData, principal: float) -> pd.DataFrame:
    trade = trades.amount

    # historical data
    idx = mkt.index[trade.index[0] <= mkt.index]  # only after first trades
    positions = pd.DataFrame(index=idx)
    positions.loc[:, "amount"] = trade.cumsum()
    positions.loc[:, "price"] = mkt.mid.loc[idx]
    fee = mkt.fee(trade)
    positions.loc[:, "principal"] = -fee.cumsum() + principal
    positions = positions.ffill()

    # add initial data
    initial_idx = idx[0] - _freq(idx)
    positions.loc[initial_idx, "amount"] = 0.0
    positions.loc[initial_idx, "price"] = 0.0
    positions.loc[initial_idx, "principal"] = principal

    return positions.sort_index()


def calc_positions(
    trades: Trades, mkt: MarketData, principal: float = 0.0
) -> Positions:
    """Create Positions from Trades and MarketData.
    Positions' frequency is determined by MarketData's frequency.
    
    Args:
        trades: Tuple of trades.
        mkt: Market data.
        principal: The initial principal value.
    """
    assert trades.symbol == mkt.symbol
    assert trades.currency_unit == mkt.currency_unit
    assert trades.index.isin(mkt.index).all()

    pos = Positions(_pricer(trades, mkt, principal))
    pos.reset_cols()
    pos.symbol = trades.symbol
    pos.currency_unit = trades.currency_unit
    return pos


def from_dataframe(df: pd.DataFrame, symbol: str, currency_unit: Currency) -> Positions:
    """
    Create Positions from dataframe and symbol.
    
    Args:
        df: DataFrame with the content of the future Positions.
        symbol: The Positions symbol.
        currency_unit: The Positions currency unit.
    """
    pos = Positions(df)
    pos.reset_cols()
    pos.symbol = symbol
    pos.currency_unit = currency_unit
    return pos
