import pandas as pd
import numpy as np
from typing import List

from backlight.positions.positions import Positions
from backlight.datasource.marketdata import MarketData
from backlight.metrics.position_metrics import calc_pl
from backlight.positions import calc_positions
from backlight.trades.trades import Trades
from joblib import Parallel, delayed


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    def __init__(self, positions: List[Positions]):
        """
        If there is positions with the same symbol, their value are sum
        """
        symbols = [position.symbol for position in positions]
        assert len(symbols) == len(set(symbols))
        self._positions = positions

    def value(self) -> pd.DataFrame:
        """ DataFrame of the portfolio valuation of each asset"""
        pl = pd.DataFrame()
        for p in self._positions:
            # Compute PL of positions of each asset
            pl[p.symbol] = calc_pl(p)
        return pl

    def get_amount(self, symbol: str) -> pd.Series:
        """ Return amounts of each asset in the portfolio at each time step"""
        for p in self._positions:
            if p.symbol == symbol:
                return p.amount
        raise ValueError("Passed symbol not found in portfolio")


def calculate_lots_size(
    mkt: List[MarketData], principal: List[float], max_amount: List[int]
) -> List[int]:
    """
    Compute lot_size based on the principal, max_amount and the makrtdata.
    lot = (Principal/max_amount) / (market_price_at_0)

    """
    lots = []
    for (m, p, amount) in zip(mkt, principal, max_amount):
        lots.append(int((p / amount) / m.mid[0]))
    return lots


def construct_portfolio(
    trades: List[Trades],
    mkt: List[MarketData],
    principal: List[float],
    lot_size: List[int],
) -> Portfolio:
    """
    Take a list of Trades and MarketData and return a portfolio
    args:
        - trades : list of unit trades (1 or -1 for th amount)
        - mkt : list of mkt data
        - principal: list of principal per asset
        - lot_size : list of lot sizes per asset
                    (e.g. trade.amount = 1 is equivalent to buying 1*lot_size assets)
    return:
        Portfolio
    """

    symbols2mkt = {m.symbol: m for m in mkt}
    symbols = [t.symbol for t in trades]
    assert set(symbols) == set(symbols2mkt.keys())

    # Transform trades following the lot_size
    for (trade, lot) in zip(trades, lot_size):
        trade["amount"] *= lot

    """
    There is an issue here :
    - We have two solutions : we fusion the trades of same symbols before calculating the positions or
    we calculate the positions and then fusion them for same symbols.
    - In the first case, the problem of the _id column risks to be heavy to solve as much in term of code as
    in term of computation time. Morever, here, the trades and the market datas are assumed to be ranked in 
    the same order (regarding of the symbols). If its not the case its impossible to know it. Same for principal.
    - In the second case, we need to duplicate the market and principal columns to fit the size of the trades. 
    It will be time and memory consuming.
    
    I implemented the 2nd one at the moment.
    """

    new_mkt = [symbols2mkt.get(t.symbol) for t in trades]
    assert len(principal) == len(trades)
    assert len(lot_size) == len(trades)

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(trade, market, principal=principal_per_asset)
            for (trade, market, principal_per_asset) in zip(trades, new_mkt, principal)
        ]
    )

    # Here are the modifications done on positions after treatment.
    symbols = [p.symbol for p in positions]
    if len(set(symbols)) != len(symbols):
        unique_positions = []
        for position in positions:
            s = position.symbol
            if symbols.count(s) > 1:
                df = pd.DataFrame(
                    data=np.array([p.values for p in positions if p.symbol == s]).sum(
                        axis=0
                    ),
                    index=position.index,
                    columns=position.columns,
                )
                pos = Positions(df)
                pos.symbol = s
                unique_positions.append(pos)
                for _ in range(symbols.count(s)):
                    symbols.remove(s)
            elif symbols.count(position.symbol) == 1:
                unique_positions.append(position)
        positions = unique_positions

    return Portfolio(positions)


def calculate_pl(pt: Portfolio, mkt: List[MarketData]) -> pd.DataFrame:
    """
    Apply the sum on the homogenized portfolio
    args:
        - portfolio : a defined portfolio
        - mkt : list of marketdata for each asset
        - base_ccy : asset of reference
    
    """
    pl = pt.value()
    return pl.sum(axis=1)
