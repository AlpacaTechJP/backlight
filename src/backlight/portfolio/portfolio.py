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

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(trade, market, principal=principal_per_asset)
            for (trade, market, principal_per_asset) in zip(trades, mkt, principal)
        ]
    )
    return Portfolio(positions)


def homogenize_pl(
    pt: Portfolio, mkt: List[MarketData], base_ccy: str = "USD"
) -> Portfolio:
    """
    Normalize PL to one asset reference and sum
    args:
       - portfolio : a defined portfolio
       - mkt : list of marketdata for each asset
       - base_ccy : asset of reference, all assets are converted this one
    """
    new_positions = []

    for position in pt._positions:
        ccy = position.symbol[-3:]
        if ccy == base_ccy:
            new_positions.append(
                position.loc[mkt[0].index.intersection(pt._positions[0].index)].copy()
            )
        else:
            new_positions.append(convert_positions(position, mkt, ccy, base_ccy))

    return Portfolio(new_positions)


def convert_positions(
    position: Positions, mkt: List[MarketData], ccy: str, base_ccy: str
) -> Positions:
    """
    Convert the values of a position in a different currency from MarketData
    args:
        - position : the position to convert
        - mkt : market forex datas
        - ccy : the currency of the position
        - base_ccy : the currency to express the position in
    """

    ratios = get_ratios(mkt, ccy, base_ccy)
    idx = pd.to_datetime(mkt[0].index.intersection(position.index))
    pos_values = position.loc[idx].values
    ratios_values = ratios.loc[idx].values.reshape(ratios.loc[idx].values.size, 1)

    new_p = Positions(
        pd.DataFrame(
            data=pos_values * ratios_values, columns=position.columns, index=idx
        )
    )
    new_p.symbol = position.symbol
    return new_p


def get_ratios(mkt: List[MarketData], ccy: str, base_ccy: str) -> pd.Series:
    """
    Get the ratios of ccy expressed in base_ccy depending on market datas
    args:
        - market : market forex datas
        - ccy : the currency to convert from
        - base_ccy : the currency to convert to
    """
    for market in mkt:
        if ccy + base_ccy == market.symbol:
            ratios = pd.Series(market.bid.values, index=market.index, dtype=float)
        elif base_ccy + ccy == market.symbol:
            ratios = pd.Series(market.ask.values, index=market.index, dtype=float)
            ratios = ratios.apply(lambda x: 0 if x == 0 else 1.0 / float(x))

    return ratios


def calculate_pl(
    pt: Portfolio, mkt: List[MarketData], base_ccy: str = "USD"
) -> pd.DataFrame:
    """
    Apply the sum on the homogenized portfolio
    args:
        - portfolio : a defined portfolio
        - mkt : list of marketdata for each asset
        - base_ccy : asset of reference    """
    hpt = homogenize_pl(pt, mkt, base_ccy)
    df = hpt._positions[0].copy()
    for position in hpt._positions[1:]:
        df = df + position
    return df
