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
        self.portfolio = positions

    def value(self) -> pd.DataFrame:
        """ DataFrame of the portfolio valuation of each asset"""
        pl = pd.DataFrame()
        for asset_positions in self.portfolio:
            # Compute PL of positions of each asset
            pl[asset_positions.symbol] = calc_pl(asset_positions)
        return pl


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
    # Transform trades following the lot_size
    for (trade, lot) in zip(trades, lot_size):
        trade.amount(lot)

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(trade, market, principal=principal_per_asset)
            for (trade, market, principal_per_asset) in zip(trades, mkt, principal)
        ]
    )
    return Portfolio(positions)


# Normalize PL to one asset reference and sum
def normalized_total_pl(
    portfolio: Portfolio, mkt: List[MarketData] = [], type_asset: str = "stock"
) -> pd.DataFrame:
    """
    Normalize PL to one asset reference and sum
    args:
        - portfolio : a defined portfolio
        - mkt : list of marketdata for each asset
        - type_asset: FX or same (same are asset that supposed to have pl in the same currency, like stocks or USDJPY and EURJPY)

        # To add when supporting cross assets
        - reference : asset of reference
                      for FX, one should pay attention to the family of asset that
                      share the same base (for example EURJPY, USDJPY, GBPJPY)
                      then an auto reference would be JPY

                      in case of cross FX (EURUSD, USDJPY) we should convert USD pl from first asset
                      to JPY using USDJPY market

    """

    # Compute pl per asset in the portfolio
    pl = portfolio.value()

    # If not FX, just add all pl
    if type_asset != "FX":
        return pl.sum(1)

    if type_asset == "FX":
        # check if in case all currecnies have same suffix, use last 3 chars as reference
        reference_to_check = pl.columns[0][-3:]

        if sum(pl.columns.str.endswith(reference_to_check)) == len(portfolio.portfolio):
            return pl.sum(1)
        else:
            # Fix me : add support of different assets
            raise ValueError(
                "Cross FX is not supported, specify FX with same base  e.g. USDJPY, GBPJPY"
            )
