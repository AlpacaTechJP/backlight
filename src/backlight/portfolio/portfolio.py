import pandas as pd
import numpy as np
from typing import List

from backlight.positions.positions import Positions
from backlight.datasource.marketdata import MarketData
from backlight.metrics.position_metrics import calc_pl
from backlight.positions import calc_positions
from backlight.trades.trades import Trades
from joblib import Parallel, delayed


def construct_portfolio(
    trades: List[Trades], mkt: List[MarketData], principal_per_asset: float
) -> Portfolio:
    """
    Take a list of Trades and MarketData and return aq portfolio
    """

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(trade, market, principal=principal_per_asset)
            for (trade, market) in zip(trades, mkt)
        ]
    )
    return Portfolio(positions)


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    def __init__(self, positions: List[Positions]):
        self.portfolio = positions
        self.pl = pd.DataFrame()

    @property
    def value(self) -> pd.DataFrame:
        """ DataFrame of the portfolio valuation of each asset"""
        for asset_positions in self.portfolio:
            # Compute PL of positions of each asset
            self.pl[asset_positions.symbol] = calc_pl(asset_positions)
        return self.pl

    # Normalize PL to one asset reference and sum
    def normalized_total_pl(
        self, mkt: List[MarketData] = [], type_asset: str = "stock"
    ) -> pd.DataFrame:
        """
        Normalize PL to one asset reference and sum
        args:
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
        # If not FX, just add all pl
        if type_asset != "FX":
            return self.pl.sum(1)

        if type_asset == "FX":
            # check if in case all currecnies have same suffix, use last 3 chars as reference
            reference_to_check = self.pl.columns[0][-3:]

            if sum(self.pl.columns.str.endswith(reference_to_check)) == len(
                self.portfolio
            ):
                return self.pl.sum(1)
            else:
                # Fix me : add support of different assets
                raise ValueError(
                    "Cross FX is not supported, specify FX with same base  e.g. USDJPY, GBPJPY"
                )
