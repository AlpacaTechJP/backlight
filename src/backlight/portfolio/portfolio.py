import pandas as pd
import numpy as np

from backlight.positions.positions import Positions


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    def __init__(self,positions: List[Positions]):
        self.portfolio = positions


    @property
    def value(self) -> pd.Series:
        """ Series of the portfolio valuation"""
        for asset_positions in self.portfolio:
            # Compute PL of positions of each asset

            # Normalize PL to one asset reference and sum
