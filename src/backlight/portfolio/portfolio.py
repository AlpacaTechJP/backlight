import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from functools import reduce

from backlight.positions.positions import Positions
import backlight.positions.positions
from backlight.datasource.marketdata import MarketData, ForexMarketData
from backlight.positions import calc_positions
from backlight.trades.trades import Trades, from_dataframe
from joblib import Parallel, delayed
from backlight.asset.currency import Currency


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    def __init__(
        self, positions: List[Positions], currency_unit: Currency = Currency.USD
    ):
        """
        If there is positions with the same symbol, their value are sum
        """
        symbols = [position.symbol for position in positions]
        assert len(symbols) == len(set(symbols))
        units = [position.currency_unit for position in positions]
        assert len(set(units)) == 1
        self.currency_unit = currency_unit
        self._positions = positions

    @property
    def value(self) -> pd.Series:
        """ DataFrame of the portfolio valuation of each asset"""
        values = [p.value for p in self._positions]
        return sum(values)

    def get_amount(self, symbol: str) -> pd.Series:
        """ Return amounts of each asset in the portfolio at each time step"""
        for p in self._positions:
            if p.symbol == symbol:
                return p.amount
        raise ValueError("Passed symbol not found in portfolio")


def construct_portfolio(
    trades: List[Trades],
    mkt: List[MarketData],
    principal: Dict[str, float],
    lot_size: Dict[str, int],
    currency_unit: Currency = Currency.USD,
) -> Portfolio:
    """
    Take a list of Trades and MarketData and return a portfolio
    args:
        - trades : list of unit trades (1 or -1 for th amount)
        - mkt : list of mkt data
        - principal: list of principal per asset
        - lot_size : list of lot sizes per asset
                    (e.g. trade.amount = 1 is equivalent to buying 1*lot_size assets)
        - currency_unit : the unit type of the future Portfolio
    return:
        Portfolio
    """

    symbols2mkt = {m.symbol: m for m in mkt}
    symbols = [t.symbol for t in trades]

    assert len(set(symbols).intersection(set(symbols2mkt.keys()))) == len(set(symbols))
    mult_trades = _apply_lot_size(trades, lot_size)

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(
                trade, symbols2mkt[trade.symbol], principal=principal.get(trade.symbol)
            )
            for trade in mult_trades
        ]
    )

    positions = _standardize_currency(positions, mkt, currency_unit)
    symbols = [s.symbol for s in positions]
    if len(set(symbols)) != len(symbols):
        positions = _fusion_positions(positions)
    positions = _fill_positions(positions)

    portfolio = Portfolio(positions, currency_unit)

    return portfolio


def _fill_positions(positions: List[Positions]) -> List[Positions]:
    """
    For a given list of positions, return the list of these positions on the union of
    their indexes. For all new indexes, amount and price are set to 0 and principal is
    set to the first non nan principal.
    args :
        - positions : a list of Positions with different indexes.
    """
    filled_positions = []
    union_indexes = positions[0].index.union_many([c.index for c in positions])
    for p in positions:
        filled_positions.append(_bfill_principal(p, union_indexes))

    return filled_positions


def _standardize_currency(
    positions: List[Positions], mkt: List[MarketData], currency_unit: Currency
) -> List[Positions]:
    """
    For a given list of Positions, return the list of these Positions converted to
    a base currency_unit given market datas.
    args :
        - positions : a list of Positions with different currency types.
        - mkt : the market datas, supposed to cover all indexes of the positions.
        - currency_unit : a base currency to convert positions to.
    """
    standardized_positions = []
    for p in positions:
        if p.currency_unit != currency_unit:
            standardized_positions.append(_convert_currency_unit(p, mkt, currency_unit))
        else:
            standardized_positions.append(p.copy())

    return standardized_positions


def _apply_lot_size(trades: List[Trades], lot_size: Dict[str, int]) -> List[Trades]:
    """
    For a given list of Trades and a lot_size dictionary, multiply all trades amounts by
    the lot_size of their symbol and return the new list of Trades.
    args :
        - trades : a list of Trades.
        - lot_size : a dictionnary taking a symbol as entry and returning a lot_size.
    """
    mult_trades = []
    for trade in trades:
        mult_trade = trade.copy()
        mult_trade["amount"] *= lot_size.get(trade.symbol)
        mult_trades.append(mult_trade)

    return mult_trades


def _bfill_principal(position: Positions, index: pd.DatetimeIndex) -> Positions:
    """
    Create Positions with all the indexes of the index parameter, and the values of position.
    If there is nan, amount and price are filled with 0, and principal is filled with the first
    non-nan principal.
    args :
        - position : filled Positions
        - index : indexes, supposed to contains at least position's indexes
    """
    if position.index[0] == index[0]:
        return position

    filled_positions = pd.DataFrame(
        data=np.zeros((index.size, position.shape[1])),
        index=index,
        columns=position.columns,
    )
    filled_positions.principal.iloc[:] = position.principal[
        position.principal.first_valid_index()
    ]
    filled_positions.loc[position.index] = position
    return backlight.positions.positions.from_dataframe(
        filled_positions, position.symbol, position.currency_unit
    )


def _fusion_positions(positions: List[Positions]) -> List[Positions]:
    """
    Take a list of Positions and sum those with the same symbols
    args :
        - positions : a list of Positions to fusion
    """

    unique_positions = []
    symbols_and_units = [(p.symbol, p.currency_unit) for p in positions]
    columns = positions[0].columns

    for symbol, currency_unit in sorted(set(symbols_and_units)):
        positions_of_symbol = [p for p in positions if p.symbol == symbol]

        indices = [p.index for p in positions if p.symbol == symbol]
        union_index = indices[0].union_many(indices)

        dfs = [p for p in positions if p.symbol == symbol]
        df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)

        position = backlight.positions.positions.from_dataframe(
            df, symbol, currency_unit
        )
        unique_positions.append(position)

    return unique_positions


def _convert_currency_unit(
    positions: Positions, mkt: List[MarketData], base_ccy: Currency
) -> pd.Series:
    """
    Convert the values of profit-loss series in a different currency from MarketData
    args:
        - pl : the profit-loss to convert
        - mkt : market forex datas
        - ccy : the currency of the profit-loss
        - base_ccy : the currency to express the profit-loss in
    """
    # assert positions.index.isin(mkt[0].index).all()

    ratios = _get_forex_ratios(mkt, positions.currency_unit, base_ccy)

    converted_values = pd.DataFrame(
        data=np.zeros(positions.shape), index=positions.index, columns=positions.columns
    )

    converted_values.iloc[:, 0] = positions.iloc[:, 0]
    converted_values.iloc[:, 1] = positions.iloc[:, 1] * ratios
    converted_values.iloc[:, 2] = positions.iloc[:, 2] * ratios

    converted_positions = Positions(converted_values)
    converted_positions.currency_unit = base_ccy
    converted_positions.symbol = positions.symbol

    return converted_positions


def _get_forex_ratios(
    mkt: List[ForexMarketData], ccy: Currency, base_ccy: Currency
) -> pd.Series:
    """
    Get the ratios of ccy expressed in base_ccy depending on market datas
    args:
        - market : market forex datas
        - ccy : the currency to convert from
        - base_ccy : the currency to convert to
    """
    for market in mkt:
        if ccy == market.quote_currency and base_ccy == market.base_currency:
            ratios = pd.Series(market.mid.values, index=market.index, dtype=float)
        elif ccy == market.base_currency and base_ccy == market.quote_currency:
            ratios = pd.Series(market.mid.values, index=market.index, dtype=float)
            ratios = ratios.apply(lambda x: 0 if x == 0 else 1.0 / float(x))

    return ratios
