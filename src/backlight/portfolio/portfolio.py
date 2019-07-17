import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from functools import reduce

from backlight.positions.positions import Positions
import backlight.positions.positions
from backlight.datasource.marketdata import MarketData
from backlight.metrics.position_metrics import calc_pl
from backlight.positions import calc_positions
from backlight.trades.trades import Trades, from_dataframe
from joblib import Parallel, delayed
from backlight.asset.currency import Currency


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    _metadata = ["currency_unit"]

    def __init__(
        self, positions: List[Positions], currency_unit: Currency = Currency.USD
    ):
        """
        If there is positions with the same symbol, their value are sum
        """
        symbols = [position.symbol for position in positions]
        assert len(symbols) == len(set(symbols))
        self._positions = positions

    @property
    def value(self) -> pd.Series:
        """ DataFrame of the portfolio valuation of each asset"""
        pl = pd.DataFrame()
        for p in self._positions:
            pl[p.symbol] = p.value
        return pl.sum(axis=1)

    def get_amount(self, symbol: str) -> pd.Series:
        """ Return amounts of each asset in the portfolio at each time step"""
        for p in self._positions:
            if p.symbol == symbol:
                return p.amount
        raise ValueError("Passed symbol not found in portfolio")

    @property
    def amount(self) -> pd.DataFrame:
        return (reduce(lambda x, y: x.add(y, fill_value=0), self._positions)).amount


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

    # Find an other place to check these or check it by sets
    #     assert len(principal) == len(trades)
    #     assert len(lot_size) == len(trades)

    symbols2mkt = {m.symbol: m for m in mkt}
    symbols = [t.symbol for t in trades]

    # Here it is a problem that the symbols have to be the same.
    # It is better that the mkt symbols contains the trades symbols.
    # assert set(symbols) == set(symbols2mkt.keys())
    assert len(set(symbols).intersection(set(symbols2mkt.keys()))) == len(set(symbols))

    # Transform trades following the lot_size
    mult_trades = []
    for trade in trades:
        mult_trade = trade.copy()
        mult_trade["amount"] *= lot_size.get(trade.symbol)
        mult_trades.append(mult_trade)

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(
                trade, symbols2mkt[trade.symbol], principal=principal.get(trade.symbol)
            )
            for trade in mult_trades
        ]
    )

    converted_positions = []
    for p in positions:
        if p.currency_unit != currency_unit:
            converted_positions.append(_convert_currency_unit(p, mkt, currency_unit))
        else:
            converted_positions.append(p.copy())

    symbols = [s.symbol for s in converted_positions]
    if len(set(symbols)) != len(symbols):
        converted_positions = _fusion_positions(converted_positions)

    filled_positions = []
    union_indexes = converted_positions[0].index.union_many(
        [c.index for c in converted_positions]
    )
    for p in converted_positions:
        filled_positions.append(_bfill_principal(p, union_indexes))

    portfolio = Portfolio(filled_positions, currency_unit)

    return portfolio


def _bfill_principal(position: Positions, index: pd.DatetimeIndex) -> Positions:
    """
    From a create a Positions with all the indexes of index, and the values of position.
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


def equally_weighted_portfolio(
    trades: List[Trades],
    mkt: List[MarketData],
    principal: float,
    max_amount: float,
    currency_unit: Currency = Currency.USD,
) -> Portfolio:
    """
    Create a Portfolio from trades and mkt, given a principal which will be divided equally between the 
    different currencies.
    args :
        - trades : a list of trades for each currencies
        - mkt : the market datas for at least each trades currencies
        - principal : the total amount allocated to the Portfolio
        - max_amount : the max amount
        - currency_unit : the unit type of the future Portfolio
    """
    nb_currencies = len(trades)

    symbols = [t.symbol for t in trades]
    symbols2mkt = {m.symbol: m for m in mkt}
    symbols2tds = {t.symbol: t for t in trades}

    principals = {}
    #     max_amounts = {}
    lts = {}
    for trade in trades:
        symbol = trade.symbol
        starting_date = trade.index[0]
        trade_currency = trade.currency_unit

        ratio = 1
        if trade_currency != currency_unit:
            # Not very optimal since only one item is needed, but easier to read. Maybe we can change it if its bottleneck.
            ratios = _get_forex_ratios(mkt, trade_currency, currency_unit)
            ratio = ratios.iloc[ratios.index.get_loc(trade.index[0]) - 1]

        count_symbol = symbols.count(symbol)
        principals[symbol] = principal / (nb_currencies * ratio * count_symbol)
        lts[symbol] = int(principal / (max_amount))

    return construct_portfolio(trades, mkt, principals, lts, currency_unit)


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
    assert positions.index.isin(mkt[0].index).all()

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
    mkt: List[MarketData], ccy: Currency, base_ccy: Currency
) -> pd.Series:
    """
    Get the ratios of ccy expressed in base_ccy depending on market datas
    args:
        - market : market forex datas
        - ccy : the currency to convert from
        - base_ccy : the currency to convert to
    """
    for market in mkt:
        if ccy.to_symbol(base_ccy) == market.symbol:
            ratios = pd.Series(market.mid.values, index=market.index, dtype=float)
        elif base_ccy.to_symbol(ccy) == market.symbol:
            ratios = pd.Series(market.mid.values, index=market.index, dtype=float)
            ratios = ratios.apply(lambda x: 0 if x == 0 else 1.0 / float(x))

    return ratios
