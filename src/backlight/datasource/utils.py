import pandas as pd
from typing import Optional, List

from backlight.datasource.marketdata import (
    MarketData,
    MidMarketData,
    AskBidMarketData,
    ForexMarketData,
)
from backlight.query import query
from backlight.asset.currency import Currency


def load_marketdata(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    url: str,
    currency_unit: Currency,
    quote_currency: Optional[Currency] = None,
    base_currency: Optional[Currency] = None,
) -> MarketData:
    """An abstraction interface for loading the market data.

    Args:
        symbol :  A symbol to query
        start_dt :  Date to query from
        end_dt :  Date to query to
        url :  An url to the data source

    Returns:
        MarketData
    """
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(
        df,
        symbol,
        currency_unit,
        col_mapping=None,
        quote_currency=quote_currency,
        base_currency=base_currency,
    )


def from_dataframe(
    df: pd.DataFrame,
    symbol: str,
    currency_unit: Currency,
    col_mapping: Optional[dict] = None,
    quote_currency: Optional[Currency] = None,
    base_currency: Optional[Currency] = None,
) -> MarketData:
    """Create a MarketData instance out of a DataFrame object

    Args:
        df :  DataFrame
        symbol :  A symbol to query
        col_mapping :  A dict to map columns

    Returns:
        MarketData
    """
    df = df.copy()

    if col_mapping is not None:
        df = df.rename(columns=col_mapping)

    mkt = None

    if ("ask" in df.columns) and ("bid" in df.columns):
        from backlight.datasource.marketdata import AskBidMarketData

        mkt = AskBidMarketData(df)
    elif "mid" in df.columns:
        from backlight.datasource.marketdata import MidMarketData

        mkt = MidMarketData(df)

    if mkt is None:
        raise ValueError("Unsupported marketdata")

    mkt.symbol = symbol
    mkt.currency_unit = currency_unit
    mkt.base_currency = base_currency
    mkt.quote_currency = quote_currency
    mkt.reset_cols()

    return mkt


def mid2askbid(mkt: MidMarketData, spread: float) -> AskBidMarketData:
    """Convert MidMarketData to AskBidMarketData.

    Args:
        mkt: MidMarketData
        spread: Constant ask/bid spread added on mid price.
    """
    mkt.loc[:, "ask"] = mkt.mid + spread
    mkt.loc[:, "bid"] = mkt.mid - spread
    return from_dataframe(mkt, mkt.symbol, mkt.currency_unit)


def get_forex_ratios(
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
