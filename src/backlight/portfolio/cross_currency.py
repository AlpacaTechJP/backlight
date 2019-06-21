import pandas as pd
from typing import List

from backlight.portfolio.portfolio import Portfolio
from backlight.datasource.marketdata import MarketData


def homogenize_portfolio(
    pt: Portfolio, mkt: List[MarketData], base_ccy: str = "USD"
) -> Portfolio:

    new_positions = []

    # We compute the intersection of index between market datas and portfolio positions for later use
    mkt_pos_intersection = mkt[0].index.intersection(pt.positions[0].index)

    for position in pt.positions:
        # ccy is the currency wich in which are expressed the element of the position, e.g. JPY for USDJPY
        ccy = position.symbol[-3:]

        if ccy == base_ccy:
            # The ccy is the base currency, no need to convert
            new_positions.append(position.loc[mkt_pos_intersection].copy())
        else:
            for market in mkt:
                # Looking for the ratio for converting in the base currency
                if ccy + base_ccy == market.symbol:
                    # We buy base_ccy at the bid price of the ccybase_ccy market
                    ratios = pd.Series(
                        market.bid.values, index=market.index, dtype=float
                    )
                elif base_ccy + ccy == market.symbol:
                    # We sell ccy at 1 / ask price of the base_ccyccy market
                    ratios = pd.Series(
                        market.ask.values, index=market.index, dtype=float
                    )
                    ratios = ratios.apply(lambda x: 0 if x == 0 else 1.0 / float(x))

            try:
                ratios
            except NameError:
                print(
                    "The currency "
                    + ccy
                    + " can't be convert to "
                    + base_ccy
                    + " because data is not on the market."
                )
            else:
                idx = pd.to_datetime(mkt_pos_intersection)
                pos_values = position.loc[idx].values
                ratios_values = ratios.loc[idx].values.reshape(
                    ratios.loc[idx].values.size, 1
                )

                # Depending of the ratios array previously computed, we get the value of the portfolio in the base_ccy
                new_p = pd.DataFrame(
                    pos_values * ratios_values, columns=position.columns, index=idx
                )

                new_positions.append(new_p)

    return Portfolio(new_positions)
