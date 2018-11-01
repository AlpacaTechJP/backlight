from .positions import Positions


def _mid_trader(trades, mkt):
    positions = mkt.copy()
    positions.loc[:, "amount"] = trades.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid
    return Positions(positions, trades.symbol)


def calc_positions(trades, mkt, trader=_mid_trader):
    assert trades.symbol == mkt.symbol
    positions = trader(trades, mkt)
    return positions
