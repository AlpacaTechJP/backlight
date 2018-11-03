from backlight.positions.positions import Positions


def _mid_trader(trades, mkt):
    positions = mkt.copy()
    positions.loc[:, "amount"] = trades.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid
    pos = Positions(positions)
    pos.symbol = trades.symbol
    return pos


def calc_positions(trades, mkt, trader=_mid_trader):
    assert trades.symbol == mkt.symbol
    positions = trader(trades, mkt)
    return positions
