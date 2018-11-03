from backlight.query import query
from backlight.signal.common import simple_signal_factory


def load_signal(symbol, url, start_dt, end_dt, factory=simple_signal_factory):
    df = query(symbol, start_dt, end_dt, url)
    sig = factory(df)
    if len(sig):
        sig.symbol = symbol
        sig.reset_cols()
        sig.reset_pred()
    return sig
