from ..query import query
from .common import simple_signal_factory


def load_signal(model_id, url, start_dt, end_dt, factory=simple_signal_factory):
    df = query(model_id, start_dt, end_dt, url)

    return factory(df=df, symbol=model_id, start_dt=start_dt, end_dt=end_dt)
