import pandas as pd

from backlight.datasource.marketdata import MarketData


class Label(pd.DataFrame):

    _metadata = ["label_type"]

    def stats(self) -> pd.DataFrame:
        return self.label.describe()

    @property
    def _constructor(self):
        return Label


class Labelizer:
    def __init__(self, **kwargs):
        self._params = kwargs.copy()
        self.validate_params()

    def validate_params(self) -> None:
        pass

    def generate(self, mkt: MarketData) -> pd.DataFrame:
        raise NotImplementedError
