import pandas as pd
from typing import Type

from backlight.datasource.marketdata import MarketData


class Label(pd.DataFrame):

    _metadata = ["label_type"]

    def stats(self) -> pd.DataFrame:
        return self.label.describe()

    @property
    def _constructor(self) -> Type[Label]:
        return Label


class Labelizer:
    def __init__(self, **kwargs: str) -> None:
        self._params = kwargs.copy()
        self.validate_params()

    def validate_params(self) -> None:
        pass

    def generate(self, mkt: MarketData) -> pd.DataFrame:
        raise NotImplementedError
