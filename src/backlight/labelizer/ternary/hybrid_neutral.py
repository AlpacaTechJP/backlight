import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.labelizer.common import LabelType, TernaryDirection
from backlight.labelizer.labelizer import Label
from backlight.labelizer.ternary.static_neutral import StaticNeutralLabelizer
from backlight.labelizer.ternary.dynamic_neutral import (
    MarketCloseAwareDynamicNeutralLabelizer,
)


class HybridNeutralLabelizer(
    StaticNeutralLabelizer, MarketCloseAwareDynamicNeutralLabelizer
):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.validate_params()

    def validate_params(self) -> None:
        super(HybridNeutralLabelizer, self).validate_params()
        super(MarketCloseAwareDynamicNeutralLabelizer, self).validate_params()
        assert "alpha" in self._params
        assert 0 <= self._params["alpha"] <= 1

    def _calculate_hybrid_neutral_range(self, diff_abs: pd.Series) -> pd.Series:
        snr = self._calculate_static_neutral_range(diff_abs)
        dnr = self._calculate_dynamic_neutral_range(diff_abs)
        return self.alpha * snr + (1 - self.alpha) * dnr

    def create(self, mkt: MarketData) -> pd.DataFrame:
        mid = mkt.mid.copy()
        future_price = mid.shift(freq="-{}".format(self._params["lookahead"]))
        diff = (future_price - mid).reindex(mid.index)
        diff_abs = diff.abs()
        neutral_range = self._calculate_hybrid_neutral_range(diff_abs)
        df = mid.to_frame("mid")
        df.loc[:, "label_diff"] = diff
        df.loc[:, "neutral_range"] = neutral_range
        df.loc[df.label_diff > 0, "label"] = TernaryDirection.UP.value
        df.loc[df.label_diff < 0, "label"] = TernaryDirection.DOWN.value
        df.loc[diff_abs < neutral_range, "label"] = TernaryDirection.NEUTRAL.value
        df = Label(df[["label_diff", "label", "neutral_range"]])
        df.label_type = LabelType.TERNARY
        return df

    @property
    def alpha(self) -> float:
        return self._params["alpha"]
