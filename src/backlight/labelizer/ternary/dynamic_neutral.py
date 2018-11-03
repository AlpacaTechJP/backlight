from backlight.labelizer.common import LabelType, TernaryDirection
from backlight.labelizer.labelizer import Label, Labelizer


class DynamicNeutralLabelizer(Labelizer):
    def validate_params(self):
        assert "lookahead" in self._params
        assert "neutral_ratio" in self._params
        assert "neutral_window" in self._params
        assert "neutral_hard_limit" in self._params

    def _calc_dynamic_neutral_range(self, diff_abs):
        dnr = diff_abs.rolling(self._params["neutral_window"]).quantile(
            self.neutral_ratio
        )
        dnr[dnr < self.neutral_hard_limit] = self.neutral_hard_limit
        return dnr

    def generate(self, mkt):
        mid = mkt.mid.copy()
        future_price = mid.shift(freq="-{}".format(self._params["lookahead"]))
        diff = (future_price - mid).reindex(mid.index)
        diff_abs = diff.abs()
        neutral_range = self._calc_dynamic_neutral_range(diff_abs)
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
    def neutral_ratio(self):
        return self._params["neutral_ratio"]

    @property
    def neutral_window(self):
        return self._params["neutral_window"]

    @property
    def neutral_hard_limit(self):
        return self._params["neutral_hard_limit"]
