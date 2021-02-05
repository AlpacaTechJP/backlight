import pandas as pd
import numpy as np

from backlight.datasource.marketdata import MarketData
from backlight.labelizer.common import LabelType, TernaryDirection
from backlight.labelizer.labelizer import Labelizer, Label


class StaticNeutralLabelizer(Labelizer):
    """Generates session-aware static labels

    Args:
        lookahead (str): Lookahead period
        session_splits (list[datetime.time]): EST local time to split sessions
        neutral_ratio (float): 0 < x < 1, Percentage of NEUTRAL labels
        window_start (str): Start date for lookback window
        window_end (str): End date for lookback window
        neutral_hard_limit (float): The minimum diff to label UP/DOWN
    """

    def validate_params(self) -> None:
        assert "lookahead" in self._params
        assert "session_splits" in self._params
        assert len(self._params["session_splits"])
        assert "neutral_ratio" in self._params
        assert "window_start" in self._params
        assert "window_end" in self._params
        assert "neutral_hard_limit" in self._params

    def _calculate_static_neutral_range(self, diff_abs: pd.Series) -> pd.Series:
        df = pd.DataFrame(diff_abs, columns=["diff"])
        df.loc[:, "est"] = df.index.tz_convert("America/New_York")
        df.loc[:, "res"] = np.nan

        mask = (
            (df.index >= self._params["window_start"])
            & (df.index < self._params["window_end"])
            & ~((df.est.dt.hour <= 17) & (df.est.dt.dayofweek == 6))
            & ((df.est.dt.hour < 16) | (df.est.dt.hour > 17))
            & ~((df.est.dt.hour >= 16) & (df.est.dt.dayofweek == 4))
            & (df.est.dt.dayofweek != 5)
        )

        splits = sorted(self._params["session_splits"])
        shifted_splits = splits[1:] + splits[:1]

        for s, t in list(zip(splits, shifted_splits)):
            if s >= t:
                scope = (df.est.dt.time >= s) | (df.est.dt.time < t)
            else:
                scope = (df.est.dt.time >= s) & (df.est.dt.time < t)
            df.loc[scope, "res"] = df.loc[scope & mask, "diff"].quantile(
                self.neutral_ratio
            )

        return df.res

    def create(self, mkt: MarketData) -> pd.DataFrame:
        mid = mkt.mid.copy()
        future_price = mid.shift(freq="-{}".format(self._params["lookahead"]))
        diff = (future_price - mid).reindex(mid.index)
        diff_abs = diff.abs()
        neutral_range = self._calculate_static_neutral_range(diff_abs)
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
    def neutral_ratio(self) -> str:
        return self._params["neutral_ratio"]

    @property
    def session_splits(self) -> str:
        return self._params["session_splits"]

    @property
    def neutral_hard_limit(self) -> str:
        return self._params["neutral_hard_limit"]
