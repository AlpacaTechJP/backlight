import numpy as np
import pandas as pd

from backlight.labelizer.common import TernaryDirection, get_majority_label


def fill_binary_label_toward_zero(df: pd.DataFrame) -> pd.DataFrame:
    in_df = df.copy()
    in_df.loc[in_df.label_diff > 0, "label"] = TernaryDirection.UP.value
    in_df.loc[in_df.label_diff < 0, "label"] = TernaryDirection.DOWN.value
    # Dealing with Edge Case
    majority_label = get_majority_label(in_df.label)
    in_df.loc[in_df.label_diff == 0, "label"] = majority_label
    in_df.loc[np.isinf(in_df.label_diff), "label"] = majority_label
    in_df.loc[np.isnan(in_df.label_diff), "label"] = majority_label
    return in_df
