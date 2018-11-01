from ..common import TernaryDirection, get_majority_label
import numpy as np


def fill_binary_label_toward_zero(df):
    in_df = df.copy()
    in_df.loc[in_df.label_diff > 0, "label"] = TernaryDirection.UP.value
    in_df.loc[in_df.label_diff < 0, "label"] = TernaryDirection.DOWN.value
    # Dealing with Edge Case
    majority_label = get_majority_label(in_df.label)
    in_df.loc[in_df.label_diff == 0, "label"] = majority_label
    in_df.loc[np.isinf(in_df.label_diff), "label"] = majority_label
    in_df.loc[np.isnan(in_df.label_diff), "label"] = majority_label
    return in_df
