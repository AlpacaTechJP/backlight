from enum import Enum

from backlight.labelizer.labelizer import Label


class LabelType:
    BINARY = "binary"
    TERNARY = "ternary"


class TernaryDirection(Enum):
    UP = 1
    NEUTRAL = 0
    DOWN = -1
    U = 1
    N = 0
    D = -1


def get_majority_label(orig_result):
    """Count the number of values in each class for classification problem and
    return the class whose count is highest.

    Args:
        orig_result(pandas.Series): labels Series

    Return:
        a value from orig_result and this value have the most count among all
            other values in orig_result
    """
    series_labels_counts = orig_result.value_counts()
    possible_labels = sorted(series_labels_counts.index.tolist(), reverse=True)
    majority_label = possible_labels[0]
    majority_label_count = series_labels_counts[possible_labels[0]]
    for v in possible_labels:
        if series_labels_counts[v] > majority_label_count:
            majority_label = v
            majority_label_count = series_labels_counts[v]
    return majority_label


def simple_label_factory(**kwargs):
    df = kwargs["df"].copy()

    if "neutral_range" in df.columns:
        df = df.rename(
            columns={"label_up_neutral_down": "label", "label_up_down": "label"}
        )
        if "mapping" in kwargs:
            df.loc[:, "label"] = df.label.apply(lambda x: kwargs["mapping"][x])
        lbl = Label(df[["label_diff", "label", "neutral_range"]])
        lbl.label_type = LabelType.TERNARY
        return lbl

    raise ValueError("Unsupported label format")
