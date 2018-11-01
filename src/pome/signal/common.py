from .signal import BinaryOneColumnLabelSignal, Signal, TernarySignal


def simple_signal_factory(**kwargs):
    df = kwargs["df"]
    if ("up" in df.columns) and ("neutral" in df.columns) and ("down" in df.columns):
        return TernarySignal(**kwargs)
    if "label" in df.columns:
        return BinaryOneColumnLabelSignal(**kwargs)
    return Signal(**kwargs)
