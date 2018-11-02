from backlight.signal.signal import (
    BinarySignal,
    TernarySignal,
)


def simple_signal_factory(**kwargs):
    df = kwargs["df"]

    if ("up" in df.columns) and ("neutral" in df.columns) and ("down" in df.columns):
        return TernarySignal(**kwargs)

    if ("up" in df.columns) and ("down" in df.columns):
        return BinarySignal(**kwargs)

    raise ValueError("Unsupported signal format")
