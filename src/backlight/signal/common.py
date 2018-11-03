from backlight.signal.signal import BinarySignal, TernarySignal


def simple_signal_factory(df):

    if ("up" in df.columns) and ("neutral" in df.columns) and ("down" in df.columns):
        return TernarySignal(df)

    if ("up" in df.columns) and ("down" in df.columns):
        return BinarySignal(df)

    raise ValueError("Unsupported signal format")
