import pandas as pd

from backlight.positions.positions import Positions
from backlight.metrics.position_metrics import calculate_pl


def plot_pl(
    positions: Positions,
    window: pd.Timedelta = pd.Timedelta("1D"),
    freq: pd.Timedelta = pd.Timedelta("1D"),
) -> None:
    """Plot rolled pl."""
    pl = calculate_pl(positions)
    pl.rolling(window).sum().resample(freq).first().plot()


def plot_cumulative_pl(positions: Positions) -> None:
    """Plot cumulative pl"""
    pl = calculate_pl(positions)
    pl.cumsum().plot()
