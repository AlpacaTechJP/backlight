import re
import matplotlib.pylab as plt
from pathlib import Path


def _replace_special_char(f):
    return re.compile("[^a-zA-Z0-9_().-]+").sub("_", f)


def save_fig(
    fig, ax, filename=None, save_dir="/notebooks/forecast/reports/figures/manual"
):
    """Save a figure and create directory if necessary."""
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    if filename is None:
        try:
            filename = str(fig._suptitle.get_text())
            filename = str(ax.get_title()) if filename is None else filename
            filename = str(ax[0].get_title()) if filename is None else filename
        except Exception:
            filename = "untitled"
    path = root.joinpath(_replace_special_char(filename))
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0.1)
    plt.close()
