import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sensitivity(
    sensitivity: pd.DataFrame,
    save_path=None,
    show=False,
    max_xticks=50,
    **kwargs,
):
    cmap = kwargs.get("cmap", "gray")
    figsize = kwargs.get("figsize", (20, 10))
    plt.figure(figsize=figsize)
    c = plt.imshow(
        sensitivity.T, aspect="auto", cmap=cmap, interpolation="nearest"
    )

    xtickslabels = list(sensitivity.index)
    xticks = list(
        range(0, len(sensitivity), len(sensitivity) // max_xticks + 1)
    )

    xtickslabels = np.array(xtickslabels)[xticks].tolist()
    xtickslabels = [l.strftime("%Y-%m-%d") for l in xtickslabels]
    plt.xticks(xticks, xtickslabels, rotation=45)

    ytickslabels = list(sensitivity.columns)
    plt.yticks(range(len(ytickslabels)), ytickslabels)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
