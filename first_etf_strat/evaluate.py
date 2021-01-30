import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_train_history(train_history: Dict, test_history: Dict, save_dir: Optional[str] = None, show: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    axs[0].plot(train_history['loss'])
    axs[0].plot(test_history['loss'])
    axs[0].set_title('Loss')
    axs[1].plot(train_history['avg_ret'])
    axs[1].plot(test_history['avg_ret'])
    axs[1].set_title('Average return')
    axs[2].plot(train_history['cum_ret'])
    axs[2].plot(test_history['cum_ret'])
    axs[2].set_title('Cum return')
    if save_dir:
        plt.savefig(save_dir)
    if show:
        plt.show()
