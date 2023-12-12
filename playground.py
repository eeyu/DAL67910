import numpy as np
from deep_active_learning.utils import get_strategy
import paths
import sampling
import architecture
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt
import evaluation

if __name__=="__main__":
    accuracies, deviations_from_full = evaluation.test(param_defaults, distribution, label_distribution)

    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(accuracies)
    axs[1].plot(deviations_from_full)
    plt.show()