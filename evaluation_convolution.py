import numpy as np
from deep_active_learning.utils import get_strategy, get_dataset
import sampling
import architecture
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
from plotting_util import MultiPlotHandler, PlotHandler, DictList


@dataclass
class Hyperparameters:
    architecture: dict
    dataset: str = "MNIST"

    seed: int = 1

    n_init_labeled: int = 10000
    n_query: int = 1000
    n_round: int = 10
    num_train: int = 100000
    num_test: float = 0.25

    strategy_name: str = "LeastConfidence"
    # TODO do not edit this!!


# choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"]

# choices = ["RandomSampling",
#            "LeastConfidence",
#            "MarginSampling",
#            "EntropySampling",
#            "LeastConfidenceDropout",
#            "MarginSamplingDropout",
#            "EntropySamplingDropout",
#            "KMeansSampling",
#            "KCenterGreedy",
#            "BALDDropout",
#            "AdversarialBIM",
#            "AdversarialDeepFool"], help = "query strategy")

num_epochs_for_architecture = {
    "small": 20,
     "medium": 30,
     "large": 40
}
optim_params = {'n_epoch': 20,
                'train_args': {'batch_size': 256, 'num_workers': 0},
                'test_args': {'batch_size': 1024, 'num_workers': 0},
                'optimizer_args': {'lr': 1e-3,
                                   'weight_decay': 1e-4}}

RUN_NAME = "400_200_15k"
DEFAULT_ARGS = Hyperparameters(seed=200,
                               dataset="FashionMNIST",
                               n_init_labeled=400,
                               n_query=200,
                               n_round=15,
                               num_train=15000,
                               num_test=0.5,
                               architecture={
                                   "conv_channels": [10, 20, 20],
                                   "fc_dims": [50, 20],
                                   "conv_kernel": 5,
                                   "pool_kernel": 2
                               },
                               strategy_name="EntropySamplingDropout")

datasets = [
    "MNIST",  # 28x28x1, 10
    # "FashionMNIST",  # 28x28x1, 10
    "SVHN", #32x32x3, 10
    # "CIFAR10"  # 32x32x3, 10
]

strategies = [
              "RandomSampling",
              "LeastConfidence",
              "LeastConfidenceDropout",
              "BALDDropout",
              "KMeansSampling",
]

cnn_layouts = {
    "small": {"conv_channels": [10, 20],
                        "fc_dims": [50],
                        "conv_kernel": 5,
                        "pool_kernel": 2},
    "medium": {"conv_channels": [10, 20, 20],
                        "fc_dims": [50, 40],
                        "conv_kernel": 5,
                        "pool_kernel": 2},
    "large": {"conv_channels": [10, 20, 20, 20],
                        "fc_dims": [50, 40, 30],
                        "conv_kernel": 5,
                        "pool_kernel": 2},
}

param_overlay_name = "strategy_name"
param_overlay_options = strategies

param_horiz_name = "architecture"
param_horiz_options = [
    "small",
    "medium",
    "large"
]

param_vert_name = "dataset"
param_vert_options = datasets



def calc_deviation_from_full(preds, preds_full):
    return 1.0 - 1.0 * (preds_full == preds).sum().item() / len(preds_full)


def test(args: Hyperparameters):
    print(args.__dict__)

    num_test = int(args.num_test * args.num_train)
    dataset = get_dataset(args.dataset, train_size=args.num_train, test_size=num_test)
    input_shape = dataset.input_shape
    output_dims = dataset.output_dims

    cnn_layout = cnn_layouts[args.architecture]

    classifier_params = architecture.CNNParams(input_dim=input_shape,
                                               output_dim=output_dims,
                                               fc_dims=cnn_layout["fc_dims"],
                                               channel_dims=cnn_layout["conv_channels"],
                                               conv_kernel=cnn_layout["conv_kernel"],
                                               pool_kernel=cnn_layout["pool_kernel"])

    net = architecture.get_cnn_net(classifier_params, optim_params)
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy



    # start experiment
    lists = DictList()
    lists.start_list("test_accuracies")
    lists.start_list("train_accuracies")
    lists.start_list("test_deviation")
    lists.start_list("train_deviation")
    lists.start_list("dataset_size")

    dataset.initialize_labels(args.n_init_labeled)

    epochs = num_epochs_for_architecture[args.architecture]
    if args.dataset == "SVHN":
        epochs *= 2
    optim_params["n_epoch"] = epochs

    print("Full Accuracy")
    net_full = architecture.get_cnn_net(classifier_params, optim_params)
    dataset_full = dataset.copy()
    strategy_full = get_strategy(args.strategy_name)(dataset_full, net_full)  # load strategy
    strategy_full.label_entire_dataset()
    strategy_full.train()
    preds_full_test = strategy_full.predict(dataset_full.get_test_data())
    full_accuracy_test = dataset_full.cal_test_acc(preds_full_test)
    _, train_data = dataset.get_train_data()
    preds_full_train = strategy_full.predict(train_data)
    full_accuracy_train = dataset_full.cal_train_acc(preds_full_train)
    print()
    print(f"Full testing accuracy: {full_accuracy_test}")

    def do_round(i):
        # round 0 accuracy
        print("Round " + str(i))
        dataset_size = len(np.extract(dataset.labeled_idxs == True, dataset.labeled_idxs))
        print("dataset size: ", dataset_size)
        strategy.train()
        preds = strategy.predict(dataset.get_test_data())
        _, train_data = dataset.get_train_data()
        preds_train = strategy.predict(train_data)
        accuracy_test = dataset.cal_test_acc(preds)
        accuracy_train = dataset.cal_train_acc(preds_train)
        print(f"Round " + str(i) + f") testing accuracy: {accuracy_test}")

        deviation_from_full = accuracy_train
        print("Training Accuracy: ", deviation_from_full)
        # deviation_from_full = calc_deviation_from_full(preds, preds_full)
        # print("Deviation from full: ", deviation_from_full)
        lists.add_to_list("test_accuracies", accuracy_test)
        lists.add_to_list("train_accuracies", accuracy_train)
        lists.add_to_list("test_deviation", np.abs(full_accuracy_test - accuracy_test))
        lists.add_to_list("train_deviation", np.abs(full_accuracy_train - accuracy_train))
        lists.add_to_list("dataset_size", dataset_size * 1.0 / args.num_train)

    do_round(0)
    for rd in range(1, args.n_round + 1):
        # query
        print("Querying....")
        query_idxs = strategy.query(args.n_query)


        # update labels
        strategy.update(query_idxs)
        do_round(rd)

    return lists


def get_evaluations_multiple(param_defaults,
                             param_overlay_name, param_overlay_options,
                             param_horiz_name, param_horiz_options,
                             param_vert_name, param_vert_options):
    plt.ion()
    c = len(param_horiz_options)
    r = len(param_vert_options)

    plot_names = ["test_accuracies", "train_accuracies", "test_deviation", "train_deviation"]
    multi_plot_handler = MultiPlotHandler(r=r, c=c, list_names=plot_names)

    for i_pv, param_vert in enumerate(param_vert_options):
        for i_ph, param_horiz in enumerate(param_horiz_options):
            for i_po, param_overlay in enumerate(param_overlay_options):
                print("=" * 100)
                param_defaults.__dict__[param_overlay_name] = param_overlay
                param_defaults.__dict__[param_vert_name] = param_vert
                param_defaults.__dict__[param_horiz_name] = param_horiz

                lists = test(param_defaults)
                title = (str(param_horiz_name) + ": " + str(param_horiz) + ", " +
                         str(param_vert_name) + ": " + str(param_vert))

                multi_plot_handler.fill_with_list(v=i_pv, h=i_ph, label=str(param_overlay), lists=lists, title=title,
                                                  x_axis_name="frac labeled data", y_axis_name="")

                plt.draw()
                plt.pause(0.5)

    # plt.show()
    multi_plot_handler.save(RUN_NAME)


if __name__ == "__main__":
    get_evaluations_multiple(param_defaults=DEFAULT_ARGS,
                             # distribution=distribution, label_distribution=label_distribution,
                             param_overlay_name=param_overlay_name, param_overlay_options=param_overlay_options,
                             param_horiz_name=param_horiz_name, param_horiz_options=param_horiz_options,
                             param_vert_name=param_vert_name, param_vert_options=param_vert_options)
