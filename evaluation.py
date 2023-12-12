import argparse
import numpy as np
import torch
from deep_active_learning.utils import (get_dataset, get_net, get_strategy)
from pprint import pprint
import paths
import sampling
import architecture
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt


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

DEFAULT_ARGS = architecture.Hyperparameters(seed=100,
                                           n_init_labeled=100,
                                           n_query=100,
                                           n_round=10,
                                           num_train = 2000,
                                           num_test = int(1000/4),
                                           dimension = 30,
                                           num_classes = 2,
                                           nn_hidden_widths=[5,5,5],
                                           strategy_name="EntropySamplingDropout")

strategies = ["RandomSampling",
              "LeastConfidence",
              "MarginSamplingDropout",
              "KMeansSampling",
              "KCenterGreedy"]

param_overlay_name = "strategy_name"
param_overlay_options = strategies

param_horiz_name = "nn_hidden_widths"
param_horiz_options = [[10], [10, 10], [10, 10, 10]]

param_vert_name = "dimension"
param_vert_options = [10, 50, 100]

def calc_deviation_from_full(preds, preds_full):
    return 1.0 - 1.0 * (preds_full == preds).sum().item() / len(preds_full)

def test(args, distribution: sampling.Distribution, label_distribution: sampling.LabelDistribution):
    print(args.__dict__)

    num_train = args.num_train
    num_test = args.num_test
    dimension = args.dimension
    num_classes = args.num_classes

    # dataset = get_dataset(args.dataset_name)                   # load dataset
    dataset = architecture.get_randomized_dataset(x_distribution=distribution, y_distribution=label_distribution,
                                                  num_train=num_train, num_test=num_test)

    optim_params = {'n_epoch': 50,
                   'train_args':{'batch_size': 128, 'num_workers': 4},
                   'test_args':{'batch_size': 1000, 'num_workers': 4},
                   'optimizer_args':{'lr': 0.01,
                                     # 'momentum': 0.5,
                                     'weight_decay': 1e-6}}

    classifier_params = architecture.ClassifierParams(input_dim=dimension,
                                                      output_dim=num_classes,
                                                      hidden_widths=args.nn_hidden_widths,
                                                      activation=nn.ReLU)

    net = architecture.get_net(classifier_params, optim_params)
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

    net_full = architecture.get_net(classifier_params, optim_params)
    dataset_full = dataset.copy()
    strategy_full = get_strategy(args.strategy_name)(dataset_full, net_full)  # load strategy
    strategy_full.label_entire_dataset()

    # start experiment
    accuracies = []
    deviations_from_full = []

    dataset.initialize_labels(args.n_init_labeled)
    # print(f"number of labeled pool: {args.n_init_labeled}")
    # print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    # print(f"number of testing pool: {dataset.n_test}")
    # print()

    print("Full Accuracy")
    strategy_full.train()
    preds_full = strategy_full.predict(dataset_full.get_test_data())
    print()
    print(f"Full testing accuracy: {dataset_full.cal_test_acc(preds_full)}")

    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    accuracy = dataset.cal_test_acc(preds)
    print(f"Round 0 testing accuracy: {accuracy}")
    deviation_from_full = calc_deviation_from_full(preds, preds_full)
    print("Deviation from full: ", deviation_from_full)

    accuracies.append(accuracy)
    deviations_from_full.append(deviation_from_full)

    for rd in range(1, args.n_round+1):
        print(f"Round {rd}")

        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)
        strategy.train()

        dataset_size = len(np.extract(dataset.labeled_idxs == True, dataset.labeled_idxs))
        # print("dataset size: ", dataset_size)

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        accuracy = dataset.cal_test_acc(preds)
        print(f"Round {rd} testing accuracy: {accuracy}")

        deviation_from_full = calc_deviation_from_full(preds, preds_full)
        print("Deviation from full: ", deviation_from_full)

        accuracies.append(accuracy)
        deviations_from_full.append(deviation_from_full)

    return accuracies, deviations_from_full

def get_evaluations_multiple(param_defaults,
                             # distribution: sampling.Distribution, label_distribution: sampling.LabelDistribution,
                             param_overlay_name, param_overlay_options,
                             param_horiz_name, param_horiz_options,
                             param_vert_name, param_vert_options):

    plt.ion()
    fig1, axs1 = plt.subplots(ncols=len(param_horiz_options), nrows=len(param_vert_options))
    fig2, axs2 = plt.subplots(ncols=len(param_horiz_options), nrows=len(param_vert_options))
    # fig.suptitle('Vertically stacked subplots')


    for i_po, param_overlay in enumerate(param_overlay_options):
        for i_pv, param_vert in enumerate(param_vert_options):
            for i_ph, param_horiz in enumerate(param_horiz_options):
                print("="*100)
                param_defaults.__dict__[param_overlay_name] = param_overlay
                param_defaults.__dict__[param_vert_name] = param_vert
                param_defaults.__dict__[param_horiz_name] = param_horiz

                dimension = param_defaults.dimension
                input_range = 1.0
                mins = np.ones(dimension) * -input_range
                maxs = np.ones(dimension) * input_range
                distribution = sampling.FastUniformOnRangeDistribution(range_min=mins, range_max=maxs)
                # distribution = sampling.FastGaussianDistribution(mean=np.array([-0.5, 0]), covar_matrix=np.array([[0.1, 0], [0, 0.1]]))

                # label_distribution = sampling.RandomLabelDistribution()
                # weights = np.zeros(dimension)
                weights = (np.random.rand(dimension) - 0.5) * 5.0
                # weights[0] = 100
                label_distribution = sampling.LinearLabelDistribution(weights=weights)

                accuracies, deviations_from_full = test(param_defaults, distribution, label_distribution)
                axs2[i_pv, i_ph].plot(accuracies, label=str(param_overlay))
                axs2[i_pv, i_ph].title(str(param_horiz_name) + ": " + str(param_horiz) + ", " +
                                       str(param_vert_name) + ": " + str(param_vert_options))
                axs1[i_pv, i_ph].plot(deviations_from_full, label=str(param_overlay))
                axs1[i_pv, i_ph].title(str(param_horiz_name) + ": " + str(param_horiz) + ", " +
                                       str(param_vert_name) + ": " + str(param_vert_options))

                plt.draw()
                plt.pause(0.01)

    plt.show()

if __name__=="__main__":
    get_evaluations_multiple(param_defaults=DEFAULT_ARGS,
                             # distribution=distribution, label_distribution=label_distribution,
                             param_overlay_name=param_overlay_name, param_overlay_options=param_overlay_options,
                             param_horiz_name=param_horiz_name, param_horiz_options=param_horiz_options,
                             param_vert_name=param_vert_name, param_vert_options=param_vert_options)
