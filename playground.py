import numpy as np
from deep_active_learning import (get_strategy)
import paths
import sampling
import architecture
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt

def calc_deviation_from_full(preds, preds_full):
    return 1.0 - 1.0 * (preds_full == preds).sum().item() / len(preds_full)

@dataclass
class Hyperparameters:
    seed: int = 1
    n_init_labeled: int = 10000
    n_query: int = 1000
    n_round: int = 10
    num_train: int = 100000
    num_test: int = int(100000 / 4)
    dimension: int = 30
    num_classes: int = 2
    strategy_name: str = "LeastConfidence"


if __name__=="__main__":
    args = Hyperparameters(seed=100,
                           n_init_labeled=1000,
                           n_query=1000,
                           n_round=10,
                           num_train = 100000,
                           num_test = int(100000/4),
                           dimension = 30,
                           num_classes = 2,
                           strategy_name="EntropySamplingDropout")
    device = paths.device

    num_train = args.num_train
    num_test = args.num_test
    dimension = args.dimension
    num_classes = args.num_classes

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
                                                      hidden_widths=[10],
                                                      activation=nn.ReLU)

    # net = get_net(args.dataset_name, device)                   # load network
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
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    print("Full Accuracy")
    strategy_full.train()
    preds_full = strategy_full.predict(dataset_full.get_test_data())
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
        print("dataset size: ", dataset_size)

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        accuracy = dataset.cal_test_acc(preds)
        print(f"Round {rd} testing accuracy: {accuracy}")

        deviation_from_full = calc_deviation_from_full(preds, preds_full)
        print("Deviation from full: ", deviation_from_full)

        accuracies.append(accuracy)
        deviations_from_full.append(deviation_from_full)

    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(accuracies)
    axs[1].plot(deviations_from_full)
    plt.show()