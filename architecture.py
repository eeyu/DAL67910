from torch.utils.data import Dataset
from deep_active_learning.data import Data
from sampling import Distribution, LabelDistribution
from deep_active_learning.nets import Net
import torch.nn as nn
import torch.nn.functional as F
import paths
from dataclasses import dataclass
import torch


class GeneralHandler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

def get_randomized_dataset(x_distribution: Distribution, y_distribution: LabelDistribution, num_train, num_test):
    all_x = torch.from_numpy(x_distribution.sample(num_train+num_test)).type(torch.FloatTensor)
    all_y = torch.from_numpy(y_distribution.sample_labels(all_x)).type(torch.LongTensor)
    handler_class = GeneralHandler
    return Data(all_x[:num_train], all_y[:num_train],
                all_x[num_train:], all_y[num_train:],
                handler_class)


@dataclass
class ClassifierParams:
    input_dim: int
    output_dim: int
    hidden_widths: list
    activation: nn.Module.__class__

def get_net(net_parameters, optim_params):
    return Net(ClassifierNet, optim_params, paths.device, net_parameters)

class ClassifierNet(nn.Module):
    def __init__(self, classifier_params: ClassifierParams):
        super(ClassifierNet, self).__init__()

        input_dim = classifier_params.input_dim
        output_dim = classifier_params.output_dim
        hidden_widths = classifier_params.hidden_widths
        self.activation = classifier_params.activation

        layers = []
        last_layer_dim = input_dim

        for i in range(len(hidden_widths)): # Loop over layers, adding conv2d, layernorm, and relu.
            # print(last_layer_dim)
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=last_layer_dim, out_features=hidden_widths[i]),
                    self.activation()
                )
            )
            last_layer_dim = hidden_widths[i]
        # print(last_layer_dim)
        self.layers = nn.ModuleList(layers)
        self.last_layer_dim = last_layer_dim
        self.last_layer = nn.Linear(last_layer_dim, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # summary(self.to(paths.device), input_size=(10, 30))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        e1 = x
        x = F.dropout(e1, training=self.training)
        y = self.last_layer(x)
        return y, e1

    def get_embedding_dim(self):
        return self.last_layer_dim


@dataclass
class Hyperparameters:
    nn_hidden_widths: list
    seed: int = 1
    n_init_labeled: int = 10000
    n_query: int = 1000
    n_round: int = 10
    num_train: int = 100000
    num_test: int = int(100000 / 4)
    dimension: int = 30
    num_classes: int = 2
    strategy_name: str = "LeastConfidence"
    # TODO do not edit this!!



