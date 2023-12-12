from torch.utils.data import Dataset
from deep_active_learning.data import Data
from sampling import Distribution, LabelDistribution
from deep_active_learning.nets import Net
import torch.nn as nn
import torch.nn.functional as F
import paths
from dataclasses import dataclass
import torch
from torchsummary import summary

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


def get_fcc_net(net_parameters, optim_params):
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
class CNNParams:
    input_dim: tuple # ex 32x32x3
    output_dim: int

    fc_dims: list
    channel_dims: list

    conv_kernel: int
    pool_kernel: int

def get_cnn_net(net_parameters: CNNParams, optim_params):
    return Net(CNNNet, optim_params, paths.device, net_parameters)

class CNNNet(nn.Module):
    def __init__(self, classifier_params: CNNParams):
        super(CNNNet, self).__init__()

        input_channels = classifier_params.input_dim[0]
        output_dim = classifier_params.output_dim

        channel_dims = classifier_params.channel_dims
        conv_layers = []
        last_channel_dim = input_channels
        image_width = classifier_params.input_dim[1]

        conv_size = classifier_params.conv_kernel
        pool_size = classifier_params.pool_kernel

        for i in range(len(channel_dims)): # Loop over layers, adding conv2d, layernorm, and relu.
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(last_channel_dim, channel_dims[i], kernel_size=conv_size),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size, stride=1)
                )
            )
            image_width = image_width - (conv_size-1) - (pool_size-1)
            last_channel_dim = channel_dims[i]

        # Dropout here

        self.cnn_output_dims = image_width * image_width * last_channel_dim
        fc_dims = classifier_params.fc_dims
        fc_layers = []
        last_fc_dims = self.cnn_output_dims
        for i in range(len(fc_dims)): # Loop over layers, adding conv2d, layernorm, and relu.
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=last_fc_dims, out_features=fc_dims[i]),
                    nn.ReLU(),
                )
            )
            last_fc_dims = fc_dims[i]

        # Dropout here

        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc_layers = nn.ModuleList(fc_layers)

        self.embedding_dims = last_fc_dims
        self.last_layer = nn.Linear(last_fc_dims, output_dim)

        # print(summary(self.to(paths.device),
        #               input_size=classifier_params.input_dim
        #               # input_size=classifier_params.input_dim
        #               ))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = F.dropout2d(x)
        x = x.view(-1, self.cnn_output_dims)
        for layer in self.fc_layers:
            x = layer(x)
        e1 = x
        x = F.dropout(e1, training=self.training)
        y = self.last_layer(x)
        return y, e1

    def get_embedding_dim(self):
        return self.embedding_dims




