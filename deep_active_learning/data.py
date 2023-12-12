import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.input_shape = X_train.shape[1,:]
        shape = X_train.shape
        if len(shape) == 3:
            self.input_shape = (1, X_train.shape[1], X_train.shape[2])
        else:
            self.input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        # self.input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        self.output_dims = torch.max(Y_train) + 1

    def copy(self):
        return Data(self.X_train, self.Y_train, self.X_test, self.Y_test, self.handler)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def cal_train_acc(self, preds):
        return 1.0 * (self.Y_train==preds).sum().item() / self.n_pool

    
def get_MNIST(handler, train_size=40000, test_size=40000):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:train_size], raw_train.targets[:train_size],
                raw_test.data[:test_size], raw_test.targets[:test_size], handler)

def get_FashionMNIST(handler, train_size=40000, test_size=40000):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:train_size], raw_train.targets[:train_size],
                raw_test.data[:test_size], raw_test.targets[:test_size], handler)

def get_SVHN(handler, train_size=40000, test_size=40000):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:train_size], torch.from_numpy(data_train.labels)[:train_size],
                data_test.data[:test_size], torch.from_numpy(data_test.labels)[:test_size], handler)

def get_CIFAR10(handler, train_size=40000, test_size=40000):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:train_size], torch.LongTensor(data_train.targets)[:train_size],
                data_test.data[:test_size], torch.LongTensor(data_test.targets)[:test_size], handler)
