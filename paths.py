import argparse
import numpy as np
import torch

# fix random seed
np.random.seed(100)
torch.manual_seed(100)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# print(device)