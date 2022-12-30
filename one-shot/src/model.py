import torch
import torch.nn as nn
from collections import OrderedDict
import random
import os
import numpy as np
import torch.nn.functional as F

def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Model(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Model, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, n_class))
        ]))

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x