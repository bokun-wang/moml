import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# put this in the main file
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
#
# root = '../data'
# if not os.path.exists(root):
#     os.mkdir(root)
#
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# # if not exist, download mnist dataset
# train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
# test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

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

class Classification_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, task_list, data, targets, N_way):
        """
        selected_data: n_way x data_per_class x
        """
        self.data = [] # list of size N_way, each element N_i x 784
        self.targets = [] # list of N_way, each element N_i  (note: this is the target within each n-way)
        self.in_dim = data.shape[-1]
        for i, task in enumerate(task_list):
            idx = (targets == task).nonzero(as_tuple=True)[0]
            self.data.append(data[idx, :])
            self.targets.append([i] * len(idx))
        self.N_way = N_way

        # # WARNING: the following code only works for the case that all classes have the same number of data points
        # self.flatten_data = torch.empty((N_way * len(self.targets[0]), self.data[0].shape[-1]))
        # self.flatten_targets = torch.empty(N_way * len(self.targets[0]))
        # for i in range(N_way):
        #     self.flatten_data[(i * len(self.targets[i])):((i + 1) * len(self.targets[i])), :] = self.data[i]
        #     self.flatten_targets[(i * len(self.targets[i])):((i + 1) * len(self.targets[i]))] = torch.LongTensor(self.targets[i])
        
    def sample_data(self, size=1):
        """
        Sample data from N_way tasks
        
        returns: 
            x: list of size N_way, each element size x 784
            y: list of N_way, each element B_i
        """
        N_way = self.N_way
        batch_data = torch.empty((N_way * size, self.in_dim))
        batch_targets = torch.empty(N_way * size)
        # sample K data for each class
        for i in range(N_way):
            sampled_ids = list(np.random.choice(len(self.targets[i]), size))
            for j, id in enumerate(sampled_ids):
                batch_data[(i * size + j), :] = self.data[i][id, :]
                batch_targets[(i * size + j)] = self.targets[i][id]
        
        return batch_data, batch_targets


