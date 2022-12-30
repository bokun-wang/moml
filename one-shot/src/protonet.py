import torch
import random
import torch.nn as nn
import numpy as np
import os
from config import plot_interv
from .utils import prototypical_loss

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

class ProtoNet():
    def __init__(self, model, all_tasks, in_dim, inner_lr, meta_lr, n_shot=1, n_query=16, N_tasks=3, seed=0):
        # important objects
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.CrossEntropyLoss()
        self.seed = seed
        seed_everything(self.seed)
        self.all_tasks = all_tasks
        self.in_dim = in_dim
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_shot = n_shot
        self.n_query = n_query
        self.N_tasks = N_tasks
        # metrics
        self.plot_every = plot_interv

    def main_loop(self, num_iterations):
        seed_everything(self.seed)

        for iteration in range(1, num_iterations + 1):

            if iteration == np.floor(0.5 * num_iterations):
                self.meta_lr /= 10

            # sample a batch of tasks
            sampled_id = np.random.choice(self.N_tasks, 1)[0]

            # zero out the gradient of previous iteration
            self.model.zero_grad()


            # compute meta loss
            task_i = self.all_tasks[sampled_id]
            X, y = task_i.sample_data(size=(self.n_shot + self.n_query))
            y = y.to(torch.int64)

            embeddings = self.model.forward(X)
            meta_loss, _ = prototypical_loss(input=embeddings, target=y, n_support=self.n_shot)

            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimization step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g

            for param in self.model.parameters():
                param.data -= self.meta_lr * param.grad.data

            # log metrics
            if iteration % self.plot_every == 0:
                print("{}/{}".format(iteration, num_iterations))

