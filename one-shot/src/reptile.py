import torch
import random
import torch.nn as nn
import numpy as np
import os
from config import plot_interv


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


class REPTILE():
    def __init__(self, model, all_tasks, inner_lr, meta_lr, n_shot=1, n_query=16, N_tasks=3,
                 seed=0):
        # important objects
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.CrossEntropyLoss()
        self.seed = seed
        seed_everything(self.seed)
        self.all_tasks = all_tasks
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_shot = n_shot
        self.n_query = n_query
        self.N_tasks = N_tasks

        # metrics
        self.plot_every = plot_interv

    def inner_loop(self, X, y):

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        # perform training on data sampled from task
        loss = self.criterion(self.model.parameterised(X, temp_weights), y)
        # compute grad and update inner loop weights
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        return temp_weights

    def main_loop(self, num_iterations):
        seed_everything(self.seed)

        for iteration in range(1, num_iterations + 1):

            if iteration == np.floor(0.75 * num_iterations):
                self.meta_lr /= 10

            # sample a batch of tasks
            sampled_id = np.random.choice(self.N_tasks, 1)[0]

            # zero out the gradient of previous iteration
            self.model.zero_grad()

            before_weights = [w.clone() for w in self.weights]

            avg_diff = [torch.zeros_like(w) for w in self.weights]

            task_i = self.all_tasks[sampled_id]
            X, y = task_i.sample_data(size=(self.n_shot + self.n_query))
            y = y.to(torch.int64)
            temp_weights = self.inner_loop(X, y)
            for i in range(len(avg_diff)):
                avg_diff[i] += (temp_weights[i] - before_weights[i])

            # # compute meta gradient of loss with respect to maml weights
            # meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # # assign meta gradient to weights and take optimization step
            # for w, g in zip(self.weights, meta_grads):
            #     w.grad = g

            for param, diff in zip(self.model.parameters(), avg_diff):
                param.data = param.data + self.meta_lr * diff.data


            # log metrics
            if iteration % self.plot_every == 0:
                print("{}/{}".format(iteration, num_iterations))