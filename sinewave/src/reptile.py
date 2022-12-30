import torch
import random
import torch.nn as nn
import numpy as np
import os
from src.my_timer import Timer
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


class Reptile():
    def __init__(self, model, all_tasks, inner_lr, meta_lr, K=5, N_tasks=3, B_tasks=3,
                 seed=0):
        # important objects
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.seed = seed
        seed_everything(self.seed)
        self.all_tasks = all_tasks
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.N_tasks = N_tasks
        self.B_tasks = B_tasks

        # metrics
        self.plot_every = plot_interv

    def inner_loop(self, tr_X, tr_y, val_X, val_y):

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        # perform training on data sampled from task
        loss = self.criterion(self.model.parameterised(tr_X, temp_weights), tr_y)
        # compute grad and update inner loop weights
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # # sample new data for meta-update and compute loss
        # loss = self.criterion(self.model.parameterised(val_X, temp_weights), val_y)

        return temp_weights

    def inner_loop_eval(self, tr_X, tr_y, val_X, val_y):

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        # perform training on data sampled from task
        loss = self.criterion(self.model.parameterised(tr_X, temp_weights), tr_y)
        # compute grad and update inner loop weights
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        loss = self.criterion(self.model.parameterised(val_X, temp_weights), val_y)

        return loss

    def main_loop(self, num_iterations):
        seed_everything(self.seed)
        all_iters = []
        all_meta_losses = []
        avg_time = 0

        timer = Timer()

        for iteration in range(1, num_iterations + 1):

            if iteration == np.floor(0.75 * num_iterations):
                self.meta_lr /= 10

            timer.start()

            # sample a batch of tasks
            sampled_ids = np.random.choice(self.N_tasks, self.B_tasks, replace=False)

            # zero out the gradient of previous iteration
            self.model.zero_grad()

            before_weights = [w.clone() for w in self.weights]

            avg_diff = [torch.zeros_like(w) for w in self.weights]
            for id in sampled_ids:
                # compute meta loss
                task_i = self.all_tasks[id]
                tr_X, tr_y = task_i.sample_data(size=self.K)
                val_X, val_y = task_i.sample_data(size=self.K)
                temp_weights = self.inner_loop(tr_X, tr_y, val_X, val_y)
                for i in range(len(avg_diff)):
                    avg_diff[i] += (temp_weights[i] - before_weights[i]) / self.B_tasks

            # # compute meta gradient of loss with respect to maml weights
            # meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # # assign meta gradient to weights and take optimization step
            # for w, g in zip(self.weights, meta_grads):
            #     w.grad = g

            for param, diff in zip(self.model.parameters(), avg_diff):
                param.data = param.data + self.meta_lr * diff.data

            avg_time += timer.stop() / num_iterations

            # log metrics
            if iteration % self.plot_every == 0:
                eval_loss = torch.Tensor([0])
                for id in range(self.N_tasks):
                    # compute meta loss
                    task_i = self.all_tasks[id]
                    tr_X, tr_y = task_i.sample_data(size=100)
                    val_X, val_y = task_i.sample_data(size=100)
                    eval_loss += self.inner_loop_eval(tr_X, tr_y, val_X, val_y) / self.N_tasks
                all_iters.append(iteration)
                eval_loss = eval_loss.detach().item()
                print("{}/{}. loss: {}".format(iteration, num_iterations, eval_loss))
                all_meta_losses.append(eval_loss)

        avg_time *= 1000.0

        return all_meta_losses, all_iters, avg_time
