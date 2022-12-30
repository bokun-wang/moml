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

class LCMOML():
    def __init__(self, model, all_tasks, inner_lr, meta_lr, K=5, N_tasks=3, B_tasks=3,
                 seed=0, beta=0.9, K0=10, H=10):

        # important objects
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimizing
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
        self.beta = beta
        self.K0 = K0
        self.H = H

        # metrics
        self.plot_every = plot_interv // self.H

    def inner_loop(self, tr_X, tr_y, val_X, val_y, per_model, weights_id):

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in weights_id]

        # perform training on data sampled from task
        loss = self.criterion(self.model.parameterised(tr_X, temp_weights), tr_y)
        # compute grad and update inner loop weights
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # momentum step
        temp_weights = [self.beta * temp_w + (1 - self.beta) * per_w for temp_w, per_w in zip(temp_weights, per_model)]
        per_model = [w.clone().detach() for w in temp_weights]

        # sample new data for meta-update and compute loss
        loss = self.criterion(self.model.parameterised(val_X, temp_weights), val_y)

        return loss, per_model

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
        N_epoch = num_iterations // self.H
        avg_time = 0

        timer = Timer()

        for epoch in range(1, N_epoch + 1):

            if epoch == np.floor(0.75 * N_epoch):
                self.meta_lr /= 10

            timer.start()
            epoch_weights = [torch.zeros_like(w.clone()) for w in self.weights]
            # sample a batch of tasks
            sampled_ids = np.random.choice(self.N_tasks, self.B_tasks)

            for id in sampled_ids:
                weights_id = [w.clone() for w in self.weights]
                task_id = self.all_tasks[id]
                self.model.zero_grad()
                tr0_X, tr0_y = task_id.sample_data(size=self.K0)
                loss = self.criterion(self.model.parameterised(tr0_X, weights_id), tr0_y)
                grad = torch.autograd.grad(loss, weights_id)
                per_model = [w.clone().detach() - self.inner_lr * g.clone().detach() for w, g in
                             zip(weights_id, grad)]

                for iter in range(1, self.H + 1):

                    # zero out the gradient of previous iteration
                    self.model.zero_grad()

                    tr_X, tr_y = task_id.sample_data(size=self.K)
                    val_X, val_y = task_id.sample_data(size=self.K)

                    meta_loss, per_model = self.inner_loop(tr_X, tr_y, val_X, val_y, per_model, weights_id)

                    # compute meta gradient of loss with respect to maml weights
                    meta_grads = torch.autograd.grad(meta_loss, weights_id)

                    # assign meta gradient to weights and take optimization step
                    for w_id, g in zip(weights_id, meta_grads):
                        w_id.grad = g

                    for w_id in weights_id:
                        w_id.data -= self.meta_lr * w_id.grad.data

                epoch_weights = [(w_epoch + w_id/self.B_tasks) for w_epoch, w_id in zip(epoch_weights, weights_id)]

            for w, w_epoch in zip(self.weights, epoch_weights):
                w.data = w_epoch.data

            avg_time += timer.stop() / num_iterations

            # log metrics
            if epoch % self.plot_every == 0:
                eval_loss = torch.Tensor([0])
                for id in range(self.N_tasks):
                    # compute meta loss
                    task_i = self.all_tasks[id]
                    tr_X, tr_y = task_i.sample_data(size=100)
                    val_X, val_y = task_i.sample_data(size=100)
                    eval_loss += self.inner_loop_eval(tr_X, tr_y, val_X, val_y) / self.N_tasks
                all_iters.append(epoch * self.H)
                eval_loss = eval_loss.detach().item()
                print("{}/{}. loss: {}".format(epoch * self.H, num_iterations, eval_loss))
                all_meta_losses.append(eval_loss)

        avg_time *= 1000.0

        return all_meta_losses, all_iters, avg_time