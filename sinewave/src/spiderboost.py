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


class SpiderBoost():
    def __init__(self, model, all_tasks, inner_lr, meta_lr, K=5, N_tasks=3, N1=10, N2=1, q=10,
                 seed=0):

        # B_tasks = N1 or N2

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
        self.N1 = N1
        self.N2 = N2
        self.q = q

        # metrics
        self.plot_every = plot_interv * self.N2 // (self.N1 / self.q + self.N2 * (self.q - 1) / self.q)

    def inner_loop(self, tr_X, tr_y, val_X, val_y, cur_weights):

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in cur_weights]
        # perform training on data sampled from task
        loss = self.criterion(self.model.parameterised(tr_X, temp_weights), tr_y)
        # compute grad and update inner loop weights
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        loss = self.criterion(self.model.parameterised(val_X, temp_weights), val_y)

        return loss

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
        all_samples = []
        all_meta_losses = []
        prev_weights = [w.clone() for w in self.weights]
        meta_grads_buf = []
        acc_samples = 0
        avg_time = 0

        timer = Timer()

        for iteration in range(1, num_iterations + 1):

            timer.start()

            # zero out the gradient of previous iteration
            self.model.zero_grad()

            if (iteration - 1) % self.q == 0:
                # sample a batch of tasks
                sampled_ids = np.random.choice(self.N_tasks, self.N1)

                meta_loss = torch.Tensor([0])
                for id in sampled_ids:
                    # compute meta loss
                    task_i = self.all_tasks[id]
                    tr_X, tr_y = task_i.sample_data(size=self.K)
                    val_X, val_y = task_i.sample_data(size=self.K)
                    meta_loss += self.inner_loop(tr_X, tr_y, val_X, val_y, cur_weights=self.weights) / self.N1

                # compute meta gradient of loss with respect to maml weights
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                acc_samples += self.N1 * self.K
            else:
                # sample a batch of tasks
                sampled_ids = np.random.choice(self.N_tasks, self.N2)
                # zero out the gradient of previous iteration
                self.model.zero_grad()

                meta_loss = torch.Tensor([0])
                for id in sampled_ids:
                    # compute meta loss
                    task_i = self.all_tasks[id]
                    tr_X, tr_y = task_i.sample_data(size=self.K)
                    val_X, val_y = task_i.sample_data(size=self.K)
                    meta_loss += self.inner_loop(tr_X, tr_y, val_X, val_y, cur_weights=self.weights) / self.N2

                # compute meta gradient of loss with respect to current maml weights
                cur_meta_grads = torch.autograd.grad(meta_loss, self.weights)

                meta_loss = torch.Tensor([0])
                for id in sampled_ids:
                    # compute meta loss
                    task_i = self.all_tasks[id]
                    tr_X, tr_y = task_i.sample_data(size=self.K)
                    val_X, val_y = task_i.sample_data(size=self.K)
                    meta_loss += self.inner_loop(tr_X, tr_y, val_X, val_y, cur_weights=prev_weights) / self.N2

                # compute meta gradient of loss with respect to previous maml weights
                prev_meta_grads_ = torch.autograd.grad(meta_loss, prev_weights)
                prev_meta_grads = [g.detach().clone() for g in prev_meta_grads_]

                meta_grads = [g1 + g2 - g3 for g1, g2, g3 in zip(meta_grads_buf, cur_meta_grads, prev_meta_grads)]
                acc_samples += self.N2 * self.K

            meta_grads_buf = [g.detach().clone() for g in meta_grads]

            # assign meta gradient to weights and take optimization step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g

            prev_weights = [w.clone() for w in self.model.parameters()]

            for param in self.model.parameters():
                param.data -= self.meta_lr * param.grad.data

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
                all_samples.append(acc_samples)
                eval_loss = eval_loss.detach().item()
                print("{}/{}. loss: {}".format(iteration, num_iterations, eval_loss))
                all_meta_losses.append(eval_loss)

        avg_time *= 1000.0

        return all_meta_losses, all_samples, avg_time
