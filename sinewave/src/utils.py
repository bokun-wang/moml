import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.tasks import Sine_Task, Sine_Task_Distribution
import random
import os

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


def loss_on_random_task(task, initial_model, K, num_steps, X_plot, y_plot, X_te, y_te, optim=torch.optim.SGD, seed=3333):
    """
    trains the model on a random sine task and measures the loss curve.

    for each n in num_steps_measured, records the model function after n gradient updates.
    """
    seed_everything(seed)

    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ]))
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), 0.01)

    # train model on a random task
    # X, y = task.sample_data(K)
    losses = []
    for step in range(1, num_steps + 1):
        loss = criterion(model(X_te), y_te)
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    # curve fitting after adaptation
    predicted = model.forward(X_plot).detach()
    final_loss = criterion(model(X_plot), y_plot).detach().numpy()

    return final_loss, predicted
