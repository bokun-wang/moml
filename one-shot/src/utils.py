import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import random
import os
from config import n_test_episodes, lr_test
from torch.nn import functional as F
from torch.nn.modules import Module
from .model import Model

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

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val

def evaluation_proto_task(task, initial_model, N_way, in_dim, n_shot, n_query, seed=3333):
    """
    trains the model on a random sine task and measures the loss curve.
    for each n in num_steps_measured, records the model function after n gradient updates.
    """
    seed_everything(seed)

    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, N_way))
        ]))
    model.load_state_dict(initial_model.state_dict())
    X, y = task.sample_data(size=(n_shot + n_query))
    y = y.to(torch.int64)

    embeddings = model.forward(X)
    avg_acc = 0.0
    for _ in range(n_test_episodes):
        _ , acc = prototypical_loss(input=embeddings, target=y, n_support=n_shot)
        avg_acc += acc / n_test_episodes

    return avg_acc


def loss_on_random_task(task, initial_model, N_way, in_dim, num_steps, n_shot, n_query, optim=torch.optim.SGD, seed=3333):
    """
    trains the model on a random sine task and measures the loss curve.
    for each n in num_steps_measured, records the model function after n gradient updates.
    """
    seed_everything(seed)

    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, N_way))
        ]))
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimiser = optim(model.parameters(), lr_test)

    avg_acc = 0.0
    avg_loss = 0.0

    for _ in range(n_test_episodes):

        X_te, y_te = task.sample_data(size=n_shot)
        y_te = y_te.to(torch.int64)
        for step in range(1, num_steps + 1):
            outputs = model(X_te)
            loss = criterion(outputs, y_te)

            # compute grad and update inner loop weights
            model.zero_grad()
            loss.backward()
            optimiser.step()

        # evaluation
        X_te, y_te = task.sample_data(size=n_query)
        y_te = y_te.to(torch.int64)
        outputs = model(X_te)
        loss = criterion(outputs, y_te)
        final_loss = loss.cpu().item()
        _, predicted = outputs.max(1)
        correct = predicted.eq(y_te).sum().item()
        total = y_te.size(0)
        avg_loss += final_loss / n_test_episodes
        avg_acc += correct / total / n_test_episodes

    return avg_loss, avg_acc
