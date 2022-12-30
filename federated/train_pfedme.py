import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import copy
import os, re, time
import pandas as pd
# from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
from my_timer import Timer

import builtins

# from parameters import para
from models import NeuralNetwork as dnn
from mnist10 import MNIST_MAML
from cifar10 import CIFAR10_MAML
from cifar100 import CIFAR100_MAML
# from tensorflow import keras
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from sparse2coarse import sparse2coarse

import torch.distributed as dist
import argparse
import datetime

config = dict(
    distributed_backend='nccl',
    distributed_init_file=None,
    rank=0,
    n_workers=1,
    local_rank=0,
    local_world_size=1,
    image_size=32,
    local_batchsize=40,
    seed=123,
    model_name='dnn',
    dataset='cifar10',
    update_lr=0.01,
    meta_lr=0.01,
    weight_decay=1e-5,
    ft_lr=0.001,
    H=4,
    optimizer='SGD',
    test_freq=100,
    test_batchsize=32,
    test_batches=100,
    save_freq=10000,
    numGPU=1,
    total_iters=1000,
    total_epochs=1000,
    n_way=1,
    k_spt=1,
    k_qry=10,
    order=1,
    local_steps=20,
    lam=20,
    num_tasks=4,
    update_step_test=10,
)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    output_dir = "./output"
    seed = int(config["seed"])
    rank = int(config["rank"])
    size = int(config["n_workers"])
    set_all_seeds(seed + rank)

    if torch.distributed.is_available():
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        print(
            "Distributed init: rank {}/{} - {}".format(
                config["rank"], size, config["distributed_init_file"]
            )
        )
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(seconds=120),
            world_size=size,
            rank=rank,
        )

    # parameters
    dataset = config['dataset']  # 'cifar10'
    update_lr = config['update_lr']  # 0.01  # inside learner
    meta_lr = config['meta_lr']  # s  0.01   # meta learner
    N = 50
    # fixed
    update_step_test = config['update_step_test']
    eta_ft = config['ft_lr']  # 0.01
    n_way = config['n_way']  # 1
    k_shot = config['k_spt']  # 5
    k_query = config['k_qry']  # 10

    if dataset == 'mnist':
        num_classes = 10
        meta_data = np.load('mnist_data.npz')
        train_data, train_labels, x_test, y_test = meta_data['train_data'], meta_data['train_labels'], meta_data[
            'x_test'], meta_data['y_test']
        trainSet = MNIST_MAML(train_data, train_labels, mode='train', N=50, a=168, batchsz=1000, n_way=n_way,
                              k_shot=k_shot, k_query=k_query, image_size=28, size=size, num_classes=num_classes,
                              gpu_rank=rank)
        testSet = MNIST_MAML(x_test, y_test, mode='test', N=50, a=34, batchsz=50, n_way=n_way, k_shot=k_shot,
                             k_query=k_query, image_size=28, num_classes=num_classes, gpu_rank=rank)
    elif dataset == 'cifar10':
        num_classes = 10
        meta_data = np.load('cifar10_data.npz')
        train_data, train_labels, x_test, y_test = meta_data['train_data'], meta_data['train_labels'], meta_data[
            'x_test'], meta_data['y_test']
        trainSet = CIFAR10_MAML(train_data, train_labels, mode='train', N=50, a=68, batchsz=10000, n_way=n_way,
                                k_shot=k_shot, k_query=k_query, image_size=32, size=size, num_classes=num_classes,
                                gpu_rank=rank)
        testSet = CIFAR10_MAML(x_test, y_test, mode='test', N=50, a=34, batchsz=50, n_way=n_way, k_shot=k_shot,
                               k_query=k_query, image_size=32, num_classes=num_classes, gpu_rank=rank)
    elif dataset == 'cifar100':
        num_classes = 20
        meta_data = np.load('cifar100_data.npz')
        train_data, train_labels, x_test, y_test = meta_data['train_data'], meta_data['train_labels'], meta_data[
            'x_test'], meta_data['y_test']
        trainSet = CIFAR100_MAML(train_data, train_labels, mode='train', N=50, a=68, batchsz=10000, n_way=n_way,
                                 k_shot=k_shot, k_query=k_query, image_size=32, size=size, num_classes=num_classes,
                                 gpu_rank=rank)
        testSet = CIFAR100_MAML(x_test, y_test, mode='test', N=50, a=15, batchsz=50, n_way=n_way, k_shot=k_shot,
                                k_query=k_query, image_size=32, num_classes=num_classes, gpu_rank=rank)
    else:
        raise ValueError

    trainloader = torch.utils.data.DataLoader(trainSet, config['num_tasks'], shuffle=True, num_workers=2,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testSet, 1, shuffle=False, num_workers=2, pin_memory=True)

    inplanes = {'cifar10': 32 * 32 * 3, 'cifar100': 32 * 32 * 3, 'mnist': 28 * 28}[dataset]
    image_size = {'cifar10': 32, 'cifar100': 32, 'mnist': 28}[dataset]

    # datasets
    datetime_now = '2021-05-26'
    configs = '[%s]Train_%s_N_%s_KS_%s_KQ_%s_%s_wd_%s_ulr_%s_mlr_%s_ftlr_%s_ftSP_%s_B_%s_IMG_%s_CE_%s_H_%s_LocalSteps_%s_TS_%s_E_%s_GPU_%s_S_%d_C%d' % (
        datetime_now, dataset, n_way, k_shot, k_query, config['model_name'], config['weight_decay'],
        config['update_lr'],
        config['meta_lr'], eta_ft, update_step_test, config['local_batchsize'], image_size, config['optimizer'],
        config['H'], config['local_steps'], config['num_tasks'], config['total_iters'], size, config['seed'],
        num_classes)
    SAVE_LOG_PATH = '/scratch/user/bokun-wang/logs/'

    model = dnn(in_planes=inplanes, hidden_size=[40, 40], num_classes=num_classes, last_activation=None)
    model = model.cuda()
    model_names = [name for name, v in model.named_parameters()]
    model_pools = [[layer.detach().cpu().numpy() for layer in model.parameters()] for n in range(N)]

    if int(config['local_rank']) == 0 and rank == 0:
        print(configs)
        init_weights = [w.data.cpu().clone() for w in list(model.parameters())]
        print('Init weights:', init_weights[0].numpy().sum())
        print('-' * 100)

    step = 0
    start_time = time.time()
    train_acc_list, test_acc_list, train_loss_list, test_loss_list = [], [], [], []

    if rank == 0:
        timer = Timer()
    else:
        timer = None

    n_epochs = int(config['total_iters'] // len(trainloader))

    for epoch in range(n_epochs):
    # for epoch in range(config['total_epochs']):
        total_time = 0.0

        # if step >= config['total_iters']:
        #     # os.system('pkill python')
        #     break

        for _, (x_spt, y_spt, x_qry, y_qry, uid) in enumerate(trainloader):

            if rank == 0:
                timer.start()

            # if step >= config['total_iters']:
            #     # os.system('pkill python')
            #     break

            if step == int(config['total_iters'] * 0.5) or step == int(config['total_iters'] * 0.75):
                config['update_lr'] /= 10
                config['meta_lr'] /= 10
                # args.ft_lr =/= 10

            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

            # local updates
            iter_train_acc_list = []
            iter_loss_list = []
            for i in range(config['num_tasks']):  # n ways ; need to maintain #task models
                model_weigths = [torch.FloatTensor(m).cuda() for m in model_pools[uid[i].numpy()[0]]]
                model_dicts = dict(zip(model_names, model_weigths))
                model.load_state_dict(model_dicts)

                for h in range(config['H']):
                    x_batch = torch.cat((x_spt[i], x_qry[i]), 0)
                    y_batch = torch.cat((y_spt[i], y_qry[i]), 0)
                    # logits = model(x_spt[i])
                    logits = model(x_batch)
                    # loss = F.cross_entropy(logits, y_spt[i].reshape(-1))
                    loss = F.cross_entropy(logits, y_batch.reshape(-1))
                    w_grads = torch.autograd.grad(loss, model.parameters())

                    per_model = [weight.detach().clone() for weight in model.parameters()]

                    for t in range(config['local_steps']):
                        per_model = [localweight.data - update_lr * (grad.data + config['lam'] * (localweight.data - p.data)) for
                                     (localweight, p, grad) in zip(per_model, model.parameters(), w_grads)]

                    for (p, localweight) in zip(model.parameters(), per_model):
                        p.data = p.data - config['lam'] * meta_lr * (p.data - localweight.data)

                    iter_loss_list.append(loss.item())

                    with torch.no_grad():
                        logits_q = model(x_qry[i])
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # convert to numpy
                        train_acc = np.sum(y_qry[i].cpu().numpy() == pred_q.cpu().numpy()) / len(y_qry[i].cpu().numpy())
                        iter_train_acc_list.append(train_acc)

                model_pools[uid[i].numpy()[0]] = [layer.detach().cpu().numpy() for layer in model.parameters()]

            # communicate with other servers after H local steps (only average sampled tasks!!!)
            size = float(dist.get_world_size())
            sampled_model_pools = [[torch.FloatTensor(m).cuda() for m in p_model] for idx, p_model in
                                   enumerate(model_pools) if
                                   idx in uid.numpy().flatten().tolist()]  # [uid.numpy().flatten()]]
            for idx, p_model in enumerate(sampled_model_pools):
                new_p_model = []
                for layer in p_model:
                    dist.all_reduce(layer.data, op=dist.ReduceOp.SUM)
                    layer.data /= size
                    new_p_model.append(layer)
                sampled_model_pools[idx] = new_p_model

            sampled_model_pools = [[layer.detach().cpu().numpy() for layer in p_model] for p_model in
                                   sampled_model_pools]
            model_user_average = [np.zeros(w.shape) for w in model_pools[0]]
            for idx, w_numpy in enumerate(model_user_average):
                for n in range(len(sampled_model_pools)):
                    w_numpy += sampled_model_pools[n][idx]
                w_numpy /= len(sampled_model_pools)
                model_user_average[idx] = w_numpy
            model_pools = [model_user_average] * N

            if rank == 0:
                total_time += timer.stop()

            # evaluation
            # if step % 50 == 0 and int(config['local_rank']) == 0 and rank == 0:
            if step % 50 == 0 and rank == 0:
                # # finetune
                iter_test_acc_list = []
                for x_spt, y_spt, x_qry, y_qry, uid in testloader:
                    x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                    for i in range(1):

                        model_weigths = [torch.FloatTensor(m).cuda() for m in model_pools[uid[i].numpy()[0]]]
                        model_dicts = dict(zip(model_names, model_weigths))
                        model.load_state_dict(model_dicts)

                        for k in range(update_step_test):
                            logits = model(x_spt[i])
                            loss = F.cross_entropy(logits, y_spt[i].reshape(-1))
                            w_grads = torch.autograd.grad(loss, model.parameters())
                            for var, grad in zip(model.parameters(), w_grads):
                                var.data = var.data - eta_ft * grad.data

                            with torch.no_grad():
                                logits_q = model(x_qry[i])
                                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                                test_acc = np.sum(y_qry[i].cpu().numpy() == pred_q.cpu().numpy()) / len(
                                    y_qry[i].cpu().numpy())
                                iter_test_acc_list.append(test_acc)

                print('step:%s, train_loss:%.3f, train_acc:%.3f, test_acc:%3f, time:%.4f' % (
                step, np.mean(iter_loss_list), np.mean(iter_train_acc_list), np.mean(iter_test_acc_list),
                time.time() - start_time))
                start_time = time.time()
                train_acc_list.append(np.mean(iter_train_acc_list))
                test_acc_list.append(np.mean(iter_test_acc_list))
                train_loss_list.append(np.mean(iter_loss_list))
                print('Best Train Acc: %.4f' % max(train_acc_list), 'Best test Acc: %.4f' % max(test_acc_list))

                df = pd.DataFrame(data={
                    'train_acc_H_%s_k_%s' % (config['H'], config['num_tasks']): train_acc_list,
                    'train_acc_H_%s_k_%s' % (config['H'], config['num_tasks']): train_acc_list,
                    'val_acc_H_%s_k_%s' % (config['H'], config['num_tasks']): test_acc_list})
                df.to_csv(SAVE_LOG_PATH + '%s.csv' % (configs))

                # if step == config['total_iters'] + 1:
                #     break

            step += 1

        if rank == 0:
            print('Average Time: %.4f' % (total_time * 1000.0 / len(trainloader)))

    # # if int(config['local_rank']) == 0 and rank == 0:
    # if rank == 0:
    #     print('Best Train Acc: %.3f' % max(train_acc_list), 'Best test Acc: %.3f' % max(test_acc_list))


if __name__ == "__main__":
    main()

