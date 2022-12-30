import numpy as np
import torch
from src import Model, MAML, MOML, NASA, LCMOML, MOMLV2, REPTILE, ProtoNet
from src.tasks import Classification_Task
from src.utils import loss_on_random_task, seed_everything, evaluation_proto_task
import argparse
from config import *
from itertools import combinations
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


def train(args):
    # load cifar data
    if args.dataset == 'CIFAR-100':
        total_iterations = 2000
        N_way = 5
        test_iter = 1
        data = torch.load('./data/cifar100_data.pt')
        targets = torch.load('./data/cifar100_targets.pt')
        # generate cifar tasks
        seed_everything(0)
        shuffled_tasks = torch.randperm(100).tolist()
        n_train_tasks = 85 // N_way
        n_test_tasks = 15 // N_way
        all_train_tasks_id = [shuffled_tasks[(N_way * i):(N_way * (i + 1))] for i in range(n_train_tasks)]
        all_test_tasks_id = [shuffled_tasks[(N_way * (n_train_tasks + i)):(N_way * (n_train_tasks + i + 1))] for i in
                         range(n_test_tasks)]
        all_train_tasks = [Classification_Task(task_list, data, targets, N_way) for task_list in all_train_tasks_id]
        all_test_tasks = [Classification_Task(task_list, data, targets, N_way) for task_list in all_test_tasks_id]

    elif args.dataset == 'Omniglot':
        # load omniglot
        total_iterations = 500
        N_way = 20
        test_iter = 20
        data = torch.load('./data/omniglot_train_data.pt')
        targets = torch.load('./data/omniglot_train_targets.pt')
        test_data = torch.load('./data/omniglot_test_data.pt')
        test_targets = torch.load('./data/omniglot_test_targets.pt')
        n_train_tasks = 25
        n_test_tasks = 10
        seed_everything(0)
        all_train_tasks_id = [list(np.arange(N_way * i, N_way * (i + 1))) for i in range(n_train_tasks)]
        all_test_tasks_id = [list(np.arange(N_way * i, N_way * (i + 1))) for i in range(n_test_tasks)]
        all_train_tasks = [Classification_Task(task_list, data, targets, N_way) for task_list in all_train_tasks_id]
        all_test_tasks = [Classification_Task(task_list, test_data, test_targets, N_way) for task_list in all_test_tasks_id]
    else:
        raise ValueError('Unknown dataset!')

    # initialize the logs
    final_te_loss = []
    final_te_acc = []

    for seed in seeds:
        model = Model(in_dim=data.shape[-1], n_class=N_way)
        print("==================Seed: {0}, Algorithm: {1}==================".format(seed, args.alg))
        if args.alg == 'MAML':
            loss = MAML(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr, n_shot=args.n_shot,
                        n_query=args.n_query,
                        N_tasks=n_train_tasks, seed=seed)
            loss.main_loop(num_iterations=total_iterations)
        elif args.alg == 'ProtoNet':
            loss = ProtoNet(model, all_train_tasks, in_dim=data.shape[-1], inner_lr=lr_i, meta_lr=args.lr,
                            n_shot=args.n_shot, n_query=args.n_query,
                            N_tasks=n_train_tasks, seed=seed)
            loss.main_loop(num_iterations=total_iterations)
        elif args.alg == 'Reptile':
            loss = REPTILE(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr, n_shot=args.n_shot,
                          n_query=args.n_query, N_tasks=n_train_tasks, seed=seed)
            loss.main_loop(num_iterations=total_iterations)
        elif args.alg == 'MOML':
            loss = MOML(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr, n_shot=args.n_shot,
                        n_query=args.n_query, N_tasks=n_train_tasks, seed=seed, beta=args.beta)
            loss.main_loop(num_iterations=total_iterations)
        elif args.alg == 'MOML-V2':
            loss = MOMLV2(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr, n_shot=args.n_shot,
                          n_query=args.n_query, N_tasks=n_train_tasks, seed=seed, beta=args.beta)
            loss.main_loop(num_iterations=total_iterations)
        elif args.alg == 'NASA':
            loss = NASA(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr, n_shot=args.n_shot,
                        n_query=args.n_query, N_tasks=n_train_tasks, seed=seed, beta=args.beta,
                        grad_mom=args.grad_mom)
            loss.main_loop(num_iterations=int((total_iterations) / n_train_tasks))
        elif args.alg == 'LocalMOML':
            ratio = (args.n_shot + (args.n_shot + args.n_query) * H) / ((args.n_shot + args.n_query) * H)
            loss = LCMOML(model, all_train_tasks, inner_lr=lr_i, meta_lr=args.lr,
                          n_shot=args.n_shot, n_query=args.n_query, N_tasks=n_train_tasks, seed=seed,
                          beta=args.beta, H=H)
            loss.main_loop(num_iterations=int(total_iterations // ratio))
        else:
            loss = None
            raise ValueError("Unknown algorithm!")

        # test
        if args.alg == 'ProtoNet':
            avg_te_acc = 0.0
            avg_te_loss = 0.0
            for i, te_task in enumerate(all_test_tasks):
                acc_per_task = evaluation_proto_task(task=te_task, initial_model=loss.model.model, N_way=N_way, in_dim=data.shape[-1],
                                                     n_shot=args.n_shot, n_query=args.n_query)
                avg_te_acc += acc_per_task / n_test_tasks
        else:
            avg_te_loss = 0.0
            avg_te_acc = 0.0
            for i, te_task in enumerate(all_test_tasks):
                te_loss_per_task, acc_per_task = loss_on_random_task(task=te_task, initial_model=loss.model.model, N_way=N_way,
                                                                     in_dim=data.shape[-1], num_steps=test_iter,
                                                                     n_shot=args.n_shot, n_query=args.n_query)
                avg_te_loss += te_loss_per_task / n_test_tasks
                avg_te_acc += acc_per_task / n_test_tasks

        ################################################
        final_te_loss.append(avg_te_loss)
        final_te_acc.append(avg_te_acc)

    final_te_loss = np.array(final_te_loss).flatten()
    final_te_acc = np.array(final_te_acc).flatten() * 100.0
    file_name = './results/{}_{}_traces.npz'.format(args.alg, args.n_shot)
    np.savez(file_name, final_te_loss=final_te_loss, final_te_acc=final_te_acc)

    if args.alg == 'ProtoNet':
        print("{0}: final test accuracy (percent): avg:{1}, std:{2}".format(args.alg,np.mean(final_te_acc),np.std(final_te_acc)))
    else:
        print("{0}: final test accuracy (percent): avg:{1}, std:{2}".format(args.alg, np.mean(final_te_acc), np.std(final_te_acc)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Omniglot', help="Selected dataset.")
    parser.add_argument("--alg", type=str, default='LocalMOML', help="Selected algorithm.")
    parser.add_argument("--n_shot", type=int, default=1, help="N_shot.")
    parser.add_argument("--n_query", type=int, default=1, help="N_query.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.5, help="Mom.")
    parser.add_argument("--grad_mom", type=float, default=0.9, help="gradient momentum for NASA")
    args, unparsed = parser.parse_known_args()
    train(args)
