import numpy as np
from src.tasks import Sine_Task, sample_amp_pha
import torch
from src import Model, MAML, MOML, MOMLV2, LCMOML, NASA, SpiderBoost, Reptile
from src.utils import loss_on_random_task, seed_everything
import argparse
from config import N_amp, N_pha, xmax, xmin, N_test, seeds, test_size_adapt, test_size_plot, N_tasks, \
    B_tasks, H, N1, total_iterations, test_iter, lr_i, q


def train(args):
    ## generate tasks
    amplitudes = np.arange(1, N_amp + 1)
    phases = np.pi * np.arange(1, N_pha + 1) / N_pha
    all_tasks = [Sine_Task(amplitude=amp, phase=pha, xmin=xmin, xmax=xmax) for amp in amplitudes for pha in phases]
    all_te_tasks = []
    for i_te in range(N_test):
        te_amp, te_pha = sample_amp_pha(1, N_amp, np.pi / N_pha, np.pi, seed=i_te)
        all_te_tasks.append(Sine_Task(amplitude=te_amp, phase=te_pha, xmin=xmin, xmax=xmax))
    # # generate some test data for curve fitting
    all_X_plot = []
    all_y_plot = []
    all_X_te = []
    all_y_te = []
    for i_te, te_task in enumerate(all_te_tasks):
        X_plot, y_plot = te_task.sample_test_data(test_size_plot)
        X_te, y_te = te_task.sample_test_data(test_size_adapt)
        all_X_plot.append(X_plot)
        all_y_plot.append(y_plot)
        all_X_te.append(X_te)
        all_y_te.append(y_te)

    ## initialize the logs
    all_tr_loss = {}
    final_te_loss = []
    all_tr_samples = {}
    all_avg_time = []

    for seed in seeds:
        model = Model()
        print("==================Seed: {0}, Algorithm: {1}==================".format(seed, args.alg))
        if args.alg == 'MAML':
            loss = MAML(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr, K=args.K,
                        N_tasks=N_tasks, B_tasks=B_tasks, seed=seed)
            tr_losses, tr_iters, avg_time = loss.main_loop(num_iterations=total_iterations)
            tr_samples = np.multiply(tr_iters, B_tasks * args.K)
        elif args.alg == 'Reptile':
            loss = Reptile(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr, K=args.K,
                        N_tasks=N_tasks, B_tasks=B_tasks, seed=seed)
            tr_losses, tr_iters, avg_time = loss.main_loop(num_iterations=total_iterations)
            tr_samples = np.multiply(tr_iters, B_tasks * args.K)
        elif args.alg == 'MOML':
            loss = MOML(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr, K=args.K, N_tasks=N_tasks,
                        B_tasks=B_tasks, seed=seed, beta=args.beta)
            tr_losses, tr_iters, avg_time = loss.main_loop(num_iterations=total_iterations)
            tr_samples = np.multiply(tr_iters, B_tasks * args.K)
        elif args.alg == 'MOML-V2':
            loss = MOMLV2(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr, K=args.K, N_tasks=N_tasks,
                          B_tasks=B_tasks, seed=seed, beta=args.beta)
            tr_losses, tr_iters, avg_time = loss.main_loop(num_iterations=total_iterations)
            tr_samples = np.multiply(tr_iters, B_tasks * args.K)
        elif args.alg == 'LocalMOML':
            ratio = (args.K0 + args.K * H) / (args.K * H)
            loss = LCMOML(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr,
                          K=args.K, N_tasks=N_tasks, B_tasks=B_tasks, seed=seed, beta=args.beta, K0=args.K0, H=H)
            tr_losses, tr_iters, avg_time = loss.main_loop(num_iterations=int(total_iterations // ratio))
            tr_samples = np.multiply(tr_iters, B_tasks * args.K * ratio)
        elif args.alg == 'NASA':
            loss = NASA(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr,
                        K=args.K, B_tasks=B_tasks, N_tasks=N_tasks, seed=seed, beta=args.beta, grad_mom=args.grad_mom)
            tr_losses, tr_iters, avg_time = loss.main_loop(
                num_iterations=int((total_iterations * B_tasks) / N_tasks))
            tr_samples = np.multiply(tr_iters, N_tasks * args.K)
        elif args.alg == 'BSpiderBoost':
            loss = SpiderBoost(model, all_tasks, inner_lr=lr_i, meta_lr=args.lr, K=args.K,
                               N_tasks=N_tasks, N1=N1, N2=B_tasks, q=q, seed=seed)
            tr_losses, tr_samples, avg_time = loss.main_loop(
                num_iterations=int(total_iterations * B_tasks // (N1 / q + B_tasks * (q - 1) / q)))
        else:
            loss = None
            raise ValueError("Unknown algorithm!")

        # test
        all_te_loss = []
        all_predicted = []
        all_gt_x = []
        all_gt_y = []
        avg_te_loss = 0.0
        for te_i in range(N_test):
            X_plot = all_X_plot[te_i]
            y_plot = all_y_plot[te_i]
            X_te = all_X_te[te_i]
            y_te = all_y_te[te_i]
            te_task = all_te_tasks[te_i]
            te_loss_per_task, predicted_per_task = loss_on_random_task(te_task, loss.model.model,
                                                                       K=test_size_adapt,
                                                                       num_steps=test_iter,
                                                                       X_plot=X_plot, y_plot=y_plot,
                                                                       X_te=X_te, y_te=y_te)
            avg_te_loss += te_loss_per_task.item() / N_test
            all_te_loss.append(te_loss_per_task.item())
            X_plot_sq = torch.squeeze(all_X_plot[te_i]).cpu().detach().numpy()
            idx = np.argsort(X_plot_sq)
            gt_x = X_plot_sq[idx]
            gt_y = all_te_tasks[te_i].true_function(gt_x)
            all_gt_x.append(gt_x)
            all_gt_y.append(gt_y)
            all_predicted.append(predicted_per_task[idx].cpu().detach().numpy())

        all_gt_x = np.array(all_gt_x)
        all_gt_y = np.array(all_gt_y)
        all_predicted = np.array(all_predicted)
        all_te_loss = np.array(all_te_loss)
        # Save the groundtruth and the fitted curves
        file_name = './results/{}_{}_{}_curves.npz'.format(seed, args.alg, args.K)
        np.savez(file_name, gt_x=all_gt_x, gt_y=all_gt_y, predicted=all_predicted, te_loss=all_te_loss)
        ################################################
        all_tr_loss[seed] = tr_losses
        all_tr_samples[seed] = np.array(tr_samples).astype(np.float64)
        final_te_loss.append(avg_te_loss)
        all_avg_time.append(avg_time)

    final_te_loss = np.array(final_te_loss).flatten()
    file_name = './results/{}_{}_traces.npz'.format(args.alg, args.K)
    avg_tr_loss = np.zeros_like(all_tr_loss[seeds[0]])
    std_tr_loss = np.zeros_like(all_tr_loss[seeds[0]])
    avg_tr_samples = np.zeros_like(all_tr_samples[seeds[0]])
    for seed in seeds:
        avg_tr_loss += np.array(all_tr_loss[seed]) / float(len(seeds))
        avg_tr_samples += np.array(all_tr_samples[seed]) / float(len(seeds))
    for seed in seeds:
        std_tr_loss += (np.array(all_tr_loss[seed]) - avg_tr_loss) ** 2 / float(len(seeds))
    std_tr_loss = np.sqrt(std_tr_loss)
    np.savez(file_name, final_te_loss=final_te_loss, all_avg_time=all_avg_time, avg_tr_loss=avg_tr_loss,
             std_tr_loss=std_tr_loss,
             avg_tr_samples=avg_tr_samples)

    print("{0}: final test error: avg:{1}, std:{2}, time per iteration: avg:{3}, std:{4}".format(args.alg,
                                                                                                 np.average(
                                                                                                     final_te_loss),
                                                                                                 np.std(final_te_loss),
                                                                                                 np.mean(all_avg_time),
                                                                                                 np.std(all_avg_time)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default='MAML', help="Selected Algorithm.")
    parser.add_argument("--K", type=int, default=1, help="Batch size of data.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Mom.")
    parser.add_argument("--K0", type=int, default=2, help="K0 for LocalMOML")
    parser.add_argument("--grad_mom", type=float, default=0.9, help="gradient momentum for NASA")
    args, unparsed = parser.parse_known_args()
    train(args)
