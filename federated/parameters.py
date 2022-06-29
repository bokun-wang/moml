import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--local_batchsize', type=int, default=40)
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--model_name', type=str, default='dnn')
parser.add_argument('--dataset', type=str, default='cifar10')

parser.add_argument('--update_lr', type=float, default=0.01)   
parser.add_argument('--meta_lr', type=float, default=0.01) 
parser.add_argument('--ft_lr', type=float, default=0.001) 
parser.add_argument('--weight_decay', type=float, default=1e-5) 
parser.add_argument('--H', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='SGD')

parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--test_batchsize', type=int, default=32)
parser.add_argument('--test_batches', type=int, default=100) # total used in testing" test_batchsize * test_batches
parser.add_argument('--save_freq', type=int, default=10000)

parser.add_argument('--numGPU', type=int, default=1)
parser.add_argument('--total_iters', type=int, default=1000)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--master_addr', type=str)

# MAML
parser.add_argument('--n_way', type=int, help='n way', default=1)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
parser.add_argument('--order', type=int, help='order for gradients', default=1)
parser.add_argument('--beta', type=float, help='decay factor for moving average', default=1.0)
parser.add_argument('--num_tasks', type=int, help='meta batch size, namely task num', default=4)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

para = parser.parse_args()


if False:
    # parameters
    para.dataset = 'cifar10'
    update_lr = 0.01  # inside learner
    meta_lr = 0.01   # meta learner
    beta = 0.7
    num_classes = 10
    N = 50
    para.H = 4
    size = 4
    
    # fixed 
    update_step_test = 10
    eta_ft = 0.001
    n_way = 1
    k_shot = 5
    k_query = 10

                            
    inplanes = {'cifar10':32*32*3, 'mnist':28*28}[para.dataset]
    image_size = {'cifar10':32, 'mnist':28}[para.dataset]
    
    # datasets 
    datetime_now = '2021-05-20'
    configs = '[%s]Train_%s_N_%s_KS_%s_KQ_%s_%s_wd_%s_ulr_%s_mlr_%s_ftlr_%s_ftSP_%s_B_%s_K_%s_IMG_%s_CE_%s_H_%s_Beta_%s_K_%s_E_%s_S_%d_C%d'%(datetime_now, para.dataset, n_way, k_shot, k_query, para.model_name, para.weight_decay, para.update_lr, para.meta_lr, eta_ft, update_step_test, para.local_batchsize, size, image_size, para.optimizer, para.H, para.beta, para.num_tasks, para.total_iters, para.random_seed, num_classes)
    print (configs)