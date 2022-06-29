import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import copy 
import os,re,time
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from parameters import para
from models import NeuralNetwork as dnn
from mnist10 import MNIST_MAML
from cifar10 import CIFAR10_MAML
from cifar100 import CIFAR100_MAML
from tensorflow import keras

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CEOptimizer:
    def __init__(self, model=None, **kwargs):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.sgd_optimizer = optim.SGD#(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.adam_optimizer = optim.Adam
        self.model = model
    def loss(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
    
    def binary_loss(self,y_pred, y_true):
        return F.binary_cross_entropy_with_logits( y_pred, y_true)
    
    def SGD(self, model=None, lr=0.1, momentum=0.9, weight_decay=1e-5):
        return self.sgd_optimizer(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def update(self):
        # average over all clients    
        size = float(dist.get_world_size())
        for name, param in self.model.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size
        

def train(rank, size, group):
    torch.cuda.set_device(para.local_rank)
    set_all_seeds(para.random_seed)
    
    # parameters
    para.dataset = para.dataset #'cifar10'
    update_lr = para.update_lr #0.01  # inside learner
    meta_lr =  para.meta_lr #s  0.01   # meta learner
    beta = para.beta #0.7
    N = 50
    # fixed 
    update_step_test = para.update_step_test
    eta_ft = para.ft_lr # 0.01
    n_way = para.n_way#  1
    k_shot = para.k_spt # 5
    k_query =  para.k_qry #10
    
    if para.dataset == 'mnist':
        num_classes = 10 
        (train_data, train_labels), (x_test, y_test) = keras.datasets.mnist.load_data()
        trainSet = MNIST_MAML(train_data, train_labels, mode='train', N=50, a=168, batchsz=1000, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=28, size=size,num_classes=num_classes,  gpu_rank=rank)
        testSet = MNIST_MAML(x_test, y_test, mode='test', N=50, a=34, batchsz=50, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=28,num_classes=num_classes,  gpu_rank=rank)                     
    elif para.dataset =='cifar10':
        num_classes = 10 
        (train_data, train_labels), (x_test, y_test) = keras.datasets.cifar10.load_data()
        trainSet = CIFAR10_MAML(train_data, train_labels, mode='train', N=50, a=68, batchsz=10000, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=32, size=size, num_classes=num_classes, gpu_rank=rank)
        testSet = CIFAR10_MAML(x_test, y_test, mode='test', N=50, a=34, batchsz=50, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=32,num_classes=num_classes,  gpu_rank=rank)
    elif para.dataset =='cifar100':
        num_classes = 20 
        (train_data, train_labels), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="coarse")
        trainSet = CIFAR100_MAML(train_data, train_labels, mode='train', N=50, a=68, batchsz=10000, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=32, size=size, num_classes=num_classes,  gpu_rank=rank)
        testSet = CIFAR100_MAML(x_test, y_test, mode='test', N=50, a=15, batchsz=50, n_way=n_way, k_shot=k_shot, k_query=k_query, image_size=32, num_classes=num_classes,  gpu_rank=rank)   
    else:
        raise ValueError

    trainloader = torch.utils.data.DataLoader(trainSet, para.num_tasks, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testSet, 1, shuffle=False, num_workers=2, pin_memory=True)
 
    inplanes = {'cifar10':32*32*3, 'cifar100':32*32*3, 'mnist':28*28}[para.dataset]
    image_size = {'cifar10':32, 'cifar100':32, 'mnist':28}[para.dataset]
    
    # datasets 
    datetime_now = '2021-05-26'
    configs = '[%s]Train_%s_N_%s_KS_%s_KQ_%s_%s_wd_%s_ulr_%s_mlr_%s_ftlr_%s_ftSP_%s_B_%s_IMG_%s_CE_%s_H_%s_Beta_%s_TS_%s_E_%s_GPU_%s_S_%d_C%d'%(datetime_now, para.dataset, n_way, k_shot, k_query, para.model_name, para.weight_decay, para.update_lr, para.meta_lr, eta_ft, update_step_test, para.local_batchsize, image_size, para.optimizer, para.H, para.beta, para.num_tasks, para.total_iters, size, para.random_seed, num_classes)
    SAVE_LOG_PATH = '/Users/zhuoning/Experiment/NIPS2021/fl_maml/distributed/logs/'
    
    model = dnn(in_planes=inplanes, hidden_size=[40, 40], num_classes=num_classes, last_activation=None)
    model = model.cuda()       
    model_names = [name for name, v in model.named_parameters()]
    model_pools = [ [layer.detach().cpu().numpy() for layer in model.parameters()] for n in range(N)]

    if para.local_rank ==0 and rank == 0 :
        print (configs)
        init_weights = [w.data.cpu().clone() for w in list(model.parameters())]
        print ('Init weights:', init_weights[0].numpy().sum())
        print ('-'*100)
        
    step = 0
    start_time = time.time()
    train_acc_list, test_acc_list, train_loss_list, test_loss_list  = [],  [],  [],  []
    for epoch in range(para.total_epochs):
        
        for _, (x_spt, y_spt, x_qry, y_qry, uid) in enumerate(trainloader):
            
            if step == para.total_iters+1:
               os.system('pkill python')
               break

            if step == int(para.total_iters*0.5) or step == int(para.total_iters*0.75):
                para.update_lr /= 10 
                para.meta_lr /= 10 
                # para.ft_lr =/= 10

            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()   
       
            # local updates 
            iter_train_acc_list = []
            iter_loss_list = []
            for i in range(para.num_tasks):  # n ways ; need to maintain #task models
          
                model_weigths = [torch.FloatTensor(m).cuda() for m in model_pools[uid[i].numpy()[0]]]
                model_dicts = dict(zip(model_names, model_weigths))
                model.load_state_dict(model_dicts)

                x_spt_init, y_spt_init = trainSet.create_batch_by_uid(uid.flatten().numpy().tolist())
                x_spt_init, y_spt_init = x_spt_init.cuda(), y_spt_init.cuda()
                logits = model(x_spt_init[i])
                
                loss = F.cross_entropy(logits, y_spt_init[i].reshape(-1))
                w_grads = torch.autograd.grad(loss, model.parameters())
                #print (w_grads.shape)
                u_weights = list(map(lambda p: p[0].data - update_lr* p[1].data, zip(model.parameters(), w_grads)))
                
                #print (logits.shape)
                
                for h in range(para.H):
                    logits = model(x_spt[i])
                    loss = F.cross_entropy(logits, y_spt[i].reshape(-1))
         
                    w_grads = torch.autograd.grad(loss, model.parameters())
                    fast_weights = list(map(lambda p: p[0].data - update_lr* p[1].data, zip(model.parameters(), w_grads)))
                    
                    u_weights = list(map(lambda p: p[0].data*(1-beta) + beta*p[1].data, zip(u_weights, fast_weights)))
                    
                    # compute gradients w.r.t "u"
                    model_w_copy = copy.deepcopy(model.state_dict())
                    for w, u in zip(model.parameters(), u_weights):
                        w.data = u.data
                            
                    logits_q = model(x_qry[i])
                    loss_q = F.cross_entropy(logits_q, y_qry[i].reshape(-1))
                    grads_q = torch.autograd.grad(loss_q, model.parameters())
      
                    # change back previous "w" an update local w     
                    for name, var in model.named_parameters():
                        var.data = model_w_copy[name].data      
                    for var, grad in zip(model.parameters(), grads_q):
                        var.data = var.data - meta_lr*grad.data     
                        
                    iter_loss_list.append(loss.item())
                        
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1) # convert to numpy
                        train_acc = accuracy_score(y_qry[i].cpu().numpy(), pred_q.cpu().numpy())
                        iter_train_acc_list.append(train_acc)
                    
                model_pools[uid[i].numpy()[0]] = [layer.detach().cpu().numpy() for layer in model.parameters()]
           
            
            # communicate with other servers after H local steps (only average sampled tasks!!!)
            size = float(dist.get_world_size())
            sampled_model_pools = [[torch.FloatTensor(m).cuda() for m in p_model] for idx, p_model in enumerate(model_pools) if idx in uid.numpy().flatten().tolist()] #[uid.numpy().flatten()]]
            for idx, p_model in enumerate(sampled_model_pools):
                new_p_model = []
                for layer in p_model:
                    dist.all_reduce(layer.data, op=dist.ReduceOp.SUM)
                    layer.data /= size
                    new_p_model.append(layer)
                sampled_model_pools[idx] = new_p_model
                
            sampled_model_pools = [[layer.detach().cpu().numpy() for layer in p_model] for p_model in sampled_model_pools]        
            model_user_average = [np.zeros(w.shape) for w in model_pools[0]]
            for idx, w_numpy in enumerate(model_user_average):
                for n in range(len(sampled_model_pools)):
                    w_numpy += sampled_model_pools[n][idx]
                w_numpy /= len(sampled_model_pools)      
                model_user_average[idx]  = w_numpy
            model_pools = [model_user_average]*N
                
            
            # if para.local_rank == 0 and rank==0:
            #    print ('Aggregates and broadcast weights...')      

            # evaluation  
            if step % 50 == 0 and para.local_rank == 0 and rank==0:              
                # # finetune
                iter_test_acc_list = []
                for x_spt, y_spt, x_qry, y_qry, uid in testloader:
                    x_spt, y_spt, x_qry, y_qry =  x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
                    
                    for i in range(1): 
                        
                        model_weigths = [torch.FloatTensor(m).cuda() for m in model_pools[uid[i].numpy()[0]]]
                        model_dicts = dict(zip(model_names, model_weigths))
                        model.load_state_dict(model_dicts)
                        
                        for k in range(update_step_test):
                            logits = model(x_spt[i])
                            loss = F.cross_entropy(logits, y_spt[i].reshape(-1))
                            w_grads = torch.autograd.grad(loss, model.parameters())
                            for var, grad in zip(model.parameters(), w_grads):
                                var.data = var.data - eta_ft*grad.data      

                            with torch.no_grad():
                                logits_q = model(x_qry[i])
                                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                                test_acc = accuracy_score(y_qry[i].cpu().numpy(), pred_q.cpu().numpy())
                                iter_test_acc_list.append(test_acc)
                
                print ('step:%s, train_loss:%.3f, train_acc:%.3f, test_acc:%3f, time:%.4f'% (step, np.mean(iter_loss_list), np.mean(iter_train_acc_list), np.mean(iter_test_acc_list), time.time() - start_time))
                start_time = time.time()
                train_acc_list.append(np.mean(iter_train_acc_list))
                test_acc_list.append(np.mean(iter_test_acc_list))
                train_loss_list.append(np.mean(iter_loss_list))
                
                df = pd.DataFrame(data={'train_acc_H_%s_b_%s_k_%s'%(para.H, para.beta, para.num_tasks):train_acc_list, 'train_acc_H_%s_b_%s_k_%s'%(para.H, para.beta, para.num_tasks):train_acc_list, 'val_acc_H_%s_b_%s_k_%s'%(para.H, para.beta, para.num_tasks) :test_acc_list})
                df.to_csv(SAVE_LOG_PATH + '%s.csv'%(configs)  )         

                if step == para.total_iters+1:
                    break             
                
            step += 1
            
    if para.local_rank == 0 and rank==0:       
        print ('Best Train Acc: %.3f'%max(train_acc_list), 'Best test Acc: %.3f'%max(test_acc_list) )
 

def init_env():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ipaddress = s.getsockname()[0]
    s.close()
    gpu_counts = torch.cuda.device_count()
    # print (gpu_counts)
    os.environ['MASTER_ADDR'] = ipaddress
    os.environ['MASTER_PORT'] = '8888'
    # os.environ['NPROC_PER_NODE'] = str(gpu_counts)
    # os.environ['nproc_per_node'] = str(gpu_counts)
    # os.environ['NNODES'] = '1'
    # os.environ['nnodes'] = '1' 
    # os.environ['NODE_RANK'] = '0'
    # os.environ['node_rank'] = '0'

if __name__ == "__main__":
    init_env()
    dist.init_process_group('nccl')
    size = dist.get_world_size()
    group = dist.new_group(range(size))
    rank = dist.get_rank()
    print ('Current Rank: %s, Number of nodes: %s'%(str(rank), str(size)))
    train(rank, size, group)

