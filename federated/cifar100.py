import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

class CIFAR100_MAML(Dataset):
    def __init__(self, train_X, train_Y, mode, N, a, batchsz, n_way, k_shot, k_query, flatten_image=True, image_size=32, num_classes=100, size=3, gpu_rank=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs (number of iterations)
        :param n_way: number of gpus
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.data = train_X
        self.targets = train_Y
        self.mode = mode
        self.image_size = image_size
        self.num_samples = train_X.shape[0]
        self.flatten_image = flatten_image
        
        if flatten_image:
            # convert to 1D datasets
            self.data = self.data.reshape(self.num_samples, -1)
        
        # split datasets
        self.N = N
        self.a = a
        self.cls_num = num_classes
        self.size = size
        
        # assert self.cls_num == np.unique(Y), 'SSS'
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.gpu_rank = gpu_rank
        
         # TODO
        user_list = np.arange(N)
        user_list_split = np.array_split(user_list, size)
        self.user_pools = {}
        for n in range(size):
           self.user_pools[n] = user_list_split[n].tolist()        
        
        print('[GPU=%s] shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (self.gpu_rank, mode, batchsz, n_way, k_shot, k_query, image_size))
        
        self.make_data_index(self.targets, num_classes=num_classes)
        self.create_batch(self.batchsz)   

        self.transform_train = transforms.Compose([   
                                  lambda x: Image.fromarray(x.astype('uint8')),          
                                  transforms.ToTensor(),
                                  transforms.RandomCrop((image_size-2, image_size-2), padding=None),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.Resize((image_size, image_size)),
                                  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])
        self.transform_test = transforms.Compose([
                             lambda x: Image.fromarray(x.astype('uint8')),      
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                             #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ])
        
    def make_data_index(self, targets, num_classes=20):
        self.class_pool1 = iter(list(range(num_classes//2))*9999999)
        self.class_pool2 = iter(list(range(num_classes//2, num_classes))*9999999)
        self.target_to_index_dicts =  self.make_dicts(targets)
        self.partitions = {}
        for n in range(self.N):
            # (1) Half of the users, each have "a" images of each of the first five classes;
            if n < self.N//2:
              query_set = []
              for i in range(num_classes//2):
                  # print (len(self.target_to_index_dicts[i]))
                  query_set.extend([self.target_to_index_dicts[i].pop(0) for _ in range(self.a)])
              self.partitions[n] = np.array(query_set)
            # (2) The rest, each have "a/2" images from only one of the first five classes and "2a" images from only one of the other five classes
            else:
               # one of first five classes
               class_idx1 = next(self.class_pool1)
               query_samples1 = [self.target_to_index_dicts[class_idx1].pop(0) for _ in range(self.a//2)]
               # one of last five classes   
               class_idx2 = next(self.class_pool2)
               query_samples2 = [self.target_to_index_dicts[class_idx2].pop(0) for _ in range(2*self.a)]
               self.partitions[n] = np.array(query_samples1 + query_samples2)
              # query_set = []
              # for i in range(num_classes//2, num_classes):
              #     # print (len(self.target_to_index_dicts[i]))
              #     query_set.extend([self.target_to_index_dicts[i].pop(0) for _ in range(self.a)])
              # self.partitions[n] = np.array(query_set)        
              
              
              
    def make_dicts(self, labels):
        label_to_index_dicts = {}
        num_classes = np.unique(labels).shape[0]
        for idx in range(num_classes):
              class_idx = np.where(labels==idx)[0]
              label_to_index_dicts[idx] = class_idx.tolist()
        return label_to_index_dicts

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.user_id_batch = []
        for b in range(batchsz):  # for each batch
            # 1.select n_way users randomly
            if self.mode == 'train':
                #self.local_N = np.array_split(np.arange(self.N), self.size)[self.gpu_rank]  # near equal-size split
                #selected_cls = np.random.choice(self.local_N, self.n_way, False)  # no duplicate
                selected_cls = np.random.choice(self.user_pools[self.gpu_rank], self.n_way, False)    #self.N 
            else:
               # assert batchsz == self.N, 'Out of scope~!'
               # print (self.N, self.N//self.n_way)
                selected_cls = np.array_split(np.arange(self.N), self.N//self.n_way)[b] # near equal-size spli  #[list(range(self.N))[b]]
                
            # print (selected_cls)
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            user_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.partitions[cls]), self.k_shot + self.k_query, False)
                #print (cls, selected_imgs_idx)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(np.array(self.partitions[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.partitions[cls])[indexDtest].tolist())
                user_x.append(cls)

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)
            # print(support_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
            self.user_id_batch.append(user_x)
            
    def create_batch_by_uid(self, uid):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        assert isinstance(uid, list), 'uid needs to be a list!'
  
        selected_cls = uid
        support_x = []
        query_x = []
        user_x = []
        for cls in selected_cls:
            # 2. select k_shot + k_query for each class
            selected_imgs_idx = np.random.choice(len(self.partitions[cls]), self.k_shot + self.k_query, False)
            #print (cls, selected_imgs_idx)
            np.random.shuffle(selected_imgs_idx)
            indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
            indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
            support_x.append(np.array(self.partitions[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
            query_x.append(np.array(self.partitions[cls])[indexDtest].tolist())
            user_x.append(cls)
            
        # get images
        support_x_imgs = []
        support_y_labs = []
        for task in range(len(uid)):
            imgs_index = np.array(support_x[task])
            support_x_imgs.append(self.data[imgs_index])
            support_y_labs.append(self.targets[imgs_index])
            
        return torch.FloatTensor(support_x_imgs), torch.LongTensor(support_y_labs)

            

    def __getitem__(self, index):
        # placeholder
        support_x = np.zeros((self.setsz, 3*self.image_size*self.image_size))  # [setsz, channels, resize, resize]
        query_x = np.zeros((self.querysz, 3*self.image_size*self.image_size))
        for i in range(self.setsz):
            if self.flatten_image:
                s_index =  np.array(self.support_x_batch[index]).flatten()
                # print(s_index[i], self.setsz)
                support_x[i] = self.data[s_index[i]]/255.
            else:
                support_x[i] = self.transform_train(self.data[self.support_x_batch[index][i]])

        support_y = self.targets[np.array(self.support_x_batch[index]).flatten()]

        for i in range(self.querysz):
            if self.flatten_image:
                q_index=  np.array(self.query_x_batch[index]).flatten()
                query_x[i] = self.data[q_index[i]]/255.
            else:
                query_x[i] = self.transform_train(self.data[self.query_x_batch[index][i]])
                
        query_y = self.targets[np.array(self.query_x_batch[index]).flatten()]
        
        support_y = support_y.flatten()
        query_y = query_y.flatten()
        
        # # convert to relative labels
        # unique = np.unique(support_y)
        # random.shuffle(unique)
        # support_y_relative = np.zeros(self.setsz)
        # query_y_relative = np.zeros(self.querysz)
        # for idx, l in enumerate(unique):
        #     support_y_relative[support_y == l] = idx
        #     query_y_relative[query_y == l] = idx
        
        # print (support_y_relative)
        uid = self.user_id_batch[index]
        # print (unique)
        # print (support_y_relative, query_y_relative)
        return torch.FloatTensor(support_x), torch.LongTensor(support_y),  torch.FloatTensor(query_x), torch.LongTensor(query_y), torch.LongTensor(uid)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    from tensorflow import keras 
    (train_data, train_labels), (test_data, test_labels) =  keras.datasets.cifar100.load_data(label_mode="coarse")
    # cifar100 = CIFAR100_MAML(train_data, train_labels, mode='train', N=50, a=68, batchsz=10000, n_way=1, k_shot=5, k_query=10, image_size=32, num_classes=20, gpu_rank=0)
    # trainloader = torch.utils.data.DataLoader(cifar100, 5, shuffle=True, num_workers=0, pin_memory=False)
    
    cifa100_test = CIFAR100_MAML(test_data, test_labels, mode='test', N=50, a=15, batchsz=50, n_way=1, k_shot=5, k_query=10, image_size=32, num_classes=20, gpu_rank=0)
    testloader = torch.utils.data.DataLoader(cifa100_test, 50, shuffle=False, num_workers=0, pin_memory=False)

    for i, sets in enumerate(testloader):
        support_x, support_y, query_x, query_y, uid = sets
        print (support_x.shape, support_y.shape, query_x.shape, query_y.shape)
        
    # for i, sets in enumerate(trainloader):
    #     support_x, support_y, query_x, query_y, uid = sets
    #     print (support_x.shape, support_y.shape, query_x.shape, query_y.shape, support_y.max(), support_y.min(), query_y.max(), query_y.min())
           