#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from math import ceil
import numpy as np
from torchvision import datasets, transforms
import torch

def mnist_iid_normal(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        train_index = np.random.choice(all_idxs, num_items, replace=False)
        
        all_idxs = list(set(all_idxs) - set(train_index))
        #print(train_index.shape, len(all_idxs))
        #train_index= list(train_index)
        val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.5), replace=False)
        train_index = np.array(list(set(train_index) - set(val_index)))

        #print(train_index.shape,val_index.shape)
        dict_users[i] = train_index
        dict_users[-i-1] = val_index
        
    return dict_users, dataset


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_real = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)

    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}

    labels = dataset.targets.numpy()

    # Imbalance
    mu=0.01**(1/9)
    probability=[mu**i for i in range(0,10)]
    #print(probability)
    all_idxs = []
    for i in range(10):
        num_sample=min(ceil(labels.shape[0]/10*probability[i]),6000)
        print(num_sample,len(np.where(labels==i)[0][:num_sample]))
        all_idxs.extend(np.where(labels==i)[0][:num_sample])
    print(len(all_idxs))
    for i in range(10):
        labels_new = labels[all_idxs]
        print(np.where(labels_new==i)[0].shape,np.where(labels==i)[0])
    dataset_real.data = dataset_real.data[all_idxs]
    dataset_real.targets = dataset_real.targets[all_idxs]
    # train-val split, for i client : train = dict_user[i], val = dict_user[-i-1]
    num_items = int(len(all_idxs)/num_users)
    dict_users = {}
    for i in range(num_users):
        train_index = np.random.choice(all_idxs, num_items, replace=False)
        
        all_idxs = list(set(all_idxs) - set(train_index))
        #print(train_index.shape, len(all_idxs))
        #train_index= list(train_index)
        val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.5), replace=False)
        train_index = np.array(list(set(train_index) - set(val_index)))

        #print(train_index.shape,val_index.shape)
        dict_users[i] = train_index
        dict_users[-i-1] = val_index
        
    return dict_users, dataset_real





def mnist_noniid_normal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    #dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}

    labels = dataset.targets.numpy()

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}
    idxs = np.arange(num_shards*num_imgs)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # print(idxs_labels.shape, idxs_labels)
    
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        train_index = []
        for rand in rand_set:
            train_index.extend(idxs[rand*num_imgs:(rand+1)*num_imgs])

        # print(len(train_index),labels[train_index])
        #train_index= list(train_index)
        train_index=np.array(train_index)
        val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.2), replace=False)
        train_index = np.array(list(set(train_index) - set(val_index)))

        dict_users[i] = train_index
        dict_users[-i-1] = val_index


    return dict_users, dataset
    # num_shards, num_imgs = 200, 300
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.targets.numpy()

    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]

    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    #     #print(labels[dict_users[i]])
    # return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_real = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)

    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}

    labels = dataset.targets.numpy()

    # Imbalance
    mu=0.01**(1/9)
    probability=[mu**i for i in range(0,10)]
    #print(probability)
    all_idxs = []
    for i in range(10):
        num_sample=min(ceil(labels.shape[0]/10*probability[i]),6000)
        #print(num_sample,len(np.where(labels==i)[0][:num_sample]))
        all_idxs.extend(np.where(labels==i)[0][:num_sample])
    #print(len(all_idxs))
    for i in range(10):
        labels_new = labels[all_idxs]
        # print(np.where(labels_new==i)[0].shape,np.where(labels==i)[0])
    dataset_real.data = dataset_real.data[all_idxs]
    dataset_real.targets = dataset_real.targets[all_idxs]

    num_shards, num_imgs = 200, int(len(all_idxs)/200)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}

    idxs_labels = np.vstack((all_idxs,labels[all_idxs]))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # print(idxs_labels.shape, idxs_labels)
    idxs = idxs_labels[0,:]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        train_index = []
        for rand in rand_set:
            train_index.extend(idxs[rand*num_imgs:(rand+1)*num_imgs])

        # print(len(train_index),labels[train_index])
        #train_index= list(train_index)
        train_index=np.array(train_index)
        val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.2), replace=False)
        train_index = np.array(list(set(train_index) - set(val_index)))

        #print(train_index.shape,val_index.shape)
        dict_users[i] = train_index
        dict_users[-i-1] = val_index

    return dict_users, dataset_real

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, dataset_train

def minmax_dataset(args):
    num_users=args.num_users
    base_s= args.minmax_s
    d= args.d
    n=args.n
    
    s_list=[]
    data=[]
    label = []
    #np.random.seed(base_s)
    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}
    if args.iid:
        for _ in range(num_users):
            s=base_s
            s_list.append(s)        
    else:
        for _ in range(num_users):
            s = abs(np.random.uniform(0,2*base_s))
            s_list.append(s)

    for i in range(num_users):
        s = s_list[i]

        # b = np.random.normal(0,s**2,n)
        # b = b - np.mean(b)
        # a = np.random.normal(1,s**2,n)
        # a[a<1]=1
        # A = np.diag(a)

        # b = np.zeros(n)
        # A = np.random.normal(0,s**2,(n,d))

        b = np.random.normal(0,s**2,n)
        a = np.random.uniform(0,1.,n)
        A = np.diag(a)

        data.append(A)
        label.extend(b)
        dict_users[i] = list(range(i*n,i*n+n))
        dict_users[-i-1] = list(range(i*n,i*n+n))

    data= np.vstack(data)
    label = np.vstack(label)
    #print(data.shape,label)
    #print(dict_users)
    #print(s_list)
    tensor_data = torch.Tensor(data)
    tensor_label = torch.Tensor(label)
    dataset = torch.utils.data.TensorDataset(tensor_data,tensor_label)
    #print(dataset)
    return dataset,None,dict_users, None, dataset


def fmnist_iid_normal(dataset, num_users):
    """
    Sample I.I.D. client data from Fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        train_index = np.random.choice(all_idxs, num_items, replace=False)
        
        all_idxs = list(set(all_idxs) - set(train_index))
        #print(train_index.shape, len(all_idxs))
        #train_index= list(train_index)
        #val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.5), replace=False)
        #train_index = np.array(list(set(train_index) - set(val_index)))
        val_index=train_index
        #print(train_index.shape,val_index.shape)
        dict_users[i] = train_index
        dict_users[-i-1] = val_index
        
    return dict_users, dataset

def fmnist_noniid_normal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    #dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}

    
    labels=dataset.targets.numpy()
    num_shards, num_imgs = num_users*2, 18000//(num_users*2)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range((-num_users-1),num_users)}
    idxs = np.arange(num_shards*num_imgs)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        train_index = []
        for rand in rand_set:
            train_index.extend(idxs[rand*num_imgs:(rand+1)*num_imgs])

        train_index=np.array(train_index)
        #val_index = np.random.choice(train_index, ceil(train_index.shape[0]*0.2), replace=False)
        #train_index = np.array(list(set(train_index) - set(val_index)))
        val_index=train_index
        dict_users[i] = train_index
        dict_users[-i-1] = val_index

    return dict_users, dataset


if __name__ == '__main__':
    dataset_train = datasets.FashionMNIST("../data/fmnist/",train=True, download=True)
    dataset_test = datasets.FashionMNIST("../data/fmnist", train = False, download=True)
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
