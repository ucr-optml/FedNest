import torch
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_iid_normal, mnist_noniid, cifar_iid, mnist_noniid_normal, minmax_dataset, fmnist_iid_normal, fmnist_noniid_normal
import numpy as np
import random
def load_data(args):
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users, dataset_train_real = mnist_iid_normal(dataset_train, args.num_users)
        else:
            dict_users, dataset_train_real = mnist_noniid_normal(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users, dataset_train_real = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'minmax_synthetic':
        dataset_train, dataset_test, dict_users, img_size, dataset_train_real = minmax_dataset(args)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.3476,), (0.3568,))])
        dataset_train = datasets.FashionMNIST("../data/fmnist/",train = True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST("../data/fmnist", train = False, transform=trans_mnist)
        labels_train = dataset_train.targets.numpy()
        labels_test = dataset_test.targets.numpy()
        train_index = np.any([labels_train == 4, labels_train == 6, labels_train == 0], axis=0)
        test_index = np.any([labels_test == 4, labels_test == 6, labels_test == 0], axis=0)
        dataset_train.data, dataset_train.targets = dataset_train.data[train_index].float()/255., dataset_train.targets[train_index]
        dataset_test.data, dataset_test.targets = dataset_test.data[test_index].float()/255., dataset_test.targets[test_index]
        
        train_index=list(range(dataset_train.data.shape[0]))
        random.shuffle(train_index)

        dataset_train.data, dataset_train.targets = dataset_train.data[train_index].float()/255., dataset_train.targets[train_index]
        #print(torch.mean(dataset_train.data.float().view(-1)), torch.std(dataset_train.data.float().view(-1)))
        #dataset_train.data = dataset_train.data.view(18000,1,28,28)
        #dataset_test.data = dataset_test.data.view(3000,1,28,28)
        labels_train = dataset_train.targets.numpy()
        labels_test = dataset_test.targets.numpy()
        print(labels_train)
        for i in range(labels_train.shape[0]):
            if labels_train[i]==4:
                labels_train[i]=1
            elif labels_train[i]==6:
                labels_train[i]=2
        for i in range(labels_test.shape[0]):
            if labels_test[i]==4:
                labels_test[i]=1
            elif labels_test[i]==6:
                labels_test[i]=2
        if args.iid:
            dict_users, dataset_train_real = fmnist_iid_normal(dataset_train, args.num_users)
        else:
            dict_users, dataset_train_real = fmnist_noniid_normal(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    return  dataset_train, dataset_test, dict_users, img_size, dataset_train_real


