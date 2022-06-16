from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
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
            dict_users, dataset_train_real = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, dataset_train_real = mnist_noniid(dataset_train, args.num_users)
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
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    return  dataset_train, dataset_test, dict_users, img_size, dataset_train_real