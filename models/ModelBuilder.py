from models.Nets import CNNCifar,CNNMnist,MLP,Linear,MM_CNN

def build_model(args):
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in args.img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200,
                       dim_out=args.num_classes).to(args.device)
    elif args.model == 'linear':
        net_glob = Linear(d=args.d,n=args.n).to(args.device)
    elif args.model == 'fmnist_cnn':
        net_glob = MM_CNN(args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    return net_glob