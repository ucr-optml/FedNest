import torch
from core.Client import Client

class SGDClient(Client):
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param= None) -> None:
        super().__init__(args, client_id, net, dataset, idxs, hyper_param)
    def train_epoch(self):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    





