from utils.svrg import SVRG_k,SVRG_Snapshot
import copy
from core.Client_hr import Client

class SVRGClient(Client):
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param=None) -> None:
        super().__init__(args, client_id, net, dataset, idxs, hyper_param)
    
    def set_avg_q(self,avg_q):
        self.avg_q=avg_q

    def train_epoch(self):
        self.net.train()
        for name, w in self.net.named_parameters():
            if not "header" in name:
                w.requires_grad= False
        # train and update
        optimizer = SVRG_k([k for k in self.net.parameters() if k.requires_grad==True], lr=self.args.lr)
        optimizer.set_u(self.avg_q)

        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optim0 = SVRG_Snapshot([k for k in self.net0.parameters() if k.requires_grad==True])

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.net0.zero_grad()
                log_probs_0=self.net0(images)
                loss0=self.loss_func(log_probs_0, labels, [k for k in self.net0.parameters() if k.requires_grad==True])
                loss0.backward()
                param_group=optim0.get_param_groups(1)

                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels, [k for k in self.net.parameters() if k.requires_grad==True])
                loss.backward()
                optimizer.step(param_group)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        for name, w in self.net.named_parameters():
            if not "header" in name:
                w.requires_grad= False
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    





