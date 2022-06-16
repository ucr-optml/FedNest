import yaml

class Logger():
    def __init__(self, filename) -> None:
        self.filename=filename
        self.logs={"clients": [], "train_loss": [],
            "test_loss": [], "test_acc": [], "train_acc": [], "round": []}
        self.counter=0
    def logging(self,client_idx, acc_test, acc_train, loss_test, loss_train, round):
        self.logs["clients"].append(client_idx.tolist())

        self.logs['test_acc'].append(acc_test.item())
        self.logs["train_acc"].append(acc_train.item())
        self.logs["test_loss"].append(loss_test)
        self.logs["train_loss"].append(loss_train)
        self.logs["round"].append(round)
        self.counter+=1
    def save(self):
        #if self.counter%100==0:
        f = open(self.filename, mode="w+")
        yaml.dump(self.logs, f)
        f.close()