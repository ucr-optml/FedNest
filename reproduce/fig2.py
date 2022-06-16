import matplotlib.pyplot as plt
import yaml
import numpy as np

file_list = [
    "results/HR/tau/1_1.yaml",
    "results/HR/optim/noniid_sgd_1.yaml",

    "results/HR/tau/iid_1_1.yaml",
    "results/HR/optim/iid_sgd_1.yaml",

]
name_list = [
    r"$\mathrm{FedNest}$, non-iid",
    r"$\mathrm{FedNest}_\mathrm{SGD}$, non-iid",
    r"$\mathrm{FedNest}$, iid",
    r"$\mathrm{FedNest}_\mathrm{SGD}$, iid",
]

def smooth(scalars, weight) :  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed
    
dict_list = []
for file in file_list:
    f = open(file, mode='r')
    d = yaml.load(f, Loader=yaml.FullLoader)
    dict_list.append(d)

plt.cla()
for name, result in zip(name_list, dict_list):
    index = list(map(lambda i: i> 4000, result["round"])).index(True)
    print(index)
    plt.plot(result["round"][:index],smooth(result["test_acc"][:index],0),linewidth=3)
plt.legend(name_list,fontsize=16)
plt.xticks([0,1000,2000,3000,4000], fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((60,100))
plt.grid('--')

plt.savefig('results/figs/hr_optim_round.pdf')

plt.cla()
for name, result in zip(name_list, dict_list):
    plt.plot(smooth(result["test_acc"],0),linewidth=3)
plt.legend(name_list,fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((60,100))
plt.grid('--')

plt.savefig('results/figs/hr_optim_epoch.pdf')


