import matplotlib.pyplot as plt
import yaml
import numpy as np
from scipy.signal import savgol_filter

file_list = [
    ["results/ib_optim/noniid_mnistLT_svrg_LR0.01_HLR0.02_N3_IT3_LE5_OT1_global_batch_0.yaml",
    "results/ib_optim/noniid_mnistLT_svrg_LR0.01_HLR0.02_N3_IT3_LE5_OT1_global_batch_1.yaml",
    ],
    ["results/ib_optim/noniid_mnistLT_sgd_LR0.01_HLR0.02_N3_IT3_LE5_OT1_global_batch_0.yaml",
    "results/ib_optim/noniid_mnistLT_sgd_LR0.01_HLR0.02_N3_IT3_LE5_OT1_global_batch_1.yaml",
    ],

    "results/global_local/iid_mnistLT_svrg_LR0.01_HLR0.01_N5_IT3_LE5_OT3_global_batch_0.yaml",
    "results/ib_optim/iid_mnistLT_sgd_LR0.01_HLR0.02_N3_IT3_LE5_OT1_global_batch_0.yaml",

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
    if isinstance(file,list):
        f = open(file[0],mode='r')
        d = yaml.load(f,Loader=yaml.FullLoader)
        dict_list.append(d)
        f.close()
        for i in range(1,len(file)):
            f = open(file[i],mode='r')
            d=yaml.load(f,Loader=yaml.FullLoader)
            dict_list[-1]["test_acc"]=list(np.array(dict_list[-1]["test_acc"])+np.array(d["test_acc"]))
        dict_list[-1]["test_acc"]=list(np.array(dict_list[-1]["test_acc"])/float(len(file)))
    else:
        f = open(file, mode='r')
        d = yaml.load(f, Loader=yaml.FullLoader)
        dict_list.append(d)

plt.cla()
for name, result in zip(name_list, dict_list):
    index = list(map(lambda i: i> 6000, result["round"])).index(True)
    print(index)
    plt.plot(result["round"][:index],smooth(result["test_acc"][:index],0),linewidth=3)
    #plt.plot(result["test_acc"])
plt.legend(name_list,fontsize=16)
plt.xticks([0,1000,2000,3000,4000,5000,6000], fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((60,100))
plt.hlines(y=92.37,linestyles='--', xmin=0, xmax=6000, colors= 'tab:brown', linewidth=2)
plt.hlines(y=87.38,linestyles='--',xmin=0, xmax=6000, colors = 'black', linewidth=2)
# plt.xlim((-5,300))
plt.grid('--')

plt.savefig('results/figs/ib_optim_round.pdf')

plt.cla()
for name, result in zip(name_list, dict_list):
    plt.plot(smooth(result["test_acc"],0),linewidth=3)
    #plt.plot(result["test_acc"])
plt.legend(name_list,fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((60,100))
# plt.xlim((-5,300))
plt.grid('--')

plt.savefig('results/figs/ib_optim_epoch.pdf')

