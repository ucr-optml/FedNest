# Algorithm: FedNest, tau=1, non-iid, corresponds to all blue lines in Fig 2 (abc)
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_noniid.yaml --epoch 500 --round 10000000 \
--local_bs 64 

# Algorithm: LFedNest, non-iid, orange line in Fig.2a
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_noniid_lfednest.yaml --epoch 500 --round 10000000 \
--local_bs 64  --hvp_method seperate --optim sgd

# Algorithm: FedNest, tau=1, iid, All green lines in Fig.2 (abc)
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_iid.yaml --epoch 500 --round 10000000 \
--local_bs 64 --iid

# Algorithm: LFedNest, iid, red line in Fig.2a
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_iid_lfednest.yaml --epoch 500 --round 10000000 \
--local_bs 64 --iid --hvp_method seperate --optim sgd

# FedNest_SGD, tau=1, noniid, orange line in Fig.2b
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_noniid_fednestsgd --epoch 500 --round 10000000 \
--local_bs 64 --gpu 1 --optim sgd

# FedNest_SGD, tau=1, iid, red line in Fig.2b
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 1 \
--output output/tau_1_iid_fednestsgd --epoch 500 --round 10000000 \
--local_bs 64 --gpu 1 --iid --optim sgd

# Algorithm: FedNest, tau=5, non-iid, orange line in Fig.2c
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 5 \
--output output/tau_5_noniid.yaml --epoch 500 --round 10000000 \
--local_bs 64 --gpu 1

# Algorithm: FedNest, tau=5, iid, red line in Fig.2c
python main_hr.py --hlr 0.01 --lr 0.01 --outer_tau 5 \
--output output/tau_5_iid.yaml --epoch 500 --round 10000000 \
--local_bs 64 --gpu 1 --iid

