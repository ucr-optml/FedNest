# Algorithm: FedNest, iid, green line in Fig. 3b & blue line in Fig.3a
python main_imbalance.py  --epoch 3000  --round 3000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 3 --local_ep 5 --outer_tau 1 \
--hvp_method global --optim svrg --iid \
--output output/im_iid_fednest.yaml

# Algorithm: LFedNest, iid, orange line in Fig.3a
python main_imbalance.py  --epoch 3000  --round 3000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 3 --local_ep 5 --outer_tau 1 \
--hvp_method seperate --optim sgd --iid \
--output output/im_iid_lfednest.yaml

# Algorithm: FedNest, non-iid, blue line in Fig. 3b
python main_imbalance.py  --epoch 3000  --round 3000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 3 --local_ep 5 --outer_tau 1 \
--hvp_method global --optim svrg \
--output output/im_noniid_fednest.yaml

# Algorithm: FedNest_SGD, non-iid, orange line in Fig.3b
python main_imbalance.py  --epoch 3000  --round 3000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 3 --local_ep 5 --outer_tau 1 \
--hvp_method global --optim sgd \
--output output/im_noniid_fednestsgd.yaml

# Algorithm: FedNest_SGD, iid, red line in Fig.3b
python main_imbalance.py  --epoch 3000  --round 3000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 3 --local_ep 5 --outer_tau 1 \
--hvp_method global --optim sgd --iid \
--output output/im_iid_fednestsgd.yaml
