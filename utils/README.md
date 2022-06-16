## Dataset structure

### iid

- imbalanced ratio 0.01 6000 vs 60
- 100 clients, each client has 148 samples, 118 train, 30 val
- randomly sample each client from imbalanced dataset

### non-iid

- same overall dataset as iid
- 100 clients, first sort all samples with their labels, then split them to 200 blocks.
e.g.
```
11111111112222222223333344444555
1111 1111 1122 2222 2223 3333 4444 4555
```
- randomly assign 2 blocks to a client without replacement. e.g.
```
client 1: 1111 3333
client 2: 1122 4555
client 3: 2223 4444
client 4: 2222 1111
```