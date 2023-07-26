import sys
import math

#Epoch 220 [1317.0s + 1027.5s]: test==[0.00000=0.00000 + 0.00000 + 0.00000], recall=[0.17076], precision=[0.05637], ndcg=[0.14694], auc=[0.95684]

num_workers = 6
num_users = 29858
stride = int(math.ceil(num_users/1.0/num_workers))

num_per_workers = [stride for _ in range(num_workers)]
num_per_workers[-1] -= stride * num_workers - sum(num_per_workers)

pools = []
for line in sys.stdin:
    line = line.strip()
    epoch_id = int(line.split(' ')[4])
    recall = float(line.split('[')[3].split(']')[0])
    precision = float(line.split('[')[4].split(']')[0])
    ndcg = float(line.split('[')[5].split(']')[0])
    auc = float(line.split('[')[6].split(']')[0])

    pools.append((epoch_id, recall, precision, ndcg, auc))

    if len(pools) == num_workers:
        dats = []
        for rate in zip(*pools):
            rate = sum([i*j for (i, j) in zip(rate, num_per_workers)])
            rate /= num_users
            dats.append(rate)
        print("Epoch %d, recall=[%.5f], precision=[%.5f], ndcg=[%.5f], auc=[%.5f]" % tuple(dats))
        pools = []
