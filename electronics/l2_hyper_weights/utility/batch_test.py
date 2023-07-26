'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.parser import parse_args
from utility.load_data import *
from utility.helper import calc_auc
from evaluator import eval_score_matrix_foldout
import queue
from queue import Queue
import threading
import multiprocessing
from multiprocessing import Pool
import heapq
import random
import numpy as np
import pickle
import os
import math

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

cores = multiprocessing.cpu_count() // 2

args = parse_args()

n_layers = len(eval(args.layer_size))

data_path = os.path.join(args.data_path, 'data_no_replace_gen.pkl') # Noted by Yinan, not used
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size,n_layers=n_layers)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size

# pos: test_items
# neg: non train and non test items
def compute_auc_score(rate_batch, test_items, train_items):
    use_train = len(train_items) > 0
    test_auc = []
    N = rate_batch.shape[1]
    sample_num = 10
    for u in range(len(test_items)):
        count = 0
        corrects = 0
        M = set()
        for _ in test_items[u]:
            M.add(_)

        if use_train:
            for _ in train_items[u]:
                M.add(_)

        for pos in test_items[u]:
            for j in random.sample(xrange(N), sample_num):
                if j not in M:
                    corrects += rate_batch[u][pos] > rate_batch[u][j]
                    count += 1
        test_auc.append(corrects/1.0/count)
    return np.mean(test_auc)

def test(sess, model, users_to_test, test_begin_index, test_end_index, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of items
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {'precision': np.zeros(len(model.Ks)), 'recall': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks)), 'auc': 0}
    count = 0

    all_result = []
    auc_result = []

    n_users, n_items = model.n_users, model.n_items
    # users
    stride = 4 * model.n_acc
    user_stride = stride * 30 # 1200(15G), all=2w

    test_begin_index = int(math.floor(test_begin_index/1.0/user_stride)) * user_stride
    test_end_index = int(math.ceil(test_end_index/1.0/user_stride)) * user_stride


    # fill ratings
    n_users_c = int(math.ceil(n_users/1.0/user_stride)) * user_stride
    n_items_c = int(math.ceil(n_items/1.0/stride)) * stride
    ratings = np.zeros([n_users_c, n_items_c])

    i_embs = []
    for u_start_idx in range(test_begin_index, test_end_index, stride):
        u_start, u_end = u_start_idx, u_start_idx+stride
        if u_start_idx % (stride*15) == 0:
            logging.info('testing user: %d to %d'%(u_start, u_end))

        # (4, n_acc, dim)
        i_embs_out = sess.run(model.embs_i,
            feed_dict=dict(zip(model.guides_u,
                [range(u_start+_*model.n_acc, u_start+(_+1)*model.n_acc) \
                    for _ in range(4)])))
        i_embs.extend(i_embs_out)

        ## calc i guide and ratings 
        if u_end % user_stride == 0:
            i_embs = np.concatenate(i_embs, 0) # (user_stride, 7w, 64)
            i_embs = np.concatenate([i_embs, 
                np.zeros([user_stride, n_items_c-n_items, model.emb_dim])], axis=1)

            for i_start_idx in range(0, n_items_c, stride):
                i_start, i_end = i_start_idx, i_start_idx+stride
                if i_start_idx % (stride*15) == 0:
                    logging.info('item: %d to %d'%(i_start, i_end))

                # guide_i
                feed_dict = dict(zip(model.guides_i,
                    [range(n_users+i_start+_*model.n_acc, n_users+i_start+(_+1)*model.n_acc) \
                        for _ in range(4)]))
                # input_embs_i
                feed_dict.update(dict(zip(model.input_embs_i,
                    [i_embs[:, i_start+_*model.n_acc: i_start+(_+1)*model.n_acc]
                        for _ in range(4)])))
                # u_stride
                feed_dict.update({model.u_stride: range(u_end-user_stride, u_end)})
                rate = sess.run(model.output_ts, feed_dict=feed_dict)
                ratings[u_end-user_stride: u_end, i_start: i_end] = rate
            i_embs = []


    # fill ratings
    ratings = ratings[:n_users, :n_items]
#    ratings = np.zeros([n_users, n_items])

    u_batch_size = BATCH_SIZE
    n_test_users = len(users_to_test)
    n_user_batchs = int(math.ceil(n_test_users /1.0/ u_batch_size))

#    ratings = sess.run(model.ratings)
#    ratings = np.array(ratings) # n_users, n_items

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = users_to_test[start: end]
        
        rate_batch = ratings[user_batch, :]
        rate_batch = np.array(rate_batch)

        logging.info('%d/%d: %s'%(u_batch_id, n_user_batchs, str(rate_batch.shape)))
        train_items = []
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                test_items.append(data_generator.test_set[user])# (B, #test_items)
            for idx, user in enumerate(user_batch):
                train_items_off = data_generator.train_items[user]
                rate_batch[idx][train_items_off] = -np.inf
                train_items.append(train_items_off)
        else:
            for user in user_batch:
                test_items.append(data_generator.train_items[user])

       
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)#(B,k*metric_num), max_top= 20

        count += len(batch_result)
        auc_result.append(compute_auc_score(rate_batch, test_items, train_items))
        all_result.append(batch_result)


    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    result['auc'] = np.mean(auc_result)
    return result








