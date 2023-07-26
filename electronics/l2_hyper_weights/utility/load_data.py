import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pdb
import tensorflow as tf
import os

NNZ = 2**31/64
class Data(object):
    def __init__(self, path, batch_size, n_layers):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        self.train_u_p_pair =[]  # used for sampling without replacement
        self.test_u_p_pair =[]  # used for sampling without replacement
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    
                    self.train_u_p_pair.extend([(uid,i) for i in items]) # used for sampling without replacement
        
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)


        self.train_items, self.test_set = {}, {}

        self.max_degree_user=0 # max node degree in training set, user
        set_all = set(range(self.n_items))
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items
                    self.neg_pools[uid] = list(set_all - set(train_items))

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

                    self.test_u_p_pair.extend([(uid,i) for i in test_items])

        for ele in self.train_items:
            self.max_degree_user=max(len(self.train_items[ele]),self.max_degree_user)
        self.print_statistics()

        # no need make_dataset
        # self.get_adj_mat()
        # self.create_subgraph(n_layers+1)
        # self.make_pkl()


    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            logging.info('already load adj matrix' + str(adj_mat.shape) + str(time() - t1))

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)  # 0 1 matrix
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat) # row normilized (adj+eye)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat) # row normlized (adj)

        try:
            self.pre_adj_mat_csr = sp.load_npz(self.path + '/s_pre_adj_mat_csr.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            logging.info('generate pre adjacency matrix.')
            self.pre_adj_mat_csr = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_csr.npz', self.pre_adj_mat_csr)
        return adj_mat, norm_adj_mat, mean_adj_mat,self.pre_adj_mat_csr  # sqrt degree of u, sqrt degree of i


    def create_subgraph(self, n_layers): # the number of layers, e.g. n_layers=3, 0-root plus  1,2,3 hop neighboors, total 4 lays
        self.graph_neighboor = [[] for _ in range(n_layers-1)] # each layer ( n_users + n_items ),no last layer
        self.graph_adj = [[] for _ in range(n_layers)] # origin last layer, no deduplication
        
        count=[]
        non_zero_adj=[]
        for user in range(self.n_users):
            cur_layer = 0

            # first level
            neighboor = self.pre_adj_mat_csr[user,:].tocoo()
            uniq_neighboor_id, uniq_neighboor_idx = np.unique(neighboor.col,return_inverse=True)
            adj = sp.coo_matrix((neighboor.data,(neighboor.row,uniq_neighboor_idx)),shape=(1,len(uniq_neighboor_id)))

            # save
            self.graph_neighboor[cur_layer].append(uniq_neighboor_id)
            self.graph_adj[cur_layer].append(adj)

            cur_layer = cur_layer+1
            # second level
            while cur_layer < n_layers-1:
                neighboor = self.pre_adj_mat_csr[self.graph_neighboor[cur_layer-1][user],:].tocoo()
                uniq_neighboor_id, uniq_neighboor_idx = np.unique(neighboor.col,return_inverse=True)
                adj = sp.coo_matrix((neighboor.data,(neighboor.row,uniq_neighboor_idx)),shape=(self.graph_neighboor[cur_layer-1][user].size,len(uniq_neighboor_id)))

                # save
                self.graph_neighboor[cur_layer].append(uniq_neighboor_id)
                self.graph_adj[cur_layer].append(adj)
                cur_layer = cur_layer + 1
 
            # last layer, no unique is needed
            adj = self.pre_adj_mat_csr[self.graph_neighboor[cur_layer-1][user],:].tocoo()
            self.graph_adj[cur_layer].append(adj)

            count.append(len(self.graph_neighboor[n_layers-2][user]))
            non_zero_adj.append(len(self.graph_adj[n_layers-1][user].data))
        logging.info("user tree: average last-1 layer node %d" %(sum(count)/len(count)))
        logging.info("user tree: nnz last adj max %d, mean %d" %(max(non_zero_adj),sum(non_zero_adj)/len(non_zero_adj)))


        count=[]
        non_zero_adj=[]
        for item in range(self.n_items):
            cur_layer = 0

            # first level
            neighboor = self.pre_adj_mat_csr[item + self.n_users,:].tocoo()
            uniq_neighboor_id, uniq_neighboor_idx = np.unique(neighboor.col,return_inverse=True)
            adj = sp.coo_matrix((neighboor.data,(neighboor.row,uniq_neighboor_idx)),shape=(1,len(uniq_neighboor_id)))

            # save
            self.graph_neighboor[cur_layer].append(uniq_neighboor_id)
            self.graph_adj[cur_layer].append(adj)

            cur_layer = cur_layer+1
            # second layer
            while cur_layer < n_layers-1:
                neighboor = self.pre_adj_mat_csr[self.graph_neighboor[cur_layer-1][item + self.n_users],:].tocoo()
                uniq_neighboor_id, uniq_neighboor_idx = np.unique(neighboor.col,return_inverse=True)
                adj = sp.coo_matrix((neighboor.data,(neighboor.row,uniq_neighboor_idx)),shape=(self.graph_neighboor[cur_layer-1][item + self.n_users].size,len(uniq_neighboor_id)))
                # save
                self.graph_neighboor[cur_layer].append(uniq_neighboor_id)
                self.graph_adj[cur_layer].append(adj)
                cur_layer = cur_layer + 1

            adj = self.pre_adj_mat_csr[self.graph_neighboor[cur_layer-1][item + self.n_users],:].tocoo()
            self.graph_adj[cur_layer].append(adj)


            non_zero_adj.append(len(self.graph_adj[n_layers-1][item + self.n_users].data))
            count.append(len(self.graph_neighboor[n_layers-2][item + self.n_users]))
        logging.info("item tree: average last-1 layer node %d" %(sum(count)/len(count)))
        logging.info("item tree: nnz last adj max %d, mean %d" %(max(non_zero_adj),sum(non_zero_adj)/len(non_zero_adj)))


    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil() # sparse to list of link
        R = self.R.tolil() # sparse
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        logging.info('already create adjacency matrix'+str(adj_mat.shape)+ str(time() - t1))

        t2 = time()
        def normalized_adj_single(adj): # row normalized
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            logging.info('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            logging.info('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        logging.info('already normalize adjacency matrix'+str(time() - t2))
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def one_subtree(self,root,root_mode,feed_dict): # concatenate all neighboors and adjs of one instance
        # concatenate all neighboors in n_layers:
        if root_mode[2:] == 'pos_' or root_mode[2:] =='neg_':
            root = root + self.n_users

        root_mode = root_mode[2:] + root_mode[:2] # user_0_

        # neighboor_user_0_0
        feed_dict['neighboor_%s0'%(root_mode)]=np.array([root])

        for layer in range(len(self.graph_neighboor)): # 2
            feed_dict['neighboor_%s%d'%(root_mode,layer+1)]=self.graph_neighboor[layer][root]
            cur_adj = self.graph_adj[layer][root]

            feed_dict['adj_values_%s%d'%(root_mode,layer)]=cur_adj.data
            feed_dict['adj_indices_%s%d'%(root_mode,layer)]=np.stack([cur_adj.row, cur_adj.col], axis=1)
            feed_dict['adj_dense_shape_%s%d'%(root_mode,layer)]=np.array(cur_adj.shape)
        
        # add the last adj
        layer = len(self.graph_neighboor)
        cur_adj = self.graph_adj[layer][root]
        feed_dict['adj_values_%s%d'%(root_mode,layer)]=cur_adj.data
        feed_dict['adj_indices_%s%d'%(root_mode,layer)]=np.stack([cur_adj.row, cur_adj.col], axis=1)
        feed_dict['adj_dense_shape_%s%d'%(root_mode,layer)]=np.array([cur_adj.shape[0],self.n_users+self.n_items])

        # not useful for one tree
#        if len(data)>NNZ:
#            return False
        return True

    def make_pkl(self):
        logging.info("make pickle!!")
        
        neighboor = []
        neighboor_size = []

        indice = []
        value = []
        dense_shape = []
        indice_size = []

        layer_num = 2
        for layer in range(layer_num): #2 layers for neighboors
            if True:#layer < layer_num - 1:
                neighboor.append(np.concatenate(self.graph_neighboor[layer]).astype(np.int32))
                neighboor_size.append(np.array([_.size for _ in self.graph_neighboor[layer]],'i')) # Noted by Yinan, int32

            for root in range(self.n_users + self.n_items):
                adj = self.graph_adj[layer][root]
                indice.extend([adj.row.astype(np.int32), adj.col.astype(np.int32)])
                value.append(adj.data.astype(np.float32))
                dense_shape.append(np.array([adj.shape[0], adj.shape[1]], np.int32))
                indice_size.append(np.array([adj.col.size], np.int32))
    
        neighboor_all = np.concatenate(neighboor)
        neighboor_size_all = np.concatenate(neighboor_size)
        indice_all = np.concatenate(indice)
        value_all = np.concatenate(value)
        dense_shape_all = np.concatenate(dense_shape)
        indice_size_all = np.concatenate(indice_size)
 
        os.system('mkdir data_bin')
        neighboor_all.tofile('data_bin/neighboor.bin')
        neighboor_size_all.tofile('data_bin/neighboor_size.bin')
        indice_all.tofile('data_bin/indice.bin')
        value_all.tofile('data_bin/value.bin')
        dense_shape_all.tofile('data_bin/dense_shape.bin')
        indice_size_all.tofile('data_bin/indice_size.bin')
        pdb.set_trace()


    def batch_subtree(self,roots,root_mode,feed_dict): # concatenate all neighboors and adjs of one batch instances
        # concatenate all neighboors in n_layers:
        if root_mode[2:] == 'pos_' or root_mode[2:] =='neg_':
            roots = [root + self.n_users for root in roots]
        feed_dict['%sneighboor_0'%(root_mode)]=np.array(roots)

        for layer in range(len(self.graph_neighboor)): # 2
            neighboor_cur_layer = []

    def batch_subtree(self,roots,root_mode,feed_dict): # concatenate all neighboors and adjs of one batch instances
        # concatenate all neighboors in n_layers:
        if root_mode[2:] == 'pos_' or root_mode[2:] =='neg_':
            roots = [root + self.n_users for root in roots]
        feed_dict['%sneighboor_0'%(root_mode)]=np.array(roots)

        for layer in range(len(self.graph_neighboor)): # 2
            neighboor_cur_layer = []
            row_cur_layer = []
            col_cur_layer = []
            data_cur_layer = []
            shape0 = 0
            shape1 = 0
            for root in roots:
                neighboor_cur_layer.append(self.graph_neighboor[layer][root])
                cur_adj = self.graph_adj[layer][root]
                data_cur_layer.append(cur_adj.data)
                row_cur_layer.append(cur_adj.row + shape0)
                col_cur_layer.append(cur_adj.col + shape1)
                shape0 = shape0 + cur_adj.shape[0]
                shape1 = shape1 + cur_adj.shape[1]
            feed_dict['%sneighboor_%d'%(root_mode,layer+1)]=np.concatenate(neighboor_cur_layer,axis=0)

            data = np.concatenate(data_cur_layer,axis=0)
            row = np.concatenate(row_cur_layer,axis=0)
            col = np.concatenate(col_cur_layer,axis=0)
            indices = np.mat([row, col]).transpose()

            feed_dict['%sadj_indices_%d'%(root_mode,layer)]=indices
            feed_dict['%sadj_values_%d'%(root_mode,layer)]=data
            feed_dict['%sadj_dense_shape_%d'%(root_mode,layer)]=np.array([shape0,shape1])
        
        # add the last adj
        layer = len(self.graph_neighboor)
        row_cur_layer = []
        col_cur_layer = []
        data_cur_layer = []
        shape0 = 0

        for root in roots:
            cur_adj = self.graph_adj[layer][root]
            data_cur_layer.append(cur_adj.data)
            row_cur_layer.append(cur_adj.row + shape0)
            col_cur_layer.append(cur_adj.col)
            shape0 = shape0 + cur_adj.shape[0]

        data = np.concatenate(data_cur_layer,axis=0)
        row = np.concatenate(row_cur_layer,axis=0)
        col = np.concatenate(col_cur_layer,axis=0)
        indices = np.mat([row, col]).transpose()

        feed_dict['%sadj_indices_%d'%(root_mode,layer)]=indices
        feed_dict['%sadj_values_%d'%(root_mode,layer)]=data
        feed_dict['%sadj_dense_shape_%d'%(root_mode,layer)]=np.array([shape0,self.n_users+self.n_items])
 
        if len(data)>NNZ:
            return False
        return True


    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        logging.info('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        logging.info('n_interactions=%d' % (self.n_train + self.n_test))
        logging.info('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
        #logging.info('max_degree_user=%d, max_degree_item=%d' %(self.max_degree_user, self.max_degree_item))

