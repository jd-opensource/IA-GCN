import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

from utility.batch_test import *
import math
import os
import sys
import threading
import tensorflow as tf
from tensorflow.python.client import device_lib

from utility.helper import *
import pdb
from utility.debug import TFTimeline
import random
import numpy as np
import time as tm
from time import time
from tensorflow.python.client import timeline
from tensorflow.python.ops.sparse_ops import KeywordRequired

_NEG_INF = -1e9
_SOFTMAX_RATE = 1.0

# dataset_path = '/user/jd_ad/liuhu1/private/models/gcn_last/%s/l%d/data_bin'%(args.dataset, len(eval(args.layer_size)))
dataset_path = 'hdfs://ns1017/user/jd_ad/ads_search/train/quality_model/dynamic_gcn/open_dataset/%s/l%d/data_bin'%(args.dataset, len(eval(args.layer_size)))
if not os.path.exists('data_bin'):
    logging.info('start downloading dataset: %s...'%dataset_path)
    os.system('hadoop fs -get %s'%dataset_path)
    logging.info('download done')

custom_ops = tf.load_op_library('./ops/l%d/tree_out_load_more.so'%len(eval(args.layer_size)))

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
#os.environ['CUDA_VISIBLE_DEVICES']=''
batch_per_gpu = args.batch_size // 4

# For distributed
task_def = eval(os.getenv('TF_CONFIG'))['task']
job_name = task_def['type']
task_index = task_def['index']

cluster_def = eval(os.getenv('TF_CONFIG'))['cluster']
cluster = tf.train.ClusterSpec(cluster_def)
server = tf.train.Server(cluster,job_name=job_name,task_index=task_index)

def add_sync_queues_and_barrier(cluster, task_index, name_prefix,
                                enqueue_after_list):
    num_workers = cluster.num_tasks('worker')
    logging.info('num workers: %d'%num_workers)
    with tf.device('/job:ps/task:0/cpu:0'):
        sync_queues = [
            tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                         shared_name='%s%s' % (name_prefix, i))
            for i in range(num_workers)]
        queue_ops = []
        # For each other worker, add an entry in a queue, signaling that it can
        # finish this step.
        token = tf.constant(False)
        with tf.control_dependencies(enqueue_after_list):
            for i, q in enumerate(sync_queues):
                if i == task_index:
                    queue_ops.append(tf.no_op())
                else:
                    queue_ops.append(q.enqueue(token))
        # Drain tokens off queue for this worker, one for each other worker.
        queue_ops.append(
            sync_queues[task_index].dequeue_many(len(sync_queues) - 1))

        return tf.group(*queue_ops)

output_size = 1
input_size = args.embed_size * output_size
hidden_size = 40

weights = {
    'wd1': [input_size, hidden_size],
    'wd2': [hidden_size, output_size],
}

biases = {
    'bd1': [hidden_size],
    'bd2': [output_size],
}

def Weights(name):
    return tf.get_variable(name,dtype=tf.float32,shape=weights[name],
        initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

def Biases(name):
    return tf.get_variable(name,dtype=tf.float32,initializer=tf.zeros(biases[name]) \
        if name!='bd2' else tf.ones(biases[name]))

class LightGCN(object):
    def __init__(self, data_config, pretrain_data, is_test=False, global_step=None):
        # argument settings
        self.model_type = 'LightGCN'
        self.is_test = is_test
        if not is_test:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 1 
        self.norm_adj = data_config['norm_adj']
#        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size # 2048
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)


        def py_gen():
#            u_p = list(data_generator.train_u_p_pair)
            num_workers = cluster.num_tasks('worker')

            seed = 1234
            while True:
                # sample a big batch across workers
                rd_this = random.Random(seed)
                seed += 1

                users_all = rd_this.sample(data_generator.exist_users, self.batch_size*num_workers)
                users_this = users_all[self.batch_size*task_index: self.batch_size*(task_index+1)]
#                u_p_all = rd_this.sample(u_p, self.batch_size*num_workers)
#                u_p_this = u_p_all[self.batch_size*task_index: self.batch_size*(task_index+1)]

                u_list = [0]*self.batch_size
                p_list = [0]*self.batch_size
                n_list = [0]*self.batch_size

                for idx, u in enumerate(users_this):
                    u_list[idx] = u
                    p_list[idx] = random.choice(data_generator.train_items[u]) + data_generator.n_users
                    n_list[idx] = random.choice(data_generator.neg_pools[u]) + data_generator.n_users

                yield (u_list, p_list, n_list)

        def op_module(u_list, p_list, n_list):
            inputs = []
            inputs.extend([u_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(len(gpus))])
            inputs.extend([p_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(len(gpus))])
            inputs.extend([n_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(len(gpus))])

            outputs = custom_ops.tree_out(*inputs)

            #"gpu0_layer0_neighboor_user"
            keys = ["gpu%d_layer%d_%s_%s"%(gpu_idx, layer_idx, type_mode, root_mode) \
                for root_mode in ["user", "pos", "neg"]
                for gpu_idx in [0, 1, 2, 3]
                for layer_idx in [0, 1, 2]
                for type_mode in (
                    ["neighboor", "adj_indices", "adj_values", "adj_dense_shape"] if layer_idx == 0
                    else ["neighboor", "neighboor_seg", "adj_indices", "adj_values", "adj_dense_shape"] if layer_idx < self.n_layers
                        else ["neighboor", "neighboor_seg"])]
#                ["neighboor", "neighboor_seg", "adj_indices", "adj_values", "adj_dense_shape"] \
#                    if layer_idx != 0 else ["neighboor", "adj_indices", "adj_values", "adj_dense_shape"])]
            logging.info('output_keys: %s'% keys)
            for (k, v) in zip(keys, outputs):
                logging.info("k:%s, v:%s, typev:%s"%(k, str(v), type(v)))

            return dict(zip(keys, outputs))


        if not self.is_test:
            Dataset = tf.data.Dataset
            # get indices
            ds = Dataset.from_generator(py_gen, (tf.int32, tf.int32, tf.int32), ([None], [None], [None]))
            ds = ds.prefetch(15)

            # preprocess
            ds = ds.map(op_module, num_parallel_calls=8)
            ds = ds.prefetch(8)
            self.Q = ds.make_one_shot_iterator().get_next()
            self._build_model()
        else:
            self._build_test_model()

    def get_degree_inv(self):
        # already csr
        degrees = []
        for i in range(self.n_users + self.n_items):
            degrees.append(self.norm_adj[i].indices.shape[0])
        degrees = np.array(degrees, np.float32)
        inv = np.power(degrees, -0.5)
        return inv

    def weighted_sum_all_layers(self, emb_layers):
        # emb_layers: (batch, layer+1, dim)
        emb_layers_batch_large = tf.reshape(tf.stop_gradient(emb_layers), [-1, self.emb_dim]) # (batch*(layer+1), dim)

        fc1 = tf.add(tf.matmul(emb_layers_batch_large, self.wd1), self.bd1)
        fc1 = tf.nn.relu(fc1)
        gate_logits = tf.add(tf.matmul(fc1, self.wd2), self.bd2)
        gate_logits = tf.reshape(gate_logits, [-1, self.n_layers + 1])
        gates = tf.nn.softmax(gate_logits)

        emb_layers = tf.reduce_sum(emb_layers * gates[:, :, None], axis=1) # (batch, dim)
        return emb_layers

    def weighted_sum_all_layers(self, emb_layers):
        gate_logits = tf.reshape(self.learned_weights, [-1, self.n_layers + 1])
        gates = tf.nn.softmax(gate_logits)
        emb_layers = tf.reduce_sum(emb_layers * gates[:, :, None], axis=1) # (batch, dim) 
        return emb_layers

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.Tensor(indices), tf.Tensor(coo.shape)
        #tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_lightgcn_embed(self, guide_in):
        guides = tf.nn.embedding_lookup(self.all_weights, guide_in) # (n_acc, dim)
        coo = self.norm_adj.tocoo()

        coo_row = tf.constant(coo.row, dtype=tf.int64)
        coo_col = tf.constant(coo.col, dtype=tf.int64)
        indices = tf.stack([coo_row, coo_col], axis=1)     # (nnz, 2)
        coo_shape = tf.constant(coo.shape, dtype=tf.int64) # (2)

        indices_all = tf.concat([indices+coo_shape[None]*i for i in range(self.n_acc)], axis=0)
        coo_shape_all = coo_shape * self.n_acc

        # calculate adj
        values = tf.matmul(self.all_weights, guides, transpose_b=True) # (7w, n_acc)
        ## softmax
        values = tf.nn.embedding_lookup(values, coo_col) # (nnz, n_acc)
#        values = tf.ones_like(values)
        degree_inv_tile = tf.reshape(tf.tile(self.degree_inv[None], (self.n_acc, 1)), [-1, 1])

        #ego_embeddings *= degree_inv_tile
        ego_embeddings = tf.reshape(tf.tile(self.all_weights[None], (self.n_acc, 1, 1)), [-1, self.emb_dim])
        all_embeddings = [ego_embeddings]
        values_l = self.sparse_softmax(values, coo_row, self.n_users+self.n_items, 0)
        values_l = tf.transpose(values_l, [1, 0]) # (n_acc, nnz)
        adj_u = tf.SparseTensor(indices_all, tf.reshape(values_l, [-1]), coo_shape_all)
        # adj_u_fold = tf.sparse_split(keyword_required=KeywordRequired(),sp_input=adj_u,num_split=self.n_fold, axis=0)
        for k in range(0, self.n_layers):
            # one softmax rate per layer
            # gather n layers
            ## n_acc, 7w, 64
            """
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(adj_u_fold[f], ego_embeddings))
            ego_embeddings = tf.concat(temp_embed, 0)
            """
            ego_embeddings = tf.sparse_tensor_dense_matmul(adj_u, ego_embeddings)
            #all_embeddings += [ego_embeddings]
            all_embeddings += [ego_embeddings/degree_inv_tile]

        all_embeddings = tf.stack(all_embeddings, 1) # n_acc*7w, n_layer+1, dim
        all_embeddings = self.weighted_sum_all_layers(all_embeddings) # n_acc*7w, dim
        # all_embeddings = tf.reduce_mean(all_embeddings, axis=1) # n_acc*7w, dim # Noted by Yinan, close weighted layer combination
        all_embeddings = tf.reshape(all_embeddings, [self.n_acc, -1, self.emb_dim])

        return all_embeddings

    def _build_test_model(self):
        self.n_acc = 12
        with tf.device(cpus[0]):
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('cpu_variables', reuse=True):
                self.all_weights = tf.concat([
                    tf.get_variable(name='embedding_u',initializer =initializer([self.n_users, self.emb_dim])),
                    tf.get_variable(name='embedding_i',initializer =initializer([self.n_items, self.emb_dim]))],
                    axis=0)
                self.learned_weights = tf.get_variable(name='learnable_layer_weights', shape=[1+self.n_layers], initializer=tf.zeros_initializer())

                self.wd1 = Weights('wd1')
                self.bd1 = Biases('bd1')

                self.wd2 = Weights('wd2')
                self.bd2 = Biases('bd2')
                self.degree_inv = tf.constant(self.get_degree_inv())

        # for calc user guided embeddings
        self.guides_u = []
        self.embs_i = []

        self.guides_i = []  # (n_acc) * 4
        self.input_embs_i = [] # (u_stride, n_acc, 64) * 4
        self.u_stride = tf.placeholder(tf.int32, shape=(None)) # (u_stride)
        self.output_ts = [] # (u_stride, 4*n_acc)

        for gpu_idx in range(len(gpus)):
            with tf.device(gpus[gpu_idx]):
                guide_u = tf.placeholder(tf.int32, shape=(self.n_acc))
                self.guides_u.append(guide_u)
                all_embeddings = self._create_lightgcn_embed(tf.minimum(guide_u, self.n_users - 1))

                embeddings_u, embeddings_i = tf.split(all_embeddings, [self.n_users, self.n_items], 1)
                self.embs_i.append(embeddings_i) # (n_acc, 4w, 64)

                #####################################
                guide_i = tf.placeholder(tf.int32, shape=(self.n_acc))
                self.guides_i.append(guide_i)
                all_embeddings = self._create_lightgcn_embed(
                    tf.minimum(guide_i, self.n_users + self.n_items - 1)) # (n_acc, 7w, 64)

                input_emb_i = tf.placeholder(tf.float32, shape=(None, self.n_acc, self.emb_dim)) # (u_stride, n_acc, dim)
                self.input_embs_i.append(input_emb_i)

                embeddings_u, embeddings_i = tf.split(all_embeddings, [self.n_users, self.n_items], 1) # (n_acc, 4w, 64)
                embeddings_u = tf.nn.embedding_lookup(tf.transpose(embeddings_u, [1, 0, 2]), self.u_stride) # (u_stride, n_acc, dim)

                rate = tf.reduce_sum(input_emb_i*embeddings_u, axis=-1) # (u_stride, n_acc) 
                self.output_ts.append(rate)

        self.output_ts = tf.concat(self.output_ts, axis=1) # (u_stride, 4*n_acc)
                

    def sparse_softmax(self, weights, seg_rows, num_segments, layer_id=0):
#        softmax_rate_all = [0.1, 1.0]
#        softmax_rate = softmax_rate_all[layer_id]
        softmax_rate = 0.1

        weights_segmax = tf.unsorted_segment_max(weights, segment_ids=seg_rows, num_segments=num_segments)
        weights -= tf.nn.embedding_lookup(weights_segmax, seg_rows)
        weights = tf.exp(weights*_SOFTMAX_RATE*softmax_rate)

        weights_segsum = tf.unsorted_segment_sum(weights, segment_ids=seg_rows, num_segments=num_segments)
        weights /= tf.nn.embedding_lookup(weights_segsum, seg_rows)
        return weights


    def _build_model(self):
        with tf.device(cpus[0]):
            # len == # devices
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('cpu_variables', reuse=False):
                self.all_weights = tf.concat([
                    tf.get_variable(name='embedding_u',initializer =initializer([self.n_users, self.emb_dim])),
                    tf.get_variable(name='embedding_i',initializer =initializer([self.n_items, self.emb_dim]))],
                    axis=0)
                #self.all_weights = tf.get_variable(name='embedding',initializer =initializer([self.n_users + self.n_items, self.emb_dim]))
                self.learned_weights = tf.get_variable(name='learnable_layer_weights', shape=[1+self.n_layers], initializer=tf.zeros_initializer())
                
                self.wd1 = Weights('wd1')
                self.bd1 = Biases('bd1')

                self.wd2 = Weights('wd2')
                self.bd2 = Biases('bd2')
                self.degree_inv = tf.constant(self.get_degree_inv()) # Noted by Yinan, D**(-1/2), a vector-like constant tensor 

            self.loss_all_device = []
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            grad_dense_all_device = []

            for device_id in range(len(gpus)):
                with tf.device(gpus[device_id]):
                    # back to sparse_tensors, part * layers
                    e_g_homo = {}
                    root_upper = {}
                    guide_idx_homo = {}

                    embedding_gs = {}
                    embedding_g_pres = {}
                    all_weights = tf.identity(self.all_weights)
                    all_weights_inv = all_weights# * self.degree_inv[:, None]

                    for root_mode in ['user', 'pos', 'neg']:
                        guide_idx_homo[root_mode] = self.Q['gpu%d_layer%d_neighboor_%s'%(device_id, 0, root_mode)]
                        e_g_homo[root_mode] = tf.nn.embedding_lookup(all_weights,
                            guide_idx_homo[root_mode])
                        root_upper[root_mode] = tf.nn.embedding_lookup(self.degree_inv[:, None, None], guide_idx_homo[root_mode])

                        '''
                        adjs[root_mode] = [{'indices': self.Q['gpu%d_layer%d_adj_indices_%s'%(device_id, _, root_mode)],
                            'dense_shape': self.Q['gpu%d_layer%d_adj_dense_shape_%s'%(device_id, _, root_mode)]}
                            for _ in range(self.n_layers)]
                        '''

                    for root_mode in ['pos', 'neg']:
                        for (guide, root) in [('user', root_mode), (root_mode, 'user')]:
                            cur_layer = self.n_layers # 2
                            uniq_embedding_last = tf.nn.embedding_lookup(all_weights_inv,
                                self.Q['gpu%d_layer%d_neighboor_%s'%(device_id, cur_layer, root)])
                            cur_layer -= 1

                            while cur_layer >= 0:
                                if cur_layer != 0:
                                    uniq_embedding_higher = tf.nn.embedding_lookup(all_weights_inv,
                                        self.Q['gpu%d_layer%d_neighboor_%s'%(device_id, cur_layer, root)])
                                else:
                                    uniq_embedding_higher = tf.nn.embedding_lookup(all_weights,
                                        self.Q['gpu%d_layer%d_neighboor_%s'%(device_id, cur_layer, root)])

                                higher_num = tf.shape(uniq_embedding_higher)[0]

                                # (nnz, 2), (0~layer-1 neighboor, 7w)
                                seg_indices = self.Q['gpu%d_layer%d_adj_indices_%s'%(device_id, cur_layer, root)]
                                seg_rows = tf.reshape(seg_indices[:, 0], [-1])

                                seg_neighboors = self.Q['gpu%d_layer%d_neighboor_seg_%s'%(device_id, cur_layer+1, root)] # (layer neighboor)
                                e_g_gather = tf.nn.embedding_lookup(e_g_homo[guide], seg_neighboors)
                                # using 0-order feature
                                weights = tf.reduce_sum(uniq_embedding_last[:, :self.emb_dim]*e_g_gather, axis=-1) # (layer neighboor)

                                # mask self
                                if cur_layer % 2 == 0:
                                    neighboors = self.Q['gpu%d_layer%d_neighboor_%s'%(device_id, cur_layer+1, root)]
                                    guides = tf.nn.embedding_lookup(guide_idx_homo[guide],
                                        seg_neighboors) 
                                    # Noted by Yinan, when a child of a user (item) tree equals the root of corresponding item (user) tree, assign a larger weight
                                    weights_mask = tf.cast(tf.equal(neighboors, guides), tf.float32) * _NEG_INF
                                    weights += weights_mask

                                # layer neighboor to nnz
                                weights = tf.nn.embedding_lookup(weights, seg_indices[:, 1])
                                weights = tf.reshape(weights, [-1])

                                ## share ops
                                # softmax, noted by Yinan, weight normalization
#                                weights = tf.ones_like(weights) 
                                weights = self.sparse_softmax(weights, seg_rows, higher_num, cur_layer)
                                '''
                                pt_op = tf.Print(weights, [tf.shape(weights), weights], message="debugging weights, layer: %d"%cur_layer)
                                with tf.control_dependencies([pt_op]):
                                    weights = tf.identity(weights)
                                '''

                                cur_adj = tf.SparseTensor(
                                    indices=seg_indices,
                                    values=weights,
                                    dense_shape=self.Q['gpu%d_layer%d_adj_dense_shape_%s'%(device_id, cur_layer, root)])

                                gathered = tf.sparse_tensor_dense_matmul(cur_adj, uniq_embedding_last)
                                uniq_embedding_last = tf.concat([uniq_embedding_higher, gathered], axis=1)
                                cur_layer = cur_layer - 1

                            embedding_all = tf.reshape(uniq_embedding_last, [-1, self.n_layers+1, self.emb_dim])
                            embedding_all = tf.concat([embedding_all[:, 0, None], embedding_all[:, 1:]/root_upper[root]], axis=1)

                            embedding_gs['%s_%s'%(root, guide)] = self.weighted_sum_all_layers(embedding_all)
                            # embedding_gs['%s_%s'%(root, guide)] = tf.reduce_mean(embedding_all, axis=1) # Noted by Yinan, close weighted layer combination
                            #tf.reshape(embedding_all, [-1, self.emb_dim]) #tf.reduce_mean(embedding_all, axis=1)
                            embedding_g_pres['%s_%s'%(root, guide)] = embedding_all[:, 0, :] # only the first layer

                    # calculate loss
                    mf_loss, emb_loss, reg_loss = self.create_bpr_loss(
                        embedding_gs['user_pos'], embedding_gs['pos_user'],
                        embedding_gs['user_neg'], embedding_gs['neg_user'],
                        embedding_g_pres['user_pos'], embedding_g_pres['pos_user'], embedding_g_pres['neg_user'])
                    
                    # Noted by Yinan, close weighted layer combination
                    # reg_loss += tf.nn.l2_loss(self.wd1) * self.decay / 10
                    # reg_loss += tf.nn.l2_loss(self.wd2) * self.decay / 10

                    loss = mf_loss + emb_loss + reg_loss
                    self.loss_all_device.append((mf_loss, emb_loss, reg_loss, loss))

                    # calculate grads
                    sparse_grad_and_vars = opt.compute_gradients(loss) # sparse gradients: IndexedSlices #v * (g,v)
                    dense_grad_and_vars = [] # convert sparse gradients to dense Tensor: # v * (g,v)
                    for g,v in sparse_grad_and_vars:
                        if g is None: # because we close weighted layer combination
                            continue
                        elif type(g)==tf.IndexedSlices:
                            g_dense = tf.unsorted_segment_sum(g.values,g.indices, num_segments=g.dense_shape[0])
                            dense_grad_and_vars.append((g_dense,v))
                        else:
                            dense_grad_and_vars.append((g,v))
                    grad_dense_all_device.append(dense_grad_and_vars)

            # in cpu
            mf_loss_all_device, emb_loss_all_device, reg_loss_all_device, sum_loss_all_device = zip(*self.loss_all_device)

            self.mf_loss = tf.reduce_mean(mf_loss_all_device)
            self.emb_loss = tf.reduce_mean(emb_loss_all_device)
            self.reg_loss = tf.reduce_mean(reg_loss_all_device)
            self.loss = tf.reduce_mean(sum_loss_all_device)

            grads_averaged = self.average_gradients(grad_dense_all_device)
            if not self.is_test:
                rep_op = tf.train.SyncReplicasOptimizer(opt,
                    replicas_to_aggregate=len(cluster_def['worker']),
                    #replica_id=task_index,
                    total_num_replicas=len(cluster_def['worker']),
                    use_locking=True)

                self.apply_gradient_op = rep_op.apply_gradients(grads_averaged,
                    global_step=self.global_step)
                self.init_token_op = rep_op.get_init_tokens_op()
                self.chief_queue_runner = rep_op.get_chief_queue_runner()

                main_fetch_group = tf.group(self.apply_gradient_op,
                    self.loss, self.mf_loss, self.emb_loss, self.reg_loss)
                self.sync_queues = add_sync_queues_and_barrier(cluster, task_index, 'sync_queues_step_end_',
                    [main_fetch_group])

                '''
                with tf.device('/job:ps/task:0/cpu:0'):
                    with tf.control_dependencies([main_fetch_group]):
                        self.inc_global_step = self.global_step.assign_add(1)
                '''
                self.init_op = tf.global_variables_initializer()#$initialize_all_variables()

    # used for dense gradients # average over all gpus
    def average_gradients(self,tower_grads):# gpu * v *(g,v)
        average_grads = []
        for grad_and_vars in zip(*tower_grads): # vars
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_and_vars] # devices
            # Average over the 'tower' dimension.
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
            logging.info('gradient var: %s' % v.name)
        return average_grads # v *(g,v)


    # never changed
    def create_model_str(self):
        log_dir = '/' + self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim)
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def create_bpr_loss(self, u_p, p_u, u_n, n_u,
        u_g_embeddings_pre, pos_i_g_embeddings_pre, neg_i_g_embeddings_pre):

        pos_scores = tf.reduce_sum(tf.multiply(u_p, p_u), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(u_n, n_u), axis=1)
        regularizer = tf.nn.l2_loss(u_g_embeddings_pre) + tf.nn.l2_loss(
                pos_i_g_embeddings_pre) + tf.nn.l2_loss(neg_i_g_embeddings_pre)
        regularizer = regularizer / batch_per_gpu
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss


# not used
def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        logging.info('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        f0 = time()
        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items

        """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
        plain_adj, norm_adj, mean_adj,pre_adj = data_generator.get_adj_mat()
        config['norm_adj']=pre_adj
        logging.info('use the pre adjcency matrix')

        if args.pretrain == -1:
            pretrain_data = load_pretrained_data()
        else:
            pretrain_data = None
        # pdb.set_trace()
        # start here
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster)):

            # train data, no reuse
            model = LightGCN(data_config=config, pretrain_data=pretrain_data)
            model_test = LightGCN(data_config=config, pretrain_data=pretrain_data,
                is_test=True)
            
        # define a saver
        saver = tf.train.Saver(max_to_keep=3)

        sv = tf.train.Supervisor(is_chief=(task_index == 0),
            init_op=model.init_op,
            summary_op=None,
            saver=None, # pass a saver
            global_step=model.global_step)

        NUM_PARALLEL_EXEC_UNITS = 16 * 2
        config_tf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        #config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    #    config.gpu_options.allow_growth = True

        #with sv.managed_session(master=server.target, config=config_tf) as sess:
        with sv.prepare_or_wait_for_session(server.target, config=config_tf) as sess:
            if task_index == 0:
                sv.start_queue_runners(sess, [model.chief_queue_runner])
                sess.run(model.init_token_op)

            # real start here
            cur_best_pre_0 = 0.
            loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, auc_loger = [], [], [], [], [], []
            stopping_step = 0
            should_stop = False

            t0 = time()
            for epoch in range(1, args.epoch + 1):
                t1 = time()
                loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.

                num_workers = cluster.num_tasks('worker')
                n_batch = data_generator.n_train // args.batch_size  # changed
                n_batch = n_batch // num_workers
                #n_batch = 2

                loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.

                #pdb.set_trace()
                logging.info('n_batch %d'%(n_batch))
                #pdb.set_trace()

                momentum = 0.99
                train_dt = 0.0

                tn = time()
                for idx in range(n_batch):
                    _, __, step, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                        [model.apply_gradient_op, model.sync_queues, model.global_step,
                        model.loss, model.mf_loss, model.emb_loss, model.reg_loss])

                    #pdb.set_trace()

                    loss += batch_loss/n_batch
                    mf_loss += batch_mf_loss/n_batch
                    emb_loss += batch_emb_loss/n_batch
                    reg_loss += batch_reg_loss/n_batch

                    dt = time()-tn
                    train_dt = train_dt * momentum + dt * (1-momentum)
#                    logging.info('training: %d:%d, loss=[%.5f=%.5f+%.5f+%.5f], dt_avg=%f, dt_now=%f'%(idx, step,
#                        batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss, train_dt, dt))
                    tn = time()

                if np.isnan(loss) == True:
                    logging.info('ERROR: loss is nan.')
                    sys.exit()

                # save model variables
                # if epoch==1000 and task_index == 0:
                #     saver.save(sess, '/export/App/training_platform/PinoModel/models/final-model-ckpt', global_step=model.global_step)
                #     logging.info("save final checkpoint")

                # # print training loss every epoch
                # if (epoch <=500) and (epoch % 100) != 0:
                #     # short print
                #     if args.verbose > 0 and epoch % args.verbose == 0:
                #         perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                #         logging.info(perf_str)
                #     continue
                # elif (epoch >500) and (epoch % 50) != 0:
                #     # short print
                #     if args.verbose > 0 and epoch % args.verbose == 0:
                #         perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                #         logging.info(perf_str)
                #     continue
                if (epoch % 20) != 0:
                    # short print
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                        logging.info(perf_str)
                    continue

                # only test every 20 epoch
                users_to_test = list(data_generator.train_items.keys())

                stride = int(math.ceil(len(users_to_test)/1.0/cluster.num_tasks('worker')))
                test_begin_idx, test_end_idx = stride*task_index, min(stride*(task_index+1), len(users_to_test))
                users_to_test = users_to_test[test_begin_idx: test_end_idx]

                ret = test(sess, model_test, users_to_test, test_begin_idx, test_end_idx, drop_flag=True, train_set_flag=1)
                perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s], auc=[%s]' % \
                           (epoch, loss, mf_loss, emb_loss, reg_loss,
                            ', '.join(['%.5f' % r for r in ret['recall']]),
                            ', '.join(['%.5f' % r for r in ret['precision']]),
                            ', '.join(['%.5f' % r for r in ret['ndcg']]),
                            '%.5f' % ret['auc'])
                logging.info(perf_str)

                t2 = time()
                ###############################
                users_to_test = list(data_generator.test_set.keys())

                stride = int(math.ceil(len(users_to_test)/1.0/cluster.num_tasks('worker')))
                test_begin_idx, test_end_idx = stride*task_index, min(stride*(task_index+1), len(users_to_test))
                users_to_test = users_to_test[test_begin_idx: test_end_idx]

                ret = test(sess, model_test, users_to_test, test_begin_idx, test_end_idx, drop_flag=True) # on test set

                t3 = time()
                loss_loger.append(loss)
                rec_loger.append(ret['recall'])
                pre_loger.append(ret['precision'])
                ndcg_loger.append(ret['ndcg'])
                auc_loger.append(ret['auc'])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%s], ' \
                        'precision=[%s], ndcg=[%s], auc=[%s]' % \
                        (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test,
                        ', '.join(['%.5f' % r for r in ret['recall']]),
                        ', '.join(['%.5f' % r for r in ret['precision']]),
                        ', '.join(['%.5f' % r for r in ret['ndcg']]),
                        '%.5f' % ret['auc'])
                    logging.info(perf_str)

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                    stopping_step, expected_order='acc', flag_step=30)

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                if should_stop == True:
                    break

            recs = np.array(rec_loger)
            pres = np.array(pre_loger)
            ndcgs = np.array(ndcg_loger)
            aucs = np.array(auc_loger)

            best_rec_0 = max(recs[:, 0])
            idx = list(recs[:, 0]).index(best_rec_0)

            final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s], auc=[%s]" % \
                                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                                  '%.5f' % aucs[idx])
            logging.info(final_perf)

            save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
            ensureDir(save_path)
            f = open(save_path, 'a')

            f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
                        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
                           args.adj_type, final_perf))
            f.close()
