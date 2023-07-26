import pdb
import time
import numpy as np
import tensorflow as tf

custom_ops = tf.load_op_library('ops/tree_out_load.so')

root_modes = ['u', 'p', 'n']
layer_num = 3
gpu_num = 4

batch_size = 256
batch_per_gpu = batch_size//gpu_num

def py_gen():
    while True:
        inputs = {}
        idx = 0
        for root_mode in root_modes:
            inputs[root_mode] = np.arange(idx, idx+batch_size, dtype=np.int32)
            idx += batch_size
        yield tuple(inputs.values())

def op_module(u_list, p_list, n_list):
    inputs = []
    inputs.extend([u_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(gpu_num)])
    inputs.extend([p_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(gpu_num)])
    inputs.extend([n_list[i*batch_per_gpu: (i+1)*batch_per_gpu] for i in range(gpu_num)])

    outputs = custom_ops.tree_out(*inputs)

    #"gpu0_layer0_neighboor_user"
    keys = ["gpu%d_layer%d_%s_%s"%(gpu_idx, layer_idx, type_mode, root_mode) \
        for root_mode in ["user", "pos", "neg"] 
        for gpu_idx in range(gpu_num)
        for layer_idx in range(layer_num)
        for type_mode in ["neighboor", "adj_indices", "adj_values", "adj_dense_shape"]]
    print('output_keys', len(keys))
    return dict(zip(keys, outputs))

Dataset = tf.data.Dataset

# get indices
ds = Dataset.from_generator(py_gen, (tf.int32, tf.int32, tf.int32), ([None], [None], [None]))
ds = ds.prefetch(50)
# preprocess
ds = ds.map(op_module, num_parallel_calls=20)
ds = ds.prefetch(20)
Q = ds.make_one_shot_iterator().get_next() 

output_tf = tf.reduce_sum([tf.cast(v[0], tf.float32) if not "indices" in k \
    else tf.cast(v[0][0], tf.float32) for (k, v) in Q.items()])

#===========================
sess = tf.InteractiveSession()

dt = 0
skip_iters = 10
for i in range(skip_iters + 100):
    begin_t = time.time()
    out = sess.run(output_tf)
#    pdb.set_trace()
    end_t = time.time()
    if i >= skip_iters:
        dt += end_t - begin_t
    print(i, end_t-begin_t)

print('Avg time=%f'%(dt/100.))
print('Done!')
