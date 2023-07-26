import time
import numpy as np
import tensorflow as tf

custom_ops = tf.load_op_library('ops/tree_out_load.so')

root_modes = ['u', 'p', 'n']
gpu_num = 4

inputs = {}
idx = 0
batch_size = 256
per_gpu_size = batch_size//gpu_num

for root_mode in root_modes:
    for gpu_idx in range(gpu_num):
        inputs[tf.placeholder(tf.int32, [None], name="%s_%d"%(root_mode, gpu_idx))] \
            = np.arange(idx, idx+per_gpu_size, dtype=np.int32)
        idx += per_gpu_size

outputs = custom_ops.tree_out(*inputs.keys())

sess = tf.InteractiveSession()

dt = 0
for i in range(1000):
    begin_t = time.time()
    out = sess.run(outputs, feed_dict=inputs)
    end_t = time.time()
    dt += end_t - begin_t
    print(i)

print('Avg time=%f'%(dt/1000.))
print('Done!')
