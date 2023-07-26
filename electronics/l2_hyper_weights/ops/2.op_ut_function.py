import time
import numpy as np
import tensorflow as tf

custom_ops = tf.load_op_library('ops/tree_out_load_one.so')
print('before statement')
my_func = custom_ops.tree_out
print('after statement')

root_modes = ['u', 'p', 'n']
gpu_num = 4

inputs = {}
idx = 0
batch_size = 256
per_gpu_size = batch_size//gpu_num

outputs = []
for root_mode in root_modes:
    for gpu_idx in range(gpu_num):
        print('at root_mode=%s, gpu=%d'%(root_mode, gpu_idx))
        per_input = tf.placeholder(tf.int32, [None], name="%s_%d"%(root_mode, gpu_idx))
        inputs[per_input] = np.arange(idx, idx+per_gpu_size, dtype=np.int32)

        outputs.append(my_func(per_input))
        idx += per_gpu_size

sess = tf.InteractiveSession()

dt = 0
for i in range(100):
    begin_t = time.time()
    out = sess.run(outputs, feed_dict=inputs)
    end_t = time.time()
    dt += end_t - begin_t
    print(i)

print('Avg time=%f'%(dt/1000.))
print('Done!')
