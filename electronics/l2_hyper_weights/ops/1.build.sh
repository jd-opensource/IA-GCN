#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo ${TF_CFLAGS[@]}
echo ${TF_LFLAGS[@]}

g++ -std=c++11 tree_out_load.cc -shared -fPIC \
     ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -Wl,--no-undefined \
    -fno-strict-aliasing -DLINUX_ -DOC_NEW_STYLE_INCLUDES -Wno-deprecated \
    -pthread -D_REENTRANT -o tree_out_load.so -O2
