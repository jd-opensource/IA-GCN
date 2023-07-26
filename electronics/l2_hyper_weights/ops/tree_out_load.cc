#include "tensorflow/core/framework/op_kernel.h"
/*
#include "ocport.h"
#define MIDAS_COMPILER_TEMPLATES
#include "valpython.h"
#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
const int n_user = 29858;
const int n_item = 40981;
const int layer_num = 3;

void load_size(int **size, int layer_num, const char* file_name)
{
    //LOG(INFO) << "load size: " << file_name << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    for(int i=0; i<layer_num; i++)
    {
        size[i] =(int*) malloc(sizeof(int)*(n_user+n_item));
        fread(size[i], sizeof(int), n_user+n_item, fd);
    }
    fclose(fd);
}

void load_data_int(int ***data, int **data_size, int layer_num, const char* file_name)
{
    //LOG(INFO) << "load data int: " << file_name << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    int rate = 1;
    if (strcmp(file_name, "data_bin/indice.bin") == 0)
        rate = 2;

    for (int i=0; i<layer_num; i++){
        data[i] = (int **)(malloc(sizeof(int*)*(n_user+n_item)));
        for (int root=0; root<n_user+n_item; root++){
            data[i][root] = (int*) malloc(sizeof(int)*data_size[i][root]*rate);
            fread(data[i][root], sizeof(int), data_size[i][root]*rate, fd);
        }
    }
    fclose(fd);
}

void load_data_float(float ***data, int **data_size, int layer_num, const char* file_name)
{
    //LOG(INFO) << "load data float: " << file_name << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    for (int i=0; i<layer_num; i++){
        data[i] = (float **)(malloc(sizeof(float*)*(n_user+n_item)));
        for (int root=0; root<n_user+n_item; root++){
            data[i][root] = (float*) malloc(sizeof(float)*data_size[i][root]);
            fread(data[i][root], sizeof(float), data_size[i][root], fd);
        }
    }
    fclose(fd);
}

void load_dense_shape(int ***data, int layer_num, const char* file_name)
{
    //LOG(INFO) << "load dense_shape: " << file_name << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    for (int i=0; i<layer_num; i++){
        data[i] = (int **)(malloc(sizeof(int*)*(n_user+n_item)));
        for (int root=0; root<n_user+n_item; root++){
            data[i][root] = (int*) malloc(sizeof(int)*2);
            fread(data[i][root], sizeof(int), 2, fd);
        }
    }
    fclose(fd);
}

std::string format(std::string root_mode, std::string type_mode, int gpu_num, int layer_num)
{
    std::ostringstream stringStream;
    stringStream << root_mode << "_gpu" << gpu_num << "_layer" << layer_num << "_" << type_mode;
    std::string copyOfStr = stringStream.str();
    return copyOfStr;
}

std::string format_input(std::string root_mode, int gpu_num)
{
    std::ostringstream stringStream;
    stringStream << root_mode << "_input" << gpu_num;
    std::string copyOfStr = stringStream.str();
    return copyOfStr;
}

using namespace tensorflow;
REGISTER_OP("TreeOut")
  .Input("u_input0: int32")
  .Input("u_input1: int32")
  .Input("u_input2: int32")
  .Input("u_input3: int32")
  .Input("p_input0: int32")
  .Input("p_input1: int32")
  .Input("p_input2: int32")
  .Input("p_input3: int32")
  .Input("n_input0: int32")
  .Input("n_input1: int32")
  .Input("n_input2: int32")
  .Input("n_input3: int32")
  .Output("u_gpu0_layer0_neighboor: int32")
  .Output("u_gpu0_layer0_adj_indices: int64")
  .Output("u_gpu0_layer0_adj_values: float32")
  .Output("u_gpu0_layer0_adj_dense_shape: int64")
  .Output("u_gpu0_layer1_neighboor: int32")
  .Output("u_gpu0_layer1_adj_indices: int64")
  .Output("u_gpu0_layer1_adj_values: float32")
  .Output("u_gpu0_layer1_adj_dense_shape: int64")
  .Output("u_gpu0_layer2_neighboor: int32")
  .Output("u_gpu0_layer2_adj_indices: int64")
  .Output("u_gpu0_layer2_adj_values: float32")
  .Output("u_gpu0_layer2_adj_dense_shape: int64")
  .Output("u_gpu1_layer0_neighboor: int32")
  .Output("u_gpu1_layer0_adj_indices: int64")
  .Output("u_gpu1_layer0_adj_values: float32")
  .Output("u_gpu1_layer0_adj_dense_shape: int64")
  .Output("u_gpu1_layer1_neighboor: int32")
  .Output("u_gpu1_layer1_adj_indices: int64")
  .Output("u_gpu1_layer1_adj_values: float32")
  .Output("u_gpu1_layer1_adj_dense_shape: int64")
  .Output("u_gpu1_layer2_neighboor: int32")
  .Output("u_gpu1_layer2_adj_indices: int64")
  .Output("u_gpu1_layer2_adj_values: float32")
  .Output("u_gpu1_layer2_adj_dense_shape: int64")
  .Output("u_gpu2_layer0_neighboor: int32")
  .Output("u_gpu2_layer0_adj_indices: int64")
  .Output("u_gpu2_layer0_adj_values: float32")
  .Output("u_gpu2_layer0_adj_dense_shape: int64")
  .Output("u_gpu2_layer1_neighboor: int32")
  .Output("u_gpu2_layer1_adj_indices: int64")
  .Output("u_gpu2_layer1_adj_values: float32")
  .Output("u_gpu2_layer1_adj_dense_shape: int64")
  .Output("u_gpu2_layer2_neighboor: int32")
  .Output("u_gpu2_layer2_adj_indices: int64")
  .Output("u_gpu2_layer2_adj_values: float32")
  .Output("u_gpu2_layer2_adj_dense_shape: int64")
  .Output("u_gpu3_layer0_neighboor: int32")
  .Output("u_gpu3_layer0_adj_indices: int64")
  .Output("u_gpu3_layer0_adj_values: float32")
  .Output("u_gpu3_layer0_adj_dense_shape: int64")
  .Output("u_gpu3_layer1_neighboor: int32")
  .Output("u_gpu3_layer1_adj_indices: int64")
  .Output("u_gpu3_layer1_adj_values: float32")
  .Output("u_gpu3_layer1_adj_dense_shape: int64")
  .Output("u_gpu3_layer2_neighboor: int32")
  .Output("u_gpu3_layer2_adj_indices: int64")
  .Output("u_gpu3_layer2_adj_values: float32")
  .Output("u_gpu3_layer2_adj_dense_shape: int64")
  .Output("p_gpu0_layer0_neighboor: int32")
  .Output("p_gpu0_layer0_adj_indices: int64")
  .Output("p_gpu0_layer0_adj_values: float32")
  .Output("p_gpu0_layer0_adj_dense_shape: int64")
  .Output("p_gpu0_layer1_neighboor: int32")
  .Output("p_gpu0_layer1_adj_indices: int64")
  .Output("p_gpu0_layer1_adj_values: float32")
  .Output("p_gpu0_layer1_adj_dense_shape: int64")
  .Output("p_gpu0_layer2_neighboor: int32")
  .Output("p_gpu0_layer2_adj_indices: int64")
  .Output("p_gpu0_layer2_adj_values: float32")
  .Output("p_gpu0_layer2_adj_dense_shape: int64")
  .Output("p_gpu1_layer0_neighboor: int32")
  .Output("p_gpu1_layer0_adj_indices: int64")
  .Output("p_gpu1_layer0_adj_values: float32")
  .Output("p_gpu1_layer0_adj_dense_shape: int64")
  .Output("p_gpu1_layer1_neighboor: int32")
  .Output("p_gpu1_layer1_adj_indices: int64")
  .Output("p_gpu1_layer1_adj_values: float32")
  .Output("p_gpu1_layer1_adj_dense_shape: int64")
  .Output("p_gpu1_layer2_neighboor: int32")
  .Output("p_gpu1_layer2_adj_indices: int64")
  .Output("p_gpu1_layer2_adj_values: float32")
  .Output("p_gpu1_layer2_adj_dense_shape: int64")
  .Output("p_gpu2_layer0_neighboor: int32")
  .Output("p_gpu2_layer0_adj_indices: int64")
  .Output("p_gpu2_layer0_adj_values: float32")
  .Output("p_gpu2_layer0_adj_dense_shape: int64")
  .Output("p_gpu2_layer1_neighboor: int32")
  .Output("p_gpu2_layer1_adj_indices: int64")
  .Output("p_gpu2_layer1_adj_values: float32")
  .Output("p_gpu2_layer1_adj_dense_shape: int64")
  .Output("p_gpu2_layer2_neighboor: int32")
  .Output("p_gpu2_layer2_adj_indices: int64")
  .Output("p_gpu2_layer2_adj_values: float32")
  .Output("p_gpu2_layer2_adj_dense_shape: int64")
  .Output("p_gpu3_layer0_neighboor: int32")
  .Output("p_gpu3_layer0_adj_indices: int64")
  .Output("p_gpu3_layer0_adj_values: float32")
  .Output("p_gpu3_layer0_adj_dense_shape: int64")
  .Output("p_gpu3_layer1_neighboor: int32")
  .Output("p_gpu3_layer1_adj_indices: int64")
  .Output("p_gpu3_layer1_adj_values: float32")
  .Output("p_gpu3_layer1_adj_dense_shape: int64")
  .Output("p_gpu3_layer2_neighboor: int32")
  .Output("p_gpu3_layer2_adj_indices: int64")
  .Output("p_gpu3_layer2_adj_values: float32")
  .Output("p_gpu3_layer2_adj_dense_shape: int64")
  .Output("n_gpu0_layer0_neighboor: int32")
  .Output("n_gpu0_layer0_adj_indices: int64")
  .Output("n_gpu0_layer0_adj_values: float32")
  .Output("n_gpu0_layer0_adj_dense_shape: int64")
  .Output("n_gpu0_layer1_neighboor: int32")
  .Output("n_gpu0_layer1_adj_indices: int64")
  .Output("n_gpu0_layer1_adj_values: float32")
  .Output("n_gpu0_layer1_adj_dense_shape: int64")
  .Output("n_gpu0_layer2_neighboor: int32")
  .Output("n_gpu0_layer2_adj_indices: int64")
  .Output("n_gpu0_layer2_adj_values: float32")
  .Output("n_gpu0_layer2_adj_dense_shape: int64")
  .Output("n_gpu1_layer0_neighboor: int32")
  .Output("n_gpu1_layer0_adj_indices: int64")
  .Output("n_gpu1_layer0_adj_values: float32")
  .Output("n_gpu1_layer0_adj_dense_shape: int64")
  .Output("n_gpu1_layer1_neighboor: int32")
  .Output("n_gpu1_layer1_adj_indices: int64")
  .Output("n_gpu1_layer1_adj_values: float32")
  .Output("n_gpu1_layer1_adj_dense_shape: int64")
  .Output("n_gpu1_layer2_neighboor: int32")
  .Output("n_gpu1_layer2_adj_indices: int64")
  .Output("n_gpu1_layer2_adj_values: float32")
  .Output("n_gpu1_layer2_adj_dense_shape: int64")
  .Output("n_gpu2_layer0_neighboor: int32")
  .Output("n_gpu2_layer0_adj_indices: int64")
  .Output("n_gpu2_layer0_adj_values: float32")
  .Output("n_gpu2_layer0_adj_dense_shape: int64")
  .Output("n_gpu2_layer1_neighboor: int32")
  .Output("n_gpu2_layer1_adj_indices: int64")
  .Output("n_gpu2_layer1_adj_values: float32")
  .Output("n_gpu2_layer1_adj_dense_shape: int64")
  .Output("n_gpu2_layer2_neighboor: int32")
  .Output("n_gpu2_layer2_adj_indices: int64")
  .Output("n_gpu2_layer2_adj_values: float32")
  .Output("n_gpu2_layer2_adj_dense_shape: int64")
  .Output("n_gpu3_layer0_neighboor: int32")
  .Output("n_gpu3_layer0_adj_indices: int64")
  .Output("n_gpu3_layer0_adj_values: float32")
  .Output("n_gpu3_layer0_adj_dense_shape: int64")
  .Output("n_gpu3_layer1_neighboor: int32")
  .Output("n_gpu3_layer1_adj_indices: int64")
  .Output("n_gpu3_layer1_adj_values: float32")
  .Output("n_gpu3_layer1_adj_dense_shape: int64")
  .Output("n_gpu3_layer2_neighboor: int32")
  .Output("n_gpu3_layer2_adj_indices: int64")
  .Output("n_gpu3_layer2_adj_values: float32")
  .Output("n_gpu3_layer2_adj_dense_shape: int64");

class TreeOutOp : public OpKernel {
 public:
  explicit TreeOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Need + 70000 for pos and negs
    LOG(INFO) << "Start initialize op..." << endl;

    // load_size
    load_size(neighboor_sizes, layer_num-1, "data_bin/neighboor_size.bin");
    load_size(indice_sizes, layer_num, "data_bin/indice_size.bin");

    // load neighboor, indices, values, dense_shape
    load_data_int(neighboors, neighboor_sizes, layer_num-1, "data_bin/neighboor.bin");
    load_data_int(indices, indice_sizes, layer_num, "data_bin/indice.bin");
    load_data_float(values, indice_sizes, layer_num, "data_bin/value.bin");
    load_dense_shape(dense_shapes, layer_num, "data_bin/dense_shape.bin");
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    //LOG(INFO) << "start compute" << endl;
    // Create an output tensor
    std::string root_modes[3] = {"u", "p", "n"};
    for (int root_id=0; root_id<3; root_id++){
      std::string root_mode = root_modes[root_id];
      for (int gpu = 0; gpu < 4; gpu++) {
        const Tensor* input_tensor;
        OP_REQUIRES_OK(context, context->input(format_input(root_mode, gpu), &input_tensor));
        auto input_ids = input_tensor->flat<int32>();
        const int N = input_ids.size();

        //LOG(INFO) << "in for: " << root_mode << gpu << endl;

        // neighboor_0
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
          format(root_mode, "neighboor", gpu, 0), {N}, &output));
        auto output_flat_neigh = output->flat<int32>();

        for (int i = 0; i < N; i++) // use roots
          output_flat_neigh(i) = input_ids(i);

        for (int layer = 0; layer < layer_num; layer++) {
          int count = 0;
          int idx = 0;
          if (layer < layer_num-1) {
            // neighboor (layer-1, num, varlen)
            // neighboor_size (layer-1, num)
            //LOG(INFO) << root_mode << "_gpu: " << gpu << ", layer: "<<layer << ", neighboor"<< endl;
            for (int i=0; i < N; i++) {count += neighboor_sizes[layer][input_ids(i)];}
              output = NULL;
              OP_REQUIRES_OK(context, context->allocate_output(
                format(root_mode, "neighboor", gpu, layer+1), {count}, &output));
              auto output_flat_neigh = output->flat<int32>();

              idx = 0;
              for (int i=0; i < N; i++) { // concatenate
                int* dat = neighboors[layer][input_ids(i)];
                for (int j=0; j<neighboor_sizes[layer][input_ids(i)]; j++) {
                  output_flat_neigh(idx) = dat[j];idx++;}
              }
          }

          // values (layer, num, varlen)
          //LOG(INFO) << root_mode << "_gpu: " << gpu << ", layer: "<<layer << ", values"<< endl;
          count = 0;
          for (int i=0; i < N; i++) {count += indice_sizes[layer][input_ids(i)];}

          output = NULL;
          OP_REQUIRES_OK(context, context->allocate_output(
            format(root_mode, "adj_values", gpu, layer), {count}, &output));
          auto output_flat_val = output->flat<float>();

          idx = 0;
          for (int i=0; i < N; i++) { // concatenate
            float* dat = values[layer][input_ids(i)];
            for (int j=0; j < indice_sizes[layer][input_ids(i)]; j++) {
              output_flat_val(idx) = dat[j];idx++;}
          }

          // indices (layer, num, 2*varlen)
          // shapes (layer, num, 2)
          //LOG(INFO) << root_mode << "_gpu: " << gpu << ", layer: "<<layer << ", indices and shapes"<< endl;

          output = NULL;
          OP_REQUIRES_OK(context, context->allocate_output(
            format(root_mode, "adj_indices", gpu, layer), {count, 2}, &output));
          auto output_flat_ind = output->flat<int64>();

          idx = 0;
          int shape0 = 0, shape1 = 0;
          for (int i=0; i < N; i++) {
            int* dat = indices[layer][input_ids(i)];
            int* dat_shape = dense_shapes[layer][input_ids(i)];

            int dat_size = indice_sizes[layer][input_ids(i)];

            for (int j=0; j < dat_size; j++) {
              output_flat_ind(idx) = shape0 + dat[j]; idx++;
              output_flat_ind(idx) = shape1 + dat[j+dat_size]; idx++;
            }
            shape0 = shape0 + dat_shape[0];
            if (layer < layer_num - 1)
                shape1 = shape1 + dat_shape[1];
          }

          output = NULL;
          OP_REQUIRES_OK(context, context->allocate_output(
            format(root_mode, "adj_dense_shape", gpu, layer), {2}, &output));
          auto output_flat_shape = output->flat<int64>();

          output_flat_shape(0) = shape0;
          if (layer < layer_num-1)
              output_flat_shape(1) = shape1;
          else
              output_flat_shape(1) = n_user + n_item;
        }
      }
    }
  }

 private:
  int *neighboor_sizes[layer_num-1];
  int **neighboors[layer_num-1];

  int *indice_sizes[layer_num];
  int **indices[layer_num];
  float **values[layer_num];
  int **dense_shapes[layer_num];
};

REGISTER_KERNEL_BUILDER(Name("TreeOut").Device(DEVICE_CPU), TreeOutOp);
