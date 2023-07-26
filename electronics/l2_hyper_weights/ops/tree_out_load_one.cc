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

int** load_size(const int layer_num, const char* file_name)
{
    int* size[layer_num];
    LOG(INFO) << "load size: " << file_name << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    for(int i=0; i<layer_num; i++)
    {
        size[i] =(int*) malloc(sizeof(int)*(n_user+n_item));
        fread(size[i], sizeof(int), n_user+n_item, fd);
        for(int j=0;j<n_user+n_item; j++)
          LOG(INFO) << size[i][j] << " ";
        LOG(INFO) << endl;
    }
    fclose(fd);
    return size;
}

int*** load_data_int(int **data_size, const int layer_num, const char* file_name)
{
    int** data[layer_num];
    LOG(INFO) << "load data int: " << file_name << layer_num << endl;
    FILE *fd = fopen(file_name, "rb");
    if (fd == NULL){
        perror("open failed!");
        exit(1);
    }
    LOG(INFO) << "opened file" << file_name << endl;
    int rate = 1;
    if (strcmp(file_name, "data_bin/indice.bin") == 0)
        rate = 2;

    for (int i=0; i<layer_num; i++){
        LOG(INFO) << "layer_idx=" << i << ", " << file_name << endl;
        data[i] = (int **)(malloc(sizeof(int*)*(n_user+n_item)));
        for (int root=0; root<n_user+n_item; root++){
            LOG(INFO) << "root_idx=" << root << "size: "<< data_size[i][root]*rate <<endl;
            data[i][root] = (int*) malloc(sizeof(int)*data_size[i][root]*rate);
            fread(data[i][root], sizeof(int), data_size[i][root]*rate, fd);
        }
    }
    fclose(fd);
    return data;
}

float*** load_data_float(int **data_size, const int layer_num, const char* file_name)
{
    float** data[layer_num];
    LOG(INFO) << "load data float: " << file_name << endl;
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
    return data;
}

int*** load_dense_shape(const int layer_num, const char* file_name)
{
    int** data[layer_num];
    LOG(INFO) << "load dense_shape: " << file_name << endl;
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
    return data;
}

std::string format(std::string type_mode, int layer_num)
{
    std::ostringstream stringStream;
    stringStream << "layer" << layer_num << "_" << type_mode;
    std::string copyOfStr = stringStream.str();
    return copyOfStr;
}

using namespace tensorflow;
REGISTER_OP("TreeOut")
  .Input("input: int32")
  .Output("layer0_neighboor: int32")
  .Output("layer0_adj_indices: int64")
  .Output("layer0_adj_values: float32")
  .Output("layer0_adj_dense_shape: int64")
  .Output("layer1_neighboor: int32")
  .Output("layer1_adj_indices: int64")
  .Output("layer1_adj_values: float32")
  .Output("layer1_adj_dense_shape: int64")
  .Output("layer2_neighboor: int32")
  .Output("layer2_adj_indices: int64")
  .Output("layer2_adj_values: float32")
  .Output("layer2_adj_dense_shape: int64");

class TreeOutOp : public OpKernel {
 public:
  explicit TreeOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Need + 70000 for pos and negs
    LOG(INFO) << "Start initialize null..." << endl;

    /*
    // load_size
    load_size(neighboor_sizes, layer_num-1, "data_bin/neighboor_size.bin");
    load_size(indice_sizes, layer_num, "data_bin/indice_size.bin");

    // load neighboor, indices, values, dense_shape
    load_data_int(neighboors, neighboor_sizes, layer_num-1, "data_bin/neighboor.bin");
    load_data_int(indices, indice_sizes, layer_num, "data_bin/indice.bin");
    load_data_float(values, indice_sizes, layer_num, "data_bin/value.bin");
    load_dense_shape(dense_shapes, layer_num, "data_bin/dense_shape.bin");
    */
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    //LOG(INFO) << "start compute" << endl;
    // Create an output tensor
        const Tensor* input_tensor;
        OP_REQUIRES_OK(context, context->input("input", &input_tensor));
        auto input_ids = input_tensor->flat<int32>();
        const int N = input_ids.size();

        //LOG(INFO) << "in for: " << root_mode << gpu << endl;

        // neighboor_0
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
          format("neighboor", 0), {N}, &output));
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
                format("neighboor", layer+1), {count}, &output));
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
            format("adj_values", layer), {count}, &output));
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
            format("adj_indices", layer), {count, 2}, &output));
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
            shape1 = shape1 + dat_shape[1];
          }

          output = NULL;
          OP_REQUIRES_OK(context, context->allocate_output(
            format("adj_dense_shape", layer), {2}, &output));
          auto output_flat_shape = output->flat<int64>();

          output_flat_shape(0) = shape0;
          if (layer < layer_num-1)
              output_flat_shape(1) = shape1;
          else
              output_flat_shape(1) = n_user + n_item;
        }
      
    
  }

 //private:
  /*
  static int *neighboor_sizes[layer_num-1];
  static int **neighboors[layer_num-1];

  static int *indice_sizes[layer_num];
  static int **indices[layer_num];
  static float **values[layer_num];
  static int **dense_shapes[layer_num];
  */
  static int** neighboor_sizes;// = load_size(layer_num-1, "data_bin/neighboor_size.bin");
  static int** indice_sizes;// = load_size(layer_num, "data_bin/indice_size.bin");

  static int*** neighboors;// = load_data_int(neighboor_sizes, layer_num-1, "data_bin/neighboor.bin");
  static int*** indices;// = load_data_int(indice_sizes, layer_num, "data_bin/indice.bin");
  static float*** values;// = load_data_float(indice_sizes, layer_num, "data_bin/value.bin");
  static int*** dense_shapes;// = load_data_int(indice_sizes, layer_num, "data_bin/dense_shape.bin");
};

int** TreeOutOp::neighboor_sizes = load_size(layer_num-1, "data_bin/neighboor_size.bin");
int** TreeOutOp::indice_sizes = load_size(layer_num, "data_bin/indice_size.bin");

int*** TreeOutOp::neighboors = load_data_int(TreeOutOp::neighboor_sizes, layer_num-1, "data_bin/neighboor.bin");
int*** TreeOutOp::indices = load_data_int(TreeOutOp::indice_sizes, layer_num, "data_bin/indice.bin");
float*** TreeOutOp::values = load_data_float(TreeOutOp::indice_sizes, layer_num, "data_bin/value.bin");
int*** TreeOutOp::dense_shapes = load_data_int(TreeOutOp::indice_sizes, layer_num, "data_bin/dense_shape.bin");

REGISTER_KERNEL_BUILDER(Name("TreeOut").Device(DEVICE_CPU), TreeOutOp);
