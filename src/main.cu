#include "../include/layer.cuh"
#include "../include/loader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define N 64

Layer il = Layer(0, 0, 28 * 28); // input layer
Layer cl = Layer(5 * 5, 6, 6 * 24 * 24); // convolutional layer
Layer pl = Layer(4 * 4, 6, 6 * 6 * 6); // pooling layer
Layer fcl = Layer(216, 10, 10); // fully connected layer

Data data; // variable into which the data set is loaded

static void ForwardPass(Image* image) {
  // clear layers
  il.ForwardClear();
  cl.ForwardClear();
  pl.ForwardClear();
  fcl.ForwardClear();
  
  il.SetOutput((float*) image->data);

  // Forward Propogation

  // Convolution Layer
  ForwardCLPreActivate<<<N, N>>>((float (*)[28]) il.output, (float (*)[24][24])cl.pre_values, (float (*)[5][5])cl.weight);
  ForwardCLBias<<<N, N>>>((float (*)[24][24]) cl.pre_values, (float*) cl.bias);
  ApplyActivate<<<N, N>>>(cl.pre_values, cl.output, cl.o_size);
  
  // Pooling Layer
  ForwardPLPreActivate<<<N, N>>>((float (*)[24][24]) cl.output, (float (*)[6][6]) pl.pre_values, (float (*)[4][4]) pl.weight);
  ForwardPLBias<<<N, N>>>((float (*)[6][6]) pl.pre_values, (float*) pl.bias);
  ApplyActivate<<<N, N>>>(pl.pre_values, pl.output, pl.o_size);

  // Fully Connected Layer
  ForwardFCLPreActivate<<<N, N>>>((float (*)[6][6]) pl.output, fcl.pre_values, (float (*)[6][6][6]) fcl.weight);
  ForwardFCLBias<<<N, N>>>((float*) fcl.pre_values, (float*) fcl.bias);
  ApplyActivate<<<N, N>>>(fcl.pre_values, fcl.output, fcl.o_size);
}

static void BackwardPass() {

}

static unsigned int Decide() {

}

static void Learn() {

}

static void Test() {

}

// load data from the root file path of the dataset
void LoadData(const char* root) {
  char train_data_path[256];
  char train_label_path[256];
  char test_data_path[256];
  char test_label_path[256];

  snprintf(train_data_path, sizeof(train_data_path), "%s/MNIST/raw/train-images-idx3-ubyte", root);
  snprintf(train_label_path, sizeof(train_label_path), "%s/MNIST/raw/train-labels-idx1-ubyte", root);
  snprintf(test_data_path, sizeof(test_data_path), "%s/MNIST/raw/t10k-images-idx3-ubyte", root);
  snprintf(test_label_path, sizeof(test_label_path), "%s/MNIST/raw/t10k-labels-idx1-ubyte", root);

  data = GenData(train_data_path, train_label_path, test_data_path,
                  test_label_path);
}

int main(int argc, char **argv) {

  if(argc != 2) {
    printf("Usage: %s <dataset-root>", argv[0]);
    return 1;
  }
  
  LoadData(argv[1]);
std::cout << "[LOG] Data set is loaded successfully!" << std::endl;

  Learn();
  std::cout << "[LOG] Learning process is completed\n" << std::endl;

  Test();
  std::cout << "[LOG] Testing Process is completed\n" << std::endl;

  return 0;
}
