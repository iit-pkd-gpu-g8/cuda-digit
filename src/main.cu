#include "../include/layer.cuh"
#include "../include/loader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define N 64

Layer il = Layer(0, 0, 28 * 28);         // input layer
Layer cl = Layer(5 * 5, 6, 6 * 24 * 24); // convolutional layer
Layer pl = Layer(4 * 4, 1, 6 * 6 * 6);   // pooling layer
Layer fcl = Layer(6 * 6 * 6, 10, 10);          // fully connected layer

Data data; // variable into which the data set is loaded

static void ForwardPass(Image *image) {
  // clear layers
  il.ForwardClear();
  cl.ForwardClear();
  pl.ForwardClear();
  fcl.ForwardClear();

  il.SetOutput((float *)image->data);

  // Forward Propogation

  // Convolution Layer
  ForwardCLPreActivate<<<N, N>>>((float (*)[28])il.output,
                                 (float (*)[24][24])cl.pre_values,
                                 (float (*)[5][5])cl.weight);
  ForwardCLBias<<<N, N>>>((float (*)[24][24])cl.pre_values, (float *)cl.bias);
  ApplyActivate<<<N, N>>>(cl.pre_values, cl.output, cl.o_size);

  // Pooling Layer
  ForwardPLPreActivate<<<N, N>>>((float (*)[24][24])cl.output,
                                 (float (*)[6][6])pl.pre_values,
                                 (float (*)[4][4])pl.weight);
  ForwardPLBias<<<N, N>>>((float (*)[6][6])pl.pre_values, (float *)pl.bias);
  ApplyActivate<<<N, N>>>(pl.pre_values, pl.output, pl.o_size);

  // Fully Connected Layer
  ForwardFCLPreActivate<<<N, N>>>((float (*)[6][6])pl.output, fcl.pre_values,
                                  (float (*)[6][6][6])fcl.weight);
  ForwardFCLBias<<<N, N>>>((float *)fcl.pre_values, (float *)fcl.bias);
  ApplyActivate<<<N, N>>>(fcl.pre_values, fcl.output, fcl.o_size);
}

static void BackwardPass() {


  // Backward Propogation

  // Fully Connected Layer
  BackwardFCLWeight<<<N, N>>>((float (*)[6][6][6])fcl.g_weight,
                              (float *)fcl.g_pre_values,
                              (float (*)[6][6])pl.output);
  BackwardFCLBias<<<N, N>>>((float *)fcl.bias, (float *)fcl.g_pre_values);

  // Pooling Layer
  BackwardPLOutput<<<N, N>>>((float (*)[6][6])pl.g_output,
                             (float (*)[6][6][6])fcl.weight,
                             (float *)fcl.g_pre_values);
  BackwardPLPreActivate<<<N, N>>>((float (*)[6][6])pl.g_pre_values,
                                  (float (*)[6][6])pl.g_output,
                                  (float (*)[6][6])pl.pre_values);
  BackwardPLWeight<<<N, N>>>((float (*)[4][4])pl.g_weight,
                             (float (*)[6][6])pl.g_pre_values,
                             (float (*)[24][24])cl.output);
  BackwardPLBias<<<N, N>>>((float *)pl.bias, (float (*)[6][6])pl.g_pre_values);

  // Convolutional Layer
  BackwardCLOutput<<<N, N>>>((float (*)[24][24])cl.g_output,
                             (float (*)[4][4])pl.weight,
                             (float (*)[6][6])pl.g_pre_values);
  BackwardCLPreActivate<<<N, N>>>((float (*)[24][24])cl.g_pre_values,
                                  (float (*)[24][24])cl.g_output,
                                  (float (*)[24][24])cl.pre_values);
  BackwardCLWeight<<<N, N>>>((float (*)[5][5])cl.g_weight,
                             (float (*)[24][24])cl.g_pre_values,
                             (float (*)[28])il.output);
  BackwardCLBias<<<N, N>>>((float *)cl.bias,
                           (float (*)[24][24])cl.g_pre_values);

  // Modify Gradients
  ApplyGradient<<<N, N>>>(fcl.weight, fcl.g_weight, fcl.on_size * fcl.in_size);
  ApplyGradient<<<N, N>>>(pl.weight, pl.g_weight, pl.on_size * pl.in_size);
  ApplyGradient<<<N, N>>>(cl.weight, cl.g_weight, cl.on_size * cl.in_size);
}

static unsigned int Decide(Image *image) {
  float prob[10]; // the last layer output is a simple table of probability
  ForwardPass(image);

  // the label with the highest probability is chosen as the decision
  unsigned int label = 0;
  cudaMemcpy(prob, fcl.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

  for (int i = 1; i < 10; i++) {
    if (prob[i] > prob[label])
      label = i;
  }

  return label;
}

static void ClearGradients() {
  il.BackwardClear();
  cl.BackwardClear();
  pl.BackwardClear();
  fcl.BackwardClear();
}

static void Learn() {
  std::clog << "[LOG] Learning Process has begun" << std::endl;

  for (int i = 0; i < data.train_size; i++) {
    Image *image = &data.train_images[i];

    std::clog << "[LOG] Training on label " << image->label << " | ";
    ForwardPass(image);
    std::clog << "Forward Pass Done | ";
    
    // clear gradients before computing error
    ClearGradients();

    ComputeError<<<10, 1>>>(fcl.g_pre_values, fcl.output, image->label, 10);
    std::clog << "Error Computed | ";

    BackwardPass();
    std::clog << "Backward Pass Done" << std::endl;
  }
}

static float Test() {
  std::clog << "[LOG] Testing Process has begun" << std::endl;

  int mispredictions = 0;

  for (int i = 0; i < data.test_size; i++) {
    Image *image = &data.test_images[i];

    int prediction = Decide(image);

    if (prediction != image->label)
      mispredictions++;

    std::clog << "[LOG] Test " << i << ": label = " << image->label
              << "| predicted = " << prediction << std::endl;
  }

  float accuracy =
      100.0f - (((float)(mispredictions) / ((float)(data.test_size))) * 100);
  return accuracy;
}

// load data from the root file path of the dataset
void LoadData(const char *root) {
  char train_data_path[256];
  char train_label_path[256];
  char test_data_path[256];
  char test_label_path[256];

  snprintf(train_data_path, sizeof(train_data_path),
           "%s/MNIST/raw/train-images-idx3-ubyte", root);
  snprintf(train_label_path, sizeof(train_label_path),
           "%s/MNIST/raw/train-labels-idx1-ubyte", root);
  snprintf(test_data_path, sizeof(test_data_path),
           "%s/MNIST/raw/t10k-images-idx3-ubyte", root);
  snprintf(test_label_path, sizeof(test_label_path),
           "%s/MNIST/raw/t10k-labels-idx1-ubyte", root);

  data = GenData(train_data_path, train_label_path, test_data_path,
                 test_label_path);
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <dataset-root>\n", argv[0]);
    return 1;
  }
    
  srand(time(NULL));

  LoadData(argv[1]);
  std::cout << "[LOG] Data set is loaded successfully!" << std::endl;

  Learn();
  std::cout << "[LOG] Learning process is completed\n" << std::endl;

  float accuracy = Test();
  std::cout << "[LOG] Testing Process is completed\n" << std::endl;

  std::cout << "[LOG] Accuracy of the CNN Model = " << accuracy << std::endl;

  return 0;
}
