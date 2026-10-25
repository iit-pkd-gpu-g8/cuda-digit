#include <cuda_runtime.h>

struct Layer {
    int in_size; // no of input neurons
    int on_size; // no of output neurons
    int o_size; // no of outputs
    
    float* output; // layer's output
    float* pre_values; // layer's values before applying activation function

    float* bias; // bias array for the layer
    float* weight; // weight array for the layer

    float* g_output; // gradient for output
    float* g_pre_values; // gradient for pre_values
    float* g_weight; // gradient for weights

    Layer(int in_size, int on_size, int o_size); // constructor
    ~Layer(); // destructor
    
    void SetOutput(float* data); // co-routine that directly sets the output to data
    void ForwardClear(); // clear the data generated during forward propogations
    void BackwardClear(); // clear the data generated during backward propogations
};

// CUDA Kernels
// Forward Propogation 
// Convolution Layer - CL
__global__ void ForwardCLPreActivate(float input[28][28], float pre_values[6][24][24], float weight[6][5][5]);
__global__ void ForwardCLBias(float pre_values[6][24][24], float bias[6]);

// Pooling Layer - PL
__global__ void ForwardPLPreActivate(float input[6][24][24], float pre_values[6][6][6], float weight[1][4][4]);
__global__ void ForwardPLBias(float pre_values[6][6][6], float bias[1]);

// FullyConnected Layer - FCL
__global__ void ForwardFCLPreActivate(float input[6][6][6], float pre_values[10], float weight[10][6][6][6]);
__global__ void ForwardFCLBias(float pre_values[10], float bias[10]);

// Backward Propogation
// FullyConnected Layer - FCL
__global__ void BackwardFCLWeight(float g_weight[10][6][6][6], float g_pre_values[10], float p_output[6][6][6]);
__global__ void BackwardFCLBias(float bias[10], float g_pre_values[10]);

// Pooling Layer - PL
__global__ void BackwardPLOutput(float g_output[6][6][6], float n_weight[10][6][6][6], float ng_pre_values[10]);
__global__ void BackwardPLPreActivate(float g_pre_values[6][6][6], float g_output[6][6][6], float pre_values[6][6][6]);
__global__ void BackwardPLWeight(float g_weight[1][4][4], float g_pre_values[6][6][6], float p_output[6][24][24]);
__global__ void BackwardPLBias(float bias[1], float g_pre_values[6][6][6]);

// Convolution Layer - CL
__global__ void BackwardCLOutput(float g_output[6][24][24], float n_weight[1][4][4], float ng_pre_values[6][6][6]);
__global__ void BackwardCLPreActivate(float g_pre_values[6][24][24], float g_output[6][24][24], float pre_values[6][24][24]);
__global__ void BackwardCLWeight(float g_weight[6][5][5], float g_pre_values[6][24][24], float p_output[28][28]);
__global__ void BackwardCLBias(float bias[6], float g_pre_values[6][24][24]);

// Utilities
__device__ float Activate(float value); // sigmoid activation function
__global__ void ApplyActivate(float* input, float* output, int size);
__global__ void ComputeError(float* error, float* output, unsigned int Y, const int N); 
__global__ void ApplyGradient(float* output, float* grad, int N);
