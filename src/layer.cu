#include "../include/layer.cuh"

// Learning rate used by the backward bias updates
static const float dt = 1.0E-01f;

// Layer structure functions
Layer::Layer(int in_size, int on_size, int o_size) {
    this->in_size = in_size;
    this->on_size = on_size;
    this->o_size = o_size;
    
    // host side staged biasses that is copied to layer on device memory
    float s_bias[in_size];
    float s_weight[in_size][on_size];

    output = NULL;
    pre_values = NULL;
    bias = NULL;
    weight = NULL;
    
    // randomly generate bias and weight
    for(int i = 0; i < in_size; i++) {
        s_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

        for(int j = 0; j < on_size; j++)
            s_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
    }

    // allocate device memory for layer variables
    cudaMalloc(&this->output, sizeof(float) * on_size);
    cudaMalloc(&this->pre_values, sizeof(float) * on_size);
    cudaMalloc(&this->bias, sizeof(float) * in_size);
    cudaMalloc(&this->weight, sizeof(float) * in_size * on_size);
    cudaMalloc(&this->g_output, sizeof(float) * on_size);
    cudaMalloc(&this->g_pre_values, sizeof(float) * on_size);
    cudaMalloc(&this->g_weight, sizeof(float) * in_size * on_size);

    // copy bias and weight
    cudaMemcpy(bias, s_bias, sizeof(float) * in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, s_weight, sizeof(float) * in_size * on_size, cudaMemcpyHostToDevice);
}

Layer::~Layer() {
    // free up device memory appropriately
    cudaFree(output);
    cudaFree(pre_values);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(g_output);
    cudaFree(g_pre_values);
    cudaFree(g_weight);
}

void Layer::SetOutput(float *data) {
    // copy data as output for layer
    cudaMemcpy(output, data, sizeof(float) * o_size, cudaMemcpyHostToDevice);
}

void Layer::ForwardClear() {
    // set each byte of forward array structures to zero
    cudaMemset(output, 0, sizeof(float) * o_size);
    cudaMemset(pre_values, 0, sizeof(float) * o_size);
}

void Layer::BackwardClear() {
    // set each byte of backward array structures to zero
    cudaMemset(g_output, 0, sizeof(float) * o_size);
    cudaMemset(g_pre_values, 0, sizeof(float) * o_size);
    cudaMemset(g_weight, 0, sizeof(float) * in_size * on_size);
}

// Utilities
__device__ float Activate(float value) { return 1 / (1 + exp(-value)); }

__global__ void ApplyActivate(float *input, float *output, int N) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  // Activate
  for (int index = N * position / size; index < N * (position + 1) / size;
       index++)
    output[index] = Activate(input[index]);
}

__global__ void ComputeError(float *error, float *output, unsigned int Y,
                             const int N) {
  const int position = blockDim.x * blockIdx.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  for (int i = N * position / size; i < N * (position + 1) / size; ++i) {
    error[i] = ((Y == i ? 1.0f : 0.0f) - output[i]);
  }
}

__global__ void ApplyGradient(float *output, float *grad, int N) {
  const int pos = blockDim.x * blockIdx.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  for (int i = N * pos / size; i < N * (pos + 1) / size; ++i) {
    output[i] += dt * grad[i];
  }
}

// CUDA Kernels
// Forward Propogation
__global__ void ForwardCLPreActivate(float input[28][28],
                                     float pre_values[6][24][24],
                                     float weight[6][5][5]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 5 * 5 * 6 * 24 * 24;

  // Convolution
  for (int j = N * position / size; j < N * (position + 1) / size; j++) {
    int index = j;

    int i1 = ((index /= 1) % 5);
    int i2 = ((index /= 5) % 5);
    int i3 = ((index /= 5) % 6);
    int i4 = ((index /= 6) % 24);
    int i5 = ((index /= 24) % 24);

    // Accumulation
    atomicAdd(&pre_values[i3][i4][i5],
              weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
  }
}

__global__ void ForwardCLBias(float pre_values[6][24][24], float bias[6]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 6 * 24 * 24;

  for (int j = N * position / size; j < N * (position + 1) / size; j++) {
    int index = j;

    int i1 = (index % 6);
    int i2 = ((index /= 6) % 24);
    int i3 = ((index /= 24) % 24);

    pre_values[i1][i2][i3] += bias[i1];
  }
}

__global__ void ForwardPLPreActivate(float input[6][24][24],
                                     float pre_values[6][6][6],
                                     float weight[1][4][4]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 4 * 4 * 6 * 6 * 6;

  // Pooling operation
  for (int j = N * position / size; j < N * (position + 1) / size; j++) {
    int index = j;
    int i1 = ((index /= 1) % 4);
    int i2 = ((index /= 4) % 4);
    int i3 = ((index /= 4) % 6);
    int i4 = ((index /= 6) % 6);
    int i5 = ((index /= 6) % 6);

    // Accumulation
    atomicAdd(&pre_values[i3][i4][i5],
              weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
  }
}

__global__ void ForwardPLBias(float pre_values[6][6][6], float bias[1]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 6 * 6 * 6;

  // Add bias
  for (int j = N * position / size; j < N * (position + 1) / size; j++) {
    int index = j;

    int i1 = ((index /= 1) % 6);
    int i2 = ((index /= 6) % 6);
    int i3 = ((index /= 6) % 6);

    pre_values[i1][i2][i3] += bias[0];
  }
}

__global__ void ForwardFCLPreActivate(float input[6][6][6],
                                      float pre_values[10],
                                      float weight[10][6][6][6]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 10 * 6 * 6 * 6;

  // Fully connected operation
  for (int j = N * position / size; j < N * (position + 1) / size; j++) {
    int index = j;
    int i1 = ((index /= 1) % 10);
    int i2 = ((index /= 10) % 6);
    int i3 = ((index /= 6) % 6);
    int i4 = ((index /= 6) % 6);

    // Accumulation
    atomicAdd(&pre_values[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
  }
}

__global__ void ForwardFCLBias(float pre_values[10], float bias[10]) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  int size = blockDim.x * gridDim.x;

  int N = 10;

  // Add bias
  for (int index = N * position / size; index < N * (position + 1) / size;
       index++)
    pre_values[index] += bias[index];
}

// Backward Propogation
// FullyConnected Layer
__global__ void BackwardFCLWeight(float g_weight[10][6][6][6],
                                  float g_pre_values[10],
                                  float p_output[6][6][6]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 10 * 6 * 6 * 6;

  // Compute weight gradients
  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 10);
    const int i2 = ((index /= 10) % 6);
    const int i3 = ((index /= 6) % 6);
    const int i4 = ((index /= 6) % 6);

    g_weight[i1][i2][i3][i4] = g_pre_values[i1] * p_output[i2][i3][i4];
  }
}

__global__ void BackwardFCLBias(float bias[10], float g_pre_values[10]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 10;

  // Update bias with gradients
  for (int index = N * position / size; index < N * (position + 1) / size;
       ++index)
    bias[index] += dt * g_pre_values[index];
}

// Pooling Layer
__global__ void BackwardPLOutput(float g_output[6][6][6],
                                 float n_weight[10][6][6][6],
                                 float ng_pre_values[10]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 10 * 6 * 6 * 6;

  // Propagate error backward from the next (FC) layer
  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 10);
    const int i2 = ((index /= 10) % 6);
    const int i3 = ((index /= 6) % 6);
    const int i4 = ((index /= 6) % 6);

    atomicAdd(&g_output[i2][i3][i4],
              n_weight[i1][i2][i3][i4] * ng_pre_values[i1]);
  }
}

__global__ void BackwardPLPreActivate(float g_pre_values[6][6][6],
                                      float g_output[6][6][6],
                                      float pre_values[6][6][6]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 6 * 6 * 6;

  // Gradient of pre-activation
  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 6);
    const int i2 = ((index /= 6) % 6);
    const int i3 = ((index /= 6) % 6);

    const float o = Activate(pre_values[i1][i2][i3]);

    g_pre_values[i1][i2][i3] = g_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void BackwardPLWeight(float g_weight[1][4][4],
                                 float g_pre_values[6][6][6],
                                 float p_output[6][24][24]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 1 * 4 * 4 * 6 * 6 * 6;
  const float d = powf(6.0f, 3.0f); // Normalization factor

  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 1);
    const int i2 = ((index /= 1) % 4);
    const int i3 = ((index /= 4) % 4);
    const int i4 = ((index /= 4) % 6);
    const int i5 = ((index /= 6) % 6);
    const int i6 = ((index /= 6) % 6);

    atomicAdd(&g_weight[i1][i2][i3],
              g_pre_values[i4][i5][i6] *
                  p_output[i4][i5 * 4 + i2][i6 * 4 + i3] / d);
  }
}

__global__ void BackwardPLBias(float bias[1], float g_pre_values[6][6][6]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 6 * 6 * 6;
  const float d = powf(6.0f, 3.0f); // Normalization factor

  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 6);
    const int i2 = ((index /= 6) % 6);
    const int i3 = ((index /= 6) % 6);

    atomicAdd(&bias[0], dt * g_pre_values[i1][i2][i3] / d);
  }
}

// Convolution Layer
__global__ void BackwardCLOutput(float g_output[6][24][24],
                                 float n_weight[1][4][4],
                                 float ng_pre_values[6][6][6]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 1 * 4 * 4 * 6 * 6 * 6;

  // Propagate error backward from the next (pooling) layer
  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 1);
    const int i2 = ((index /= 1) % 4);
    const int i3 = ((index /= 4) % 4);
    const int i4 = ((index /= 4) % 6);
    const int i5 = ((index /= 6) % 6);
    const int i6 = ((index /= 6) % 6);

    atomicAdd(&g_output[i4][i5 * 4 + i2][i6 * 4 + i3],
              n_weight[i1][i2][i3] * ng_pre_values[i4][i5][i6]);
  }
}

__global__ void BackwardCLPreActivate(float g_pre_values[6][24][24],
                                      float g_output[6][24][24],
                                      float pre_values[6][24][24]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 6 * 24 * 24;

  // Gradient of pre-activation (sigmoid derivative)
  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 6);
    const int i2 = ((index /= 6) % 24);
    const int i3 = ((index /= 24) % 24);

    const float o = Activate(pre_values[i1][i2][i3]);

    g_pre_values[i1][i2][i3] = g_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void BackwardCLWeight(float g_weight[6][5][5],
                                 float g_pre_values[6][24][24],
                                 float p_output[28][28]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 6 * 5 * 5 * 24 * 24;
  const float d = powf(24.0f, 2.0f); // Normalization factor

  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 6);
    const int i2 = ((index /= 6) % 5);
    const int i3 = ((index /= 5) % 5);
    const int i4 = ((index /= 5) % 24);
    const int i5 = ((index /= 24) % 24);

    atomicAdd(&g_weight[i1][i2][i3],
              g_pre_values[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
  }
}

__global__ void BackwardCLBias(float bias[6], float g_pre_values[6][24][24]) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  const int N = 6 * 24 * 24;
  const float d = powf(24.0f, 2.0f); // Normalization factor

  for (int n = N * position / size; n < N * (position + 1) / size; ++n) {
    int index = n;
    const int i1 = ((index /= 1) % 6);
    const int i2 = ((index /= 6) % 24);
    const int i3 = ((index /= 24) % 24);

    atomicAdd(&bias[i1], dt * g_pre_values[i1][i2][i3] / d);
  }
}
