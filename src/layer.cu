#include "../include/layer.cuh"

// Learning rate used by the backward bias updates
static const float dt = 1.0E-01f;

// Layer structure functions
Layer::Layer(int in_size, int on_size, int o_size) {
  this->in_size = in_size;
  this->on_size = on_size;
  this->o_size = o_size;

  // host side staged biasses that is copied to layer on device memory
  float s_bias[on_size];
  float s_weight[on_size][in_size];

  output = NULL;
  pre_values = NULL;
  bias = NULL;
  weight = NULL;

  // randomly generate bias and weight
  for (int i = 0; i < on_size; i++) {
    s_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

    for (int j = 0; j < in_size; j++)
      s_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
  }

  // allocate device memory for layer variables
  cudaMalloc(&this->output, sizeof(float) * o_size);
  cudaMalloc(&this->pre_values, sizeof(float) * o_size);
  cudaMalloc(&this->bias, sizeof(float) * on_size);
  cudaMalloc(&this->weight, sizeof(float) * on_size * in_size);
  cudaMalloc(&this->g_output, sizeof(float) * o_size);
  cudaMalloc(&this->g_pre_values, sizeof(float) * o_size);
  cudaMalloc(&this->g_weight, sizeof(float) * on_size * in_size);

  // copy bias and weight
  cudaMemcpy(bias, s_bias, sizeof(float) * on_size, cudaMemcpyHostToDevice);
  cudaMemcpy(weight, s_weight, sizeof(float) * on_size * in_size,
             cudaMemcpyHostToDevice);
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
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // split work for threads by stride
  for (int i = thread_id; i < N; i += stride)
    output[i] = Activate(input[i]);
}

__global__ void ComputeError(float *error, float *output, unsigned int Y,
                             const int N) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // split work for threads by stride
  for (int i = thread_id; i < N; i += stride) {
    float target = (Y == (unsigned int)i) ? 1.0f : 0.0f;
    error[i] = target - output[i];
  }
}

__global__ void ApplyGradient(float *output, float *grad, int N) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // split work for threads by stride
  for (int i = thread_id; i < N; i += stride)
    output[i] += dt * grad[i];
}

// CUDA Kernels
// Forward Propogation
__global__ void ForwardCLPreActivate(float input[28][28],
                                     float pre_values[6][24][24],
                                     float weight[6][5][5]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 24 * 24;

  // distribute work for threads between channels, at height and width
  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (24 * 24);
    const int height = (i / 24) % 24;
    const int width = i % 24;

    float result = 0.0f;

    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 5; k++)
        result += weight[channel][j][k] * input[height + j][width + k];
    }

    pre_values[channel][height][width] = result;
  }
}

__global__ void ForwardCLBias(float pre_values[6][24][24], float bias[6]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 24 * 24;

  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (24 * 24);
    const int height = (i / 24) % 24;
    const int width = i % 24;

    pre_values[channel][height][width] += bias[channel];
  }
}

__global__ void ForwardPLPreActivate(float input[6][24][24],
                                     float pre_values[6][6][6],
                                     float weight[1][4][4]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 6 * 6;

  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (6 * 6);
    const int height = (i / 6) % 6;
    const int width = i % 6;

    float result = 0.0f;

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++)
        result +=
            weight[0][j][k] * input[channel][height * 4 + j][width * 4 + k];
    }

    pre_values[channel][height][width] = result;
  }
}

__global__ void ForwardPLBias(float pre_values[6][6][6], float bias[1]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 6 * 6;

  // split work among threads with strided offset
  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (6 * 6);
    const int height = (i / 6) % 6;
    const int width = i % 6;

    // add shared bias to pre activation value
    pre_values[channel][height][width] += bias[0];
  }
}

__global__ void ForwardFCLPreActivate(float input[6][6][6],
                                      float pre_values[10],
                                      float weight[10][6][6][6]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = thread_id; i < 10; i += stride) {
    float result = 0.0f;

    for (int channel = 0; channel < 6; channel++) {
      for (int height = 0; height < 6; height++) {
        for (int width = 0; width < 6; width++)
          result +=
              weight[i][channel][height][width] * input[channel][height][width];
      }
    }

    pre_values[i] = result;
  }
}

__global__ void ForwardFCLBias(float pre_values[10], float bias[10]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = thread_id; i < 10; i += stride)
    pre_values[i] += bias[i];
}

// Backward Propogation
// FullyConnected Layer
__global__ void BackwardFCLWeight(float g_weight[10][6][6][6],
                                  float g_pre_values[10],
                                  float p_output[6][6][6]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 10 * 6 * 6 * 6;

  // split work by stride and indexing by converting iterator to 4D co-ords
  for (int i = thread_id; i < total; i += stride) {
    const int idx_1 = i / (6 * 6 * 6);
    const int rem_1 = i % (6 * 6 * 6);

    const int idx_2 = rem_1 / (6 * 6);
    const int rem_2 = rem_1 % (6 * 6);

    const int idx_3 = rem_2 / 6;
    const int idx_4 = rem_2 % 6;

    g_weight[idx_1][idx_2][idx_3][idx_4] =
        g_pre_values[idx_1] * p_output[idx_2][idx_3][idx_4];
  }
}

__global__ void BackwardFCLBias(float bias[10], float g_pre_values[10]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = thread_id; i < 10; i += stride)
    bias[i] += dt * g_pre_values[i];
}

// Pooling Layer
__global__ void BackwardPLOutput(float g_output[6][6][6],
                                 float n_weight[10][6][6][6],
                                 float ng_pre_values[10]) {

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 6 * 6;

  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (6 * 6);
    const int height = (i / 6) % 6;
    const int width = i % 6;

    float result = 0.0f;

    for (int j = 0; j < 10; j++)
      result += n_weight[j][channel][height][width] * ng_pre_values[j];

    g_output[channel][height][width] = result;
  }
}

__global__ void BackwardPLPreActivate(float g_pre_values[6][6][6],
                                      float g_output[6][6][6],
                                      float pre_values[6][6][6]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 6 * 6;

  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (6 * 6);
    const int height = (i / 6) % 6;
    const int width = i % 6;

    float out = Activate(pre_values[channel][height][width]);
    g_pre_values[channel][height][width] =
        g_output[channel][height][width] * out * (1.0f - out);
  }
}

__global__ void BackwardPLWeight(float g_weight[1][4][4],
                                 float g_pre_values[6][6][6],
                                 float p_output[6][24][24]) {

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 4 * 4;

  // normalization factor
  const float nf = 216.0f;

  for (int i = thread_id; i < total; i += stride) {
    int j = i / 4;
    int k = i % 4;

    float result = 0.0f;
    for (int channel = 0; channel < 6; channel++) {
      for (int height = 0; height < 6; height++) {
        for (int width = 0; width < 6; width++)
          result += g_pre_values[channel][height][width] *
                    p_output[channel][height * 4 + j][width * 4 + k];
      }
    }
    g_weight[0][j][k] = result / nf;
  }
}

__global__ void BackwardPLBias(float bias[1], float g_pre_values[6][6][6]) {

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 6 * 6;

  // normaliation factor
  const float nf = 216.0f;

  float result = 0.0f;

  for (int i = thread_id; i < total; i += stride) {
    const int channel = i / (6 * 6);
    const int height = (i / 6) % 6;
    const int width = i % 6;

    result += g_pre_values[channel][height][width];
  }

  atomicAdd(&bias[0], dt * result / nf);
}

// Convolution Layer
__global__ void BackwardCLOutput(float g_output[6][24][24],
                                 float n_weight[1][4][4],
                                 float ng_pre_values[6][6][6]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 24 * 24;

  for (int i = thread_id; i < total; i += stride) {
    int channel = i / (24 * 24);
    int height = (i / 24) % 24;
    int width = i % 24;

    float result = 0.0f;

    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        int x = height - j * 4;
        int y = width - k * 4;

        if (x >= 0 && x < 4 && y >= 0 && y < 4)
          result += n_weight[0][x][y] * ng_pre_values[channel][j][k];
      }
    }

    g_output[channel][height][width] = result;
  }
}

__global__ void BackwardCLPreActivate(float g_pre_values[6][24][24],
                                      float g_output[6][24][24],
                                      float pre_values[6][24][24]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total = 6 * 24 * 24;

  for (int i = thread_id; i < total; i += stride) {
    int channel = i / (24 * 24);
    int height = (i / 24) % 24;
    int width = i % 24;

    float out = Activate(pre_values[channel][height][width]);
    g_pre_values[channel][height][width] =
        g_output[channel][height][width] * out * (1.0f - out);
  }
}

__global__ void BackwardCLWeight(float g_weight[6][5][5],
                                 float g_pre_values[6][24][24],
                                 float p_output[28][28]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // total work and normalization factor
  const int total = 6 * 5 * 5;
  const float nf = 576.0f;

  for (int i = thread_id; i < total; i += stride) {
    int channel = i / (5 * 5);
    int rem = i % (5 * 5);

    int j = rem / 5;
    int k = rem % 5;

    float result = 0.0f;

    for (int x = 0; x < 24; x++) {
      for (int y = 0; y < 24; y++)
        result += g_pre_values[channel][x][y] * p_output[x + j][y + k];
    }

    g_weight[channel][j][k] = result / nf;
  }
}

__global__ void BackwardCLBias(float bias[6], float g_pre_values[6][24][24]) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // normalization factor
  const float nf = 576.0f;

  for (int channel = thread_id; channel < 6; channel += stride) {
    float result = 0.0f;
    for (int height = 0; height < 24; height++) {
      for (int width = 0; width < 24; width++)
        result += g_pre_values[channel][height][width];
    }

    bias[channel] += dt * result / nf;
  }
}
