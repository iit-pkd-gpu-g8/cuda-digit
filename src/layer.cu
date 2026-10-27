// Utilities
__device__ float Activate(float value) { return 1 / (1 + exp(-value)); }

__device__ void ApplyActivate(float *input, float *output, int N) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  // Activate
  for (int index = N * position / size; index < N * (position + 1) / size;
       index++)
    output[index] = Activate(input[index]);
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
