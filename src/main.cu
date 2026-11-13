#include "../include/layer.cuh"
#include "loader.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

Layer input = Layer(0, 0, 28 * 28);
Layer l_c1 = Layer(5 * 5, 6, 6 * 24 * 24);
Layer l_s = Layer(4 * 4, 6, 6 * 6 * 6);
Layer l_f = Layer(216, 10, 10);

Data data;

void learn();
void test();

void loaddata(char *train_data_path, char *train_label_path,
              char *test_data_path, char *test_label_path) {
  data = data_gen(train_data_path, train_label_path, test_data_path,
                  test_label_path);
}

int main(int argc, char **argv) {
  srand(time(NULL));
  if (argc != 5) {
    printf("Path missing\n");
    exit(1);
  }
  loaddata(argv[1], argv[2], argv[3], argv[4]);
  learn();
  test();

  return 0;
}

void learn() {}
