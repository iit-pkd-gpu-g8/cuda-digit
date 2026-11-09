#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float data[28][28];
  int label;
} Image;

typedef struct {
  Image *train_images;
  Image *test_images;
} Data;

Data data_gen(char *train_data_path, char *train_label_path,
              char *test_data_path, char *test_label_path) {

  Data data;

  FILE *train_img = fopen(train_data_path, "rb");

  if (!train_img) {
    printf("Error: could not open training images\n");
    exit(1);
  }

  fseek(train_img, 16, SEEK_SET);

  Image *train_img_set = malloc(60000 * sizeof(Image));

  unsigned char buffer[28 * 28];

  for (int i = 0; i < 60000; i++) {
    fread(buffer, 1, 784, train_img);

    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        train_img_set[i].data[j][k] = (float)buffer[j * 28 + k] / 255.0f;
      }
    }
  }

  fclose(train_img);

  FILE *train_labels = fopen(train_label_path, "rb");

  if (!train_labels) {
    printf("Error: could not open training labels\n");
    free(train_img_set);
    exit(1);
  }

  fseek(train_labels, 8, SEEK_SET);

  unsigned char *buffer1 = malloc(60000 * sizeof(unsigned char));
  fread(buffer1, 1, 60000, train_labels);

  for (int i = 0; i < 60000; i++) {
    train_img_set[i].label = (int)buffer1[i];
  }

  fclose(train_labels);
  free(buffer1);

  FILE *test_img = fopen(test_data_path, "rb");

  if (!test_img) {
    printf("Error: could not open test images\n");
    free(train_img_set);
    exit(1);
  }

  fseek(test_img, 16, SEEK_SET);

  Image *test_img_set = malloc(10000 * sizeof(Image));

  unsigned char buffer2[28 * 28];

  for (int i = 0; i < 10000; i++) {
    fread(buffer2, 1, 784, test_img);

    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        test_img_set[i].data[j][k] = (float)buffer2[j * 28 + k] / 255.0f;
      }
    }
  }

  fclose(test_img);

  FILE *test_labels = fopen(test_label_path, "rb");

  if (!test_labels) {
    printf("Error: could not open test labels\n");
    free(train_img_set);
    free(test_img_set);
    exit(1);
  }

  fseek(test_labels, 8, SEEK_SET);

  unsigned char *buffer3 = malloc(10000 * sizeof(unsigned char));
  fread(buffer3, 1, 10000, test_labels);

  for (int i = 0; i < 10000; i++) {
    test_img_set[i].label = (int)buffer3[i];
  }

  fclose(test_labels);
  free(buffer3);

  data.train_images = train_img_set;
  data.test_images = test_img_set;

  return data;
}
