#include "../include/loader.h"
#include <stdio.h>
#include <stdlib.h>

static inline unsigned int SwapEndian(unsigned int value) {
  return ((value >> 24) & 0xff) | ((value >> 8) & 0xff00) |
         ((value << 8) & 0xff0000) | ((value << 24) & 0xff000000);
}

static unsigned int ReadField(FILE *file) {
  unsigned int value;

  if (fread(&value, 1, 4, file) != 4) {
    printf("Error: invalid IDX header\n");
    exit(1);
  }

  return SwapEndian(value);
}

Data GenData(char *train_data_path, char *train_label_path,
             char *test_data_path, char *test_label_path) {

  Data data;

  FILE *train_img = fopen(train_data_path, "rb");

  if (!train_img) {
    printf("Error: could not open training images\n");
    exit(1);
  }

  // fseek(train_img, 16, SEEK_SET); - avoid hardcoding
  unsigned int train_images_magic = ReadField(train_img);
  data.train_size = ReadField(train_img);
  data.row_size = ReadField(train_img);
  data.col_size = ReadField(train_img);

  // verify
  if (train_images_magic != 2051 ||
      data.row_size != 28 && data.col_size != 28) {
    printf("Error: invalid trianing image verify source\n");
    exit(1);
  }

  Image *train_img_set = malloc(data.train_size * sizeof(Image));

  unsigned char buffer[data.row_size * data.col_size];

  for (int i = 0; i < data.train_size; i++) {
    fread(buffer, 1, data.row_size * data.col_size, train_img);

    for (int j = 0; j < data.row_size; j++) {
      for (int k = 0; k < data.col_size; k++) {
        train_img_set[i].data[j][k] =
            (float)buffer[j * data.col_size + k] / 255.0f;
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

  //  fseek(train_labels, 8, SEEK_SET); - avoid hardcoding
  unsigned int train_labels_magic = ReadField(train_labels);
  unsigned int train_label_size = ReadField(train_labels);

  if (train_labels_magic != 2049 || train_label_size != data.train_size) {
    printf("Error: invalid training label header\n");
    exit(1);
  }
  /*
    unsigned char *buffer1 = malloc(60000 * sizeof(unsigned char));
    fread(buffer1, 1, 60000, train_labels);

    for (int i = 0; i < 60000; i++) {
      train_img_set[i].label = (int)buffer1[i];
    }
  */

  for (int i = 0; i < data.train_size; i++) {
    unsigned char label;
    fread(&label, 1, 1, train_labels);
    train_img_set[i].label = label;
  }

  fclose(train_labels);

  FILE *test_img = fopen(test_data_path, "rb");

  if (!test_img) {
    printf("Error: could not open test images\n");
    free(train_img_set);
    exit(1);
  }

  //  fseek(test_img, 16, SEEK_SET); - avoid hardcoding

  unsigned int test_images_magic = ReadField(test_img);
  data.test_size = ReadField(test_img);
  unsigned int test_row_size = ReadField(test_img);
  unsigned int test_col_size = ReadField(test_img);

  if (test_images_magic != 2051 || test_row_size != data.row_size ||
      test_col_size != data.col_size) {
    printf("Error: invalid test image header\n");
    exit(1);
  }

  Image *test_img_set = malloc(data.test_size * sizeof(Image));

  unsigned char buffer2[data.row_size * data.col_size];

  for (int i = 0; i < data.test_size; i++) {
    fread(buffer2, 1, data.row_size * data.col_size, test_img);

    for (int j = 0; j < data.row_size; j++) {
      for (int k = 0; k < data.col_size; k++) {
        test_img_set[i].data[j][k] =
            (float)buffer2[j * data.col_size + k] / 255.0f;
      }
    }
  }

  fclose(test_img);

  FILE *test_labels = fopen(test_label_path, "rb");

  if (!test_labels) {
    printf("Error: could not open test labels\n");
    exit(1);
  }

  unsigned int test_labels_magic = ReadField(test_labels);
  unsigned int test_labels_size = ReadField(test_labels);

  if (test_labels_magic != 2049 || test_labels_size != data.test_size) {
    printf("Error: invalid test label header\n");
    exit(1);
  }

  //   fseek(test_labels, 8, SEEK_SET); - avoid hardcoding

  for (unsigned int i = 0; i < data.test_size; i++) {
    unsigned char label;
    fread(&label, 1, 1, test_labels);
    test_img_set[i].label = label;
  }
  /*
    unsigned char *buffer3 = malloc(10000 * sizeof(unsigned char));
    fread(buffer3, 1, 10000, test_labels);

    for (int i = 0; i < 10000; i++) {
      test_img_set[i].label = (int)buffer3[i];
    }
  */

  fclose(test_labels);

  // point to the heap memory holding data sets
  data.train_images = train_img_set;
  data.test_images = test_img_set;

  return data;
}
