#include<cuda_runtime.h>

#ifndef __LOADER_H_
#define __LOADER_H_


typedef struct{
	float data[28][28];
	int label;
} Image;

typedef struct {
	Image *train_image;
	Image *test_image;
}Data;

Data data_gen(char *train_data_path, char *train_label_path, char *test_data_path, char* test_label_path);

#endif
