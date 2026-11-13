#ifndef LOADER_H
#define LOADER_H

typedef struct{
	float data[28][28];
	int label;
} Image;

typedef struct {
	Image *train_images;
	Image *test_images;
    
    unsigned int train_size;
    unsigned int test_size;

    unsigned int row_size;
    unsigned int col_size;
}Data;

Data GenData(char *train_data_path, char *train_label_path, char *test_data_path, char* test_label_path);

#endif
