# Cuda-Digit
Cuda-Digit is a simple implementation of a convolution neural network (CNN) written in CUDA / C++ 
to process and classify numerical digit images from the MNIST dataset which is generally used for image-processing benchmarks

## Build
```bash 
./build.sh
```
This script generates the binary for the CNN

Read through `utils/README.md` for downloading the MNIST dataset

## Usage
```bash
cuda-digit <dataset-root-directory> (optional) <train_intensity>
```
* `<dataset-root-directory>` determines the directory to which the fetch utility script downloaded the dataset
* `<train-intensity>`  determines the number of rounds the CNN is trained (default 10)

## Formatting
* code-formatting is done using `clang-format`

## Development
* To see development of the project look at pull-requests which are closed/open/draft
* Commits are squashed and merged into the main branch for keeping the version control system simple
