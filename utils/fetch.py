import sys
import os
import torchvision as tv

def fetch(path):
    """
    Fetches the data set into the path if path is valid,
    and the data set is yet to be downloaded
    """
    # verify the validity of the path
    if not os.path.exists(path):
        print('Path does not exist')
        print('Attempting to create the path')
        
        try:
            os.makedirs(path)
            print(f'Path {path} created successfully')
        except Exception as e:
            print(f'Error creating path: {e}, fallback failed')
            exit()

    print('Fetching Training Dataset')
    # download the training data-set
    train_dataset = tv.datasets.MNIST(root = path,
                                      train = True,
                                      download = True)
    
    print('Fetching Testing Dataset')
    # donwload the testing data-set
    test_dataset = tv.datasets.MNIST(root = path,
                                     train = False,
                                     download = True)

# entry point if the program is interpreted directly
if __name__ == '__main__':
    # get arguements
    args = sys.argv
    # help for incorrect execution
    if len(args) != 2:
        print('Usage: python fetch.py <path-to-download>')
        exit()

    path = args[1]

    # fetch the data-set into path
    fetch(path)

    exit()
