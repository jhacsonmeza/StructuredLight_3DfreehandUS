import re
import numpy as np

def natsort(l):
    '''
    Lambda function for nautural sorting of strings. Useful for sorting the 
    list of file name of images with the target. Taken from:
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    
    input:
        l: list of input images with the target
    output:
        Nutural sorted list of images
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(l, key=alphanum_key)

def writePointCloud(X, C, file_name):
    n = X.shape[1]
    X = np.concatenate([X.T.astype(np.float32)] + 3*[C[:,None]], 1)

    template = "%.4f %.4f %.4f %d %d %d\n"
    template = n*template
    data = template % tuple(X.ravel())
    
    with open(file_name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        f.write(data)