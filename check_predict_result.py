# This file is aimed to check two npy files are totally equal or not.

import numpy as np
import os
import sys

def check_npy_equal(npy1, npy2):
    data1 = np.load(npy1)
    data2 = np.load(npy2)
    if data1.shape != data2.shape:
        print(f"Shape of {npy1} is {data1.shape}, and shape of {npy2} is {data2.shape}.")
        return False
    if not np.allclose(data1, data2):
        print(f"{npy1} and {npy2} are not equal.")
        return False
    return True

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_npy_equal.py npy1 npy2")
        sys.exit(1)
    npy1 = sys.argv[1]
    npy2 = sys.argv[2]
    if check_npy_equal(npy1, npy2):
        print(f"{npy1} and {npy2} are equal.")
    else:
        print(f"{npy1} and {npy2} are not equal.")