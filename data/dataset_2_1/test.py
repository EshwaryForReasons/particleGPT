import numpy as np

arr1 = np.fromfile('outputs/train.bin', dtype=np.uint16)
print(arr1)

for x in arr1:
    print(x)