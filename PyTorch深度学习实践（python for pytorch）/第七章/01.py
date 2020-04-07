#numpy矩阵的保存

import numpy as np
a = np.array(2)
np.save("nm.npy",a)
a = np.load("nm.npy")
print(a)
