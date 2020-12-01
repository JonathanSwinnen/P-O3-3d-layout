import numpy as np
import os

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "calib_out.npz")

data = np.load(path, allow_pickle=True)
lst = data.files

for item in lst:
    print("\n\t", item, "\n")
    print(data[item])