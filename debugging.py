import numpy as np

data = np.load("output/my_qtable_2Players_c3_QT_0.npy")
print(data.shape, np.mean(data), np.max(data))
