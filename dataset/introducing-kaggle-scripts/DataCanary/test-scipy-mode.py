# https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.mode.html
import numpy as np

a = np.array([[6, 8, 3, 0],
               [3, 2, 1, 7],
               [8, 1, 8, 4],
               [5, 3, 0, 5],
               [4, 7, 5, 9]])
from scipy import stats
print(stats.mode(a))
