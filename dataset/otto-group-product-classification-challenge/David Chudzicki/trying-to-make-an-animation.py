import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0,1)

for i in range(10):
    y = i*x
    p = plt.scatter(x, y)
    p.figure.savefig('image%d.png' % i)