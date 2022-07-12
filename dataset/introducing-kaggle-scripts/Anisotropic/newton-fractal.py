# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def newton_fractal(xmin, xmax, ymin, ymax, xres, yres):
    yarr, xarr = np.meshgrid(np.linspace(xmin, xmax, xres), np.linspace(ymin, ymax, yres) * 1j)
    arr = yarr + xarr
    ydim, xdim = arr.shape
    arr = arr.flatten()
    f = np.poly1d([1,0,0,-1]) # x^3 - 1
    fp = np.polyder(f)
    
    counts = np.zeros(shape=arr.shape)
    unconverged = np.ones(shape=arr.shape, dtype=bool)
    indices = np.arange(len(arr))
    
    for i in iter(int,1):               # count() iterates infinitely
        f_g = f(arr[unconverged])
        new_unconverged = np.abs(f_g) > 0.00001
        counts[indices[unconverged][~new_unconverged]] = i
        if not np.any(new_unconverged):
            return counts.reshape((ydim, xdim))
        unconverged[unconverged] = new_unconverged
        arr[unconverged] -= f_g[new_unconverged] / fp(arr[unconverged])

N = 1000
pic = newton_fractal(-5, 5, -5, 5, N, N)

ls = LightSource(azdeg = 0, altdeg = 70)
rgb = ls.shade(pic, plt.cm.hsv)
shaded = ls.shade(pic, plt.cm.prism)
picture = plt.imshow(shaded)
picture.set_cmap('flag')
plt.title('Newton Fractal')
plt.show()
