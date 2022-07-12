# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

from skimage import measure
import scipy.ndimage as ndi

from pylab import rcParams
rcParams['figure.figsize'] = (6, 6) 

touse = mpimg.imread('../input/images/53.jpg')

cy, cx = ndi.center_of_mass(touse)

plt.imshow(touse, cmap='Set3')
plt.scatter(cx, cy)
plt.show()