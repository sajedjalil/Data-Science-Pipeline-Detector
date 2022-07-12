__author__ = 'Prasetia-Utama'
from scipy import misc
import numpy as np
import glob
from scipy import ndimage
image = misc.imread('../input/train/2.png')/255.0
x, y = image.shape
kernel = np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]])
bg = ndimage.gaussian_filter(image, sigma=6)
bg = ndimage.convolve(bg,kernel)
mask = image < bg - 0.149
result = np.where(mask, image, 1.0)
misc.imsave('background.png', bg)
misc.imsave('mask.png', mask)
misc.imsave('result.png', result)

