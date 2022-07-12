#import pylab
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_adaptive

def mse(x, y):
    return np.linalg.norm(x - y)
label = 'MSE: %2.f'
image_id = 101
dirty_image_path = "../input/train/%d.png" % image_id
clean_image_path = "../input/train_cleaned/%d.png" % image_id
dirty = Image.open(dirty_image_path)
clean = Image.open(clean_image_path)
#clean = Image.open(clean_image_path)
image = np.asarray(dirty)
clean_array = np.asarray(clean)

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 40
binary_adaptive = threshold_adaptive(image, block_size, offset=20)
#Re-dpo a global threshold
binary_adaptive_2 = binary_adaptive > threshold_otsu(binary_adaptive)  

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 8))
(ax0, ax1), (ax2, ax3) = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image MSE= %2.3f' % mse(clean_array,image) )

ax1.imshow(binary_global)
ax1.set_title('Global thresholding MSE= %2.3f' % mse(clean_array,binary_global) )

#ax2.imshow(binary_adaptive)
#ax2.set_title('Adaptive thresholding')

ax2.imshow(binary_adaptive_2)
ax2.set_title('Adaptive/Global thr. MSE= %2.3f' % mse(clean_array,binary_adaptive_2) )

ax3.imshow(clean_array)
ax3.set_title('Clean  MSE= %2.3f' % mse(clean_array,clean_array) )

#for ax in axes:
#    ax.axis('off')

#plt.show()
plt.savefig('skimage.png')