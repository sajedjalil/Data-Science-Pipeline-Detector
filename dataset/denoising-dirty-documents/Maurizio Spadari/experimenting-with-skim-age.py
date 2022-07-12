import pylab
import numpy as np
from PIL import Image
from skimage.feature import canny
from scipy import ndimage

from skimage.filters import sobel
from skimage.morphology import watershed
 
image_id = 101
dirty_image_path = "../input/train/%d.png" % image_id

dirty = Image.open(dirty_image_path)
#clean = Image.open(clean_image_path)
dirty_array = np.asarray(dirty)


elevation_map = sobel(dirty_array)
#clean_array = np.asarray(clean)
markers = np.zeros_like(dirty_array)
markers[dirty_array < 50] = 1
markers[dirty_array > 150] = 2
segmentation = watershed(elevation_map, markers)
#segmentation = ndimage.binary_fill_holes(segmentation - 1)
fig = pylab.figure()
new = fig.add_subplot(1,2,1)
new.set_title("Original")
# show contours with origin upper left corner
#pylab.imshow(elevation_map,cmap=pylab.gray())#
img = pylab.imshow(dirty_array)

new = fig.add_subplot(1,2,2)
new.set_title("Post-process")
# show contours with origin upper left corner
#pylab.imshow(elevation_map,cmap=pylab.gray())#
img = pylab.imshow(segmentation,cmap=pylab.gray())

pylab.axis('equal')

pylab.savefig('skimage.png')