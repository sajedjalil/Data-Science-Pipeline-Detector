import os
os.system("ls ../input")

os.system('''echo "\n\nLet's look inside the directories:"''')
import numpy as np
from PIL import Image
import pylab




image_id = 101
dirty_image_path = "../input/train/%d.png" % image_id

dirty = Image.open(dirty_image_path)


dirty_array = np.asarray(dirty)

# create a new figure
pylab.figure()

# show contours with origin upper left corner
pylab.contour(dirty_array, levels=[100], colors='black',origin='image')
pylab.axis('equal')

pylab.savefig('contour.png')