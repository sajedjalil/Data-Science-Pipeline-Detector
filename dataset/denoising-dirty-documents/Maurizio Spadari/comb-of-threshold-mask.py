

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_adaptive
from sklearn.metrics import mean_squared_error
from scipy import signal
from scipy import ndimage
from operator import itemgetter


def mse(x,y):
	return mean_squared_error(x.flat, y.flat)**0.5

def discretize(a,thr=50):
    return np.uint8((a > thr)*1)
    
def cleanLines(image,rowSum,**kwargs):
	rowMask = np.zeros(rowSum.shape)
	win = kwargs.get('win',2)
	thresh = kwargs.get('thresh',0.8)
	myCleaned = image.copy()
	for (idx,val) in enumerate(rowSum):
		if val < thresh:
			idxLow = max(0,idx-win)
			idxUp = min(rowSum.shape[0],idx+win+1)
	#		print "Set idx %d [%d.%d] " % (idx,idxLow,idxUp)

			rowMask[idxLow:idxUp] = 1
	#print "rowMask value %d" % np.sum(rowMask)

	for (idx,val) in enumerate(rowMask):
		if  not val :
			myCleaned[idx,:] = 1
	return myCleaned   
	
def applyMask(image,mask):
	myCleaned = image.copy()
	#print mask.shape
	for row in range(mask.shape[0]):
		for col in range(mask.shape[1]):
			#print row,col
			if mask[row,col]:
				myCleaned[row,col] = 1
	return myCleaned
label = 'MSE: %2.f'
image_id = 101

dirty_image_path = "../input/train/%d.png" % image_id
clean_image_path = "../input/train_cleaned/%d.png" % image_id
dirty = Image.open(dirty_image_path)
#clean = Image.open(clean_image_path)
clean = Image.open(clean_image_path)
image = np.asarray(dirty)/255.0
image2 = np.asarray(dirty)
clean = np.asarray(clean)/255.0
thresh = 0.8
rowSum = np.sum(image,axis=1)/float(image.shape[1])
resL =[]

#global_thresh = threshold_otsu(image)
#print global_thresh
resL = []
# Loop to optimize threshold based on MSE
for thresh in np.arange(0.3,0.9,0.01) :
	myCleaned2 = image > thresh
	# Using a gaussian filter to blur image does not really help
	# for sigma in np.arange(0.0,0.8,0.05):
	# 	gaussMask = ndimage.gaussian_filter(myCleaned2, sigma=sigma)
	# 	
	# 	myCleaned4 =applyMask(image,gaussMask)
	# 	mseVal = mse(myCleaned4,clean)
	# 	print "Sigma = %3.2f MSe = %3.4f" % (sigma,mseVal)
	# #myCleaned4 = image*myCleaned2
	myCleaned4 =applyMask(image,myCleaned2)
	mseVal = mse(myCleaned4,clean)
	#print "Thr = %3.2f MSe = %3.4f" % (thresh,mseVal)
	resL.append((myCleaned4,thresh,mseVal))
best = sorted(resL,key=itemgetter(2))
#print best[0]
myCleaned4,thres,measVal = best[0]
print "Best thr = %3.2f MSE = %3.4f" % (thres,measVal)
fig, axes = plt.subplots( ncols=3, figsize=(7, 8))
plt.gray()
ax10, ax11,ax12 = axes
ax10.imshow(image)
ax10.set_title('Dirty mse=%3.4f' % mse(image,clean))
# ax11.plot(rowSum)
# ax11.set_title('Row sum')
# ax12.plot(rowMask)
# ax12.set_title('Row mask')
ax11.imshow(clean)
ax11.set_title('clean mse=%3.4f' % mse(clean,clean))
ax12.imshow(myCleaned4)
ax12.set_title('tgt mse=%3.4f' % (mse(myCleaned4,clean)))
#plt.show()
plt.savefig('skimage.png')