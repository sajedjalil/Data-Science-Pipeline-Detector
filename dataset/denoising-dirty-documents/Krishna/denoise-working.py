"""
Simple high pass filter denoising benchmark. Scores 0.09568

__author__ : Nicholas Guttenberg
"""

import numpy as np
import os
from PIL import Image
import gzip
from scipy import ndimage

submission = gzip.open("fourierSubmission.csv.gz","wt")
submission.write("id,value\n")

for f in os.listdir("../input/test/"):
	imgid = int(f[:-4])
	
	imdata = np.asarray(Image.open("../input/test/"+f).convert('L'))/255.0
	imdata = ndimage.median_filter(imdata, 3);
	
	# Fourier transform the input image
	imfft = np.fft.fft2(imdata)
	
	# Apply a high pass filter to the image. 
	# Note that since we're discarding the k=0 point, we'll have to add something back in later to match the correct white value for
	# the target images
	
	for i in range(imfft.shape[0]):
        # Fourier transformed coordinates in the array start at kx=0 and increase to pi, then flip around to -pi and increase towards 0
		kx = i/float(imfft.shape[0])
		if kx>0.5: 
			kx = kx-1
			
		for j in range(imfft.shape[1]):
			ky = j/float(imfft.shape[1])
			if ky>0.5: 
				ky = ky-1
				
			# Get rid of all the low frequency stuff - in this case, features whose wavelength is larger than about 20 pixels
			if (kx*kx + ky*ky < 0.01*0.01):
				imfft[i,j] = 0
	
	# Transform back
	newimage = 1.0*((np.fft.ifft2(imfft)).real)+0.9
	
	newimage = np.minimum(newimage, 1.0)
	newimage = np.maximum(newimage, 0.0)
	
	# Write to the submission file
	for j in range(newimage.shape[1]):
		for i in range(newimage.shape[0]):
			submission.write("{}_{}_{},{}\n".format(imgid,i+1,j+1,newimage[i,j]))

submission.close()