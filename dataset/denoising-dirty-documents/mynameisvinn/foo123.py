import os
os.system("ls ../input")

os.system('''echo "\n\nLet's look inside the directories:"''')
os.system("ls ../input/*")

# Copy a couple images to working so they are displayed as output:

os.system("cp ../input/train/101.png train_101.png")
os.system("cp ../input/train_cleaned/101.png train_cleaned_101.png")

"""
Simple high pass filter denoising benchmark. Scores 0.09568

__author__ : Nicholas Guttenberg
"""

import numpy as np
import os
from PIL import Image
import gzip

submission = gzip.open("fourierSubmission.csv.gz","wt")
submission.write("id,value\n")

for f in os.listdir("../input/test/"):
	imgid = int(f[:-4])
	
	imdata = np.asarray(Image.open("../input/test/"+f).convert('L'))/255.0
	
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
			if (kx*kx + ky*ky < 0.030*0.030):
				imfft[i,j] = 0
	
	# Transform back
	newimage = 1.0*((np.fft.ifft2(imfft)).real)+1.0
	
	newimage = np.minimum(newimage, 1.0)
	newimage = np.maximum(newimage, 0.0)
	
	# Write to the submission file
	for j in range(newimage.shape[1]):
		for i in range(newimage.shape[0]):
			submission.write("{}_{}_{},{}\n".format(imgid,i+1,j+1,newimage[i,j]))

submission.close()