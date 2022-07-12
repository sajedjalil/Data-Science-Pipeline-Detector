
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tifffile as tiff

#------------------------------
# function
#------------------------------

def make_histogram(images_dir, AMP, num_channel, colors):

	fig, axes = plt.subplots(nrows=num_channel, ncols=1,
		sharex=True, sharey=True, figsize=(16, num_channel*2))

	for channel in range(num_channel):
		
		print(images_dir + AMP + ' channel: {}'.format(channel))

		pixels = np.array([])

		for train_image in train_images:
			target_file_name = "{}/{}{}.tif".format(images_dir, train_image, AMP)
			tiff_image = tiff.imread(os.path.join(target_dir, target_file_name))
			pixels = np.append(pixels, tiff_image[channel,:,:].flatten())

		axes[channel].hist(pixels, 50, normed=True, color=colors[channel], alpha=0.8)
		axes[channel].set_ylabel('Frequency')

	plt.xlabel('Pixel Value')

	plt.tight_layout()
	plt.savefig(images_dir + AMP + '.png'.format(channel))
	plt.close()

#------------------------------
# main
#------------------------------

if __name__ == '__main__':

	target_dir = '../input'

	train_wkt = pd.read_csv(os.path.join(target_dir, 'train_wkt_v4.csv'))
	train_images = train_wkt.ImageId.unique().tolist()

	colors_eight = cm.rainbow(np.linspace(0, 1, 8))
	colors_rgb = ['r', 'g', 'b']

	make_histogram('sixteen_band', '_A', 8, colors_eight)
	make_histogram('sixteen_band', '_M', 8, colors_eight)
	make_histogram('three_band', '', 3, colors_rgb)
