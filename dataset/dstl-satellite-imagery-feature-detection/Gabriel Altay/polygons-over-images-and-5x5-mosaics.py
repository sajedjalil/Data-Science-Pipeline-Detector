"""
Script for plotting training label polygons over satellite images
and creating 5x5 mosaics from multiple imageIds. 
Makes use of ideas in some of the other great scripts posted already. 

See 
  - https://www.kaggle.com/amanbh/dstl-satellite-imagery-feature-detection/visualize-polygons-and-image-data
  - https://www.kaggle.com/jeffhebert/dstl-satellite-imagery-feature-detection/stitch-a-16-channel-image-together
  
"""

import os
import pandas
import numpy as np 
import scipy.misc
from shapely import wkt
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import cm
from descartes.patch import PolygonPatch 



N_CHANNELS = {'3': 3, 'A': 8, 'M': 8, 'P': 1}


class SatImages:

    def __init__(self, data_dir='../input'):
        self.data_dir = data_dir
        self.train_wkt = TrainWkt(data_dir)
        self.grid_sizes = GridSizes(data_dir)

        self.fnames3b = os.listdir(os.path.join(data_dir, 'three_band'))
        self.fnames16b = os.listdir(os.path.join(data_dir, 'sixteen_band'))
        
    def get_fname(self, image_id, band):
        if band == '3':
            fname = '{}/three_band/{}.tif'.format(
                self.data_dir, image_id)
        elif band in ['A', 'M', 'P']:
            fname = '{}/sixteen_band/{}_{}.tif'.format(
                self.data_dir, image_id, band)
        else:
            raise ValueError('band must be one of ["3", "A", "M", "P"]')
        return fname

    def get_image_dimensions(self, image_id, band):
        fname = self.get_fname(image_id, band)
        with tiff.TiffFile(fname) as tfile:
            shape = tfile.pages[0].shape
        return shape

    def return_image(self, image_id, band):
        fname = self.get_fname(image_id, band)
        img = tiff.imread(fname)
        return img
        
    def get_clip_values(self, img, percentile_max=99.99):
        print('finding clip values')
        vmin = img[img!=0].min()
        vmax = np.percentile(img, percentile_max)
        return vmin, vmax

    def normalize_image(self, img):
        print('normalizing pixel values to range (0,1)')
        img = (img - img.min()) / (img.max() - img.min())
        return img

    def stitch_5x5(self, image_base, band, shrink_fac=None):

        # get dimensions for all images 
        shapes = {}
        for ix in range(5):
            for iy in range(5):
                image_id = '{}_{}_{}'.format(image_base, ix, iy)
                shapes[(ix,iy)] = self.get_image_dimensions(image_id, band)
                print('image_id={}, band={}, shape={}'.format(
                    image_id, band, shapes[(ix,iy)]))

        # get total image dimensions.  shape is a tuple (nch, nrows, ncols)
        # or (nrows, ncols) for single channel image.  therefore read 
        # dimensions from the right hand side 
        ncols = sum([shapes[(0,i)][-1] for i in range(5)])
        nrows = sum([shapes[(i,0)][-2] for i in range(5)])


        # create empty stitched array
        nch = N_CHANNELS[band]
        img = np.zeros((nch, nrows, ncols), dtype=np.float32)

        # put smaller images into single array
        for im_row in range(5):
            if im_row == 0: irow = 0
            for im_col in range(5):
                if im_col == 0: icol = 0
                image_id = '{}_{}_{}'.format(image_base, im_row, im_col)
                img1 = self.return_image(image_id, band).astype(np.float32)
                nrows1, ncols1 = img1.shape[-2:]
                img[:, irow:irow+nrows1, icol:icol+ncols1] = img1
                icol += ncols1
            irow += nrows1

        if shrink_fac is not None:
            print('resizing image. shrink_fac={}, shape={}'.format(
                shrink_fac, img.shape))
            nr = int(img.shape[1] * shrink_fac)
            nc = int(img.shape[2] * shrink_fac)
            print('new nr,nc = {},{}'.format(nr,nc))
            img_sm = np.zeros((nch, nr, nc), dtype=np.float32)
            for ich in range(img.shape[0]):
                img_sm[ich,:,:] = scipy.misc.imresize(img[ich,:,:], size=(nr,nc))
            img = img_sm
            del img_sm

        return img

class TrainWkt:
    """Handles the training labels (polygons in wkt format)"""

    def __init__(self, data_dir, fname='train_wkt_v4.csv'):
        path = os.path.join(data_dir, fname)
        self._df = pandas.read_csv(path)
        self.image_ids = self._df['ImageId'].unique()

    def return_image_df(self, image_id):
        bmask = self._df['ImageId']==image_id
        return self._df[bmask]

    def return_wkt_string(self, image_id, class_id):
        bmask1 = self._df['ImageId']==image_id
        bmask2 = self._df['ClassType']==class_id
        df_row = self._df.loc[bmask1 & bmask2, 'MultipolygonWKT']
        if df_row.shape[0] == 0:
            return None
        if df_row.shape[0] == 1:
            return df_row.iloc[0]
        else:
            raise ValueError('more than one row for image/class')

    def return_multipolygon(self, image_id, class_id):
        wkt_str = self.return_wkt_string(image_id, class_id)
        if wkt_str is None:
            multipolygon = None
        else:
            multipolygon = wkt.loads(wkt_str)
        return multipolygon


class GridSizes:
    """Handles the geo-coordinates from the grid sizes file."""

    def __init__(self, data_dir, fname='grid_sizes.csv'):
        path = os.path.join(data_dir, fname)
        self._df = pandas.read_csv(
            path, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    def get_xmax_ymin(self, image_id):
        bmask = self._df['ImageId']==image_id
        xmax_ymin = self._df.loc[bmask, ['Xmax', 'Ymin']].iloc[0].to_dict()
        return xmax_ymin['Xmax'], xmax_ymin['Ymin']


si = SatImages()

# plot image in gray-scale with polygons on top
#-------------------------------------------------
image_id = '6120_2_2'
band = 'P'
xmax, ymin = si.grid_sizes.get_xmax_ymin(image_id)
print('xmax={}, ymin={}'.format(xmax, ymin))

img = si.return_image(image_id, band)
print('image shape={}'.format(img.shape))

img = si.normalize_image(img)
vmin, vmax = si.get_clip_values(img)
extent = [0.0, xmax, ymin, 0.0]

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(img, cmap=cm.gray, vmin=vmin, vmax=vmax, extent=extent)

class_id = 4
multipolygon = si.train_wkt.return_multipolygon(image_id, class_id)
if multipolygon is None:
    print('no polygons for ImageId={}, ClassId={}'.format(image_id, class_id))
npoly = len(multipolygon)
print('class_id={}, npoly={}'.format(class_id, npoly))

for ip, poly in enumerate(multipolygon):
    mpl_poly = PolygonPatch(
        poly, fc='red', ec='green', alpha=0.3)
    ax.add_patch(mpl_poly)
    
plt.title('{}_{}_gray_poly - class={}'.format(image_id, band, class_id))
plt.savefig('{}_{}_gray_poly_class_{}.png'.format(image_id, band, class_id))

# plot 5x5 mosaic 
#-------------------------------------------------
image_base = '6120'
for band, shrink_fac in zip(['P', '3', 'M', 'A'], [0.2, 0.2, None, None]):
    img5x5 = si.stitch_5x5(image_base, band, shrink_fac=shrink_fac)
    img5x5 = img5x5.sum(axis=0)
    img5x5 = si.normalize_image(img5x5)
    vmin, vmax = si.get_clip_values(img5x5)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(img5x5, cmap=cm.gray, vmin=vmin, vmax=vmax)
    plt.title('{}_band={}_5x5'.format(image_base, band))
    plt.savefig('{}_band_{}_5x5.png'.format(image_id, band))

