# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt
from skimage import io, color, filters
import cv2


def get_image(imageId):
      """ 
      Given an imageId, import the image and return a dict of image data from
      TIFF files as np array
  
  
      INPUTS
      imageID     -   the ID of the image
      path        -   where the data can be fonud, note that this defaults to three band
                      images unless otherwise specified
  
      OUTPUTS
      img         -   a dict containing the image of interest
      """

  
      # Create the full path to the image
      fullImagePath3 = '../input/three_band/' + imageId + '.tif'
      fullImagePathA = '../input/sixteen_band/' + imageId + '_A.tif'
      fullImagePathM = '../input/sixteen_band/' + imageId + '_M.tif'
      fullImagePathP = '../input/sixteen_band/' + imageId + '_P.tif'
  
  
      img3 = io.imread(fullImagePath3)
      imgA = io.imread(fullImagePathA)
      imgM = io.imread(fullImagePathM)
      imgP = io.imread(fullImagePathP)
  
      return ((img3, imgA, imgM, imgP))
      
     
     
     
def display_img(img):
     """
     Given an image, display each of the availabe bands 
     
 
     INPUTS 
         img     -   The img array
 
     OUTPUTS
         None
 
     """
 

     # Check the shape of the image to determine what to do 
     # This is the pan image and only has one dimension
     if np.ndim(img) == 2:
 
         fig = plt.figure()
         ax = fig.add_subplot(1,2,1)
         ax.imshow(img)
         
         plt.savefig('img.png')         
 
     # Three band image
     elif np.shape(img)[2] == 3:
 
         fig = plt.figure()
 
         ax = fig.add_subplot(2,2,1)
         ax.imshow(img[:,:,0])
 
         ax1 = fig.add_subplot(2,2,2)
         ax1.imshow(img[:,:,1])
 
         ax2 = fig.add_subplot(2,2,3)
         ax2.imshow(img[:,:,2])
 
 
         plt.show()
         plt.savefig('img.png')         
     
     
     # Eight bands
     elif np.shape(img)[0] == 8:
 
 
         fig = plt.figure()
 
         ax = fig.add_subplot(4,2,1)
         ax.imshow(img[0,:,:])
 
         ax1 = fig.add_subplot(4,2,2)
         ax1.imshow(img[1,:,:])
 
         ax2 = fig.add_subplot(4,2,3)
         ax2.imshow(img[2,:,:])
 
         ax3 = fig.add_subplot(4,2,4)
         ax3.imshow(img[3,:,:])
 
         ax4 = fig.add_subplot(4,2,5)
         ax4.imshow(img[4,:,:])
 
         ax5 = fig.add_subplot(4,2,6)
         ax5.imshow(img[5,:,:])
 
         ax6 = fig.add_subplot(4,2,7)
         ax6.imshow(img[6,:,:])
 
         ax7 = fig.add_subplot(4,2,8)
         ax7.imshow(img[7,:,:])
 
         #plt.show()
         plt.savefig('img.png')
      
      
      
img3, imgA, imgM, imgP = get_image('6120_4_4')

#display_img(img3)
#display_img(imgA)
#display_img(imgM)
display_img(imgP)
