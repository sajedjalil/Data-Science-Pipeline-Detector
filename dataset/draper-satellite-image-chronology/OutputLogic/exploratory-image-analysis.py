import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os
# # Read in files
# 
# This is pretty routine stuff.
# 
# * We get a list of jpeg files, reading them in as needed with `matplotlib.pyplot.imread`.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])
set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)
# # Basic exploration
# 
# Just look at image dimensions, confirm it's 3 band (RGB), byte scaled (0-255).
first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)
np.min(first), np.max(first)
# For any image specific classification, clustering, etc. transforms we'll want to 
# collapse spatial dimensions so that we have a matrix of pixels by color channels.
pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_matrix))
# Scatter plots are a go to to look for clusters and separatbility in the data, but these are busy and don't reveal density well, so we
# switch to using 2d histograms instead. The data between bands is really correlated, typical with
# visible imagery and why most satellite image analysts prefer to at least have near infrared values.
#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])
_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))
fifth = plt.imread('../input/train_sm/set175_5.jpeg')
dims = np.shape(fifth)
pixel_matrix5 = np.reshape(fifth, (dims[0] * dims[1], dims[2]))
_ = plt.hist2d(pixel_matrix5[:,1], pixel_matrix5[:,2], bins=(50,50))
# We can look at variations between the scenes now and see that there's a significant
# amount of difference, probably due to sensor angle and illumination variation. Raw band
# differences will need to be scaled or thresholded for any traditional approach.
_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix5[:,2], bins=(50,50))
plt.imshow(first)
plt.imshow(fifth)
# Without coregistering portions of the image, the naive red band subtraction for change indication
# basically just shows the location shift between images.
plt.imshow(first[:,:,2] - fifth[:,:,1])
second = plt.imread('../input/train_sm/set175_2.jpeg')
plt.imshow(first[:,:,2] - second[:,:,2])
plt.imshow(second)
# # Initial impressions
# 
# Images aren't registered, so an image registration process between images with common overlap would probably be the first step in a traditional approach.
# Using a localizer in a deep learning context would probably be the newfangled way to tackle this.
# 
# Image content and differences will be dominated by topographic and built variations
# due to sensor orientation, resolution differences between scenes, and some registration accuracy will be impossible to factor out as
# the image hasn't been orthorectified and some anciliary data would be required for it
# to be done, e.g. georeferenceing against a previously orthorectified image.
# 
# So this is basically a basic computer vision task that deep learning will be a good fit for. The usual preprocessing steps
# and data expectations you'd see in remote sensing aren't fulfilled by this dataset.
# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)
plt.imshow(first)
ind0, ind1, ind2, ind3 = [np.where(clustered == x)[0] for x in [0, 1, 2, 3]]
# This code doesn't run on the server.
# 
# ```python
# from mpl_toolkits.mplot3d import Axes3D
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# plot_vals = [('r', 'o', ind0),
#              ('b', '^', ind1),
#              ('g', '8', ind2),
#              ('m', '*', ind3)]
# 
# for c, m, ind in plot_vals:
#     xs = pixel_matrix[ind, 0]
#     ys = pixel_matrix[ind, 1]
#     zs = pixel_matrix[ind, 2]
#     ax.scatter(xs, ys, zs, c=c, marker=m)
# 
# ax.set_xlabel('Blue channel')
# ax.set_ylabel('green channel')
# ax.set_zlabel('Red channel')
# ```
# quick look at color value histograms for pixel matrix from first image
import seaborn as sns
sns.distplot(pixel_matrix[:,0], bins=12)
sns.distplot(pixel_matrix[:,1], bins=12)
sns.distplot(pixel_matrix[:,2], bins=12)
# even subsampling is throwing memory error for me, :p
#length = np.shape(pixel_matrix)[0]
#rand_ind = np.random.choice(length, size=50000)
#sns.pairplot(pixel_matrix[rand_ind,:])
# # Day 2
# 
# We'll start by considering the entire sequence of a different image set this time and look at strategies
# for matching features across scenes.
set79 = [smj for smj in smjpegs if "set79" in smj]
print(set79)
img79_1, img79_2, img79_3, img79_4, img79_5 = \
  [plt.imread("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]
img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

plt.figure(figsize=(8,10))
plt.imshow(img_list[0])
plt.show()
# Tracking dimensions across image transforms is annoying, so we'll make a class to do that.
# Also I'm going to use this brightness normalization transform and visualize the image that
# way, good test scenario for class.
class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))
msi79_1 = MSImage(img79_1)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))
# # Brightness Normalization
# 
# Brightness Normalization is preprocessing strategy you can apply prior to using strategies
# to identify materials in a scene, if you want your matching algorithm
# to be robust across variations in illumination. See [Wu's paper](https://pantherfile.uwm.edu/cswu/www/my%20publications/2004_RSE.pdf).
def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm
bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()
msi79_2 = MSImage(img79_2)
bnorm79_2 = bnormalize(msi79_2.matrix)
bnorm79_2_img = msi79_2.to_matched_img(bnorm79_2)
plt.figure(figsize=(8,10))
plt.imshow(bnorm79_2_img)
plt.show()
msinorm79_1 = MSImage(bnorm_img)
msinorm79_2 = MSImage(bnorm79_2_img)

_ = plt.hist2d(msinorm79_1.matrix[:,2], msinorm79_2.matrix[:,2], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,1], msinorm79_2.matrix[:,1], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,0], msinorm79_2.matrix[:,0], bins=(50,50))
import seaborn as sns
sns.distplot(msinorm79_1.matrix[:,0], bins=12)
sns.distplot(msinorm79_1.matrix[:,1], bins=12)
sns.distplot(msinorm79_1.matrix[:,2], bins=12)
plt.figure(figsize=(8,10))
plt.imshow(img79_1)
plt.show()
np.max(img79_1[:,:,0])
# # Using thresholds with brightness normalization
# 
# Ok, so what am I even doing here? Well, my goal is to try and figure out simple threshold selection
# methods for getting high albedo targets out of a scene so I could then theoretically track them
# between scenes. For example, a simple blob/aggregation to centroid (in coordinates or in subsampled
# image bins) would give me a means to look at plausible structural similarities in distributions
# between scenes, then use that to anchor a comparison of things that change.
# 
# The brightness normalization step is helpful because thresholds that aren't anchored by a
# preprocessing step end up being arbitrary and can't generalize between scenes even in the same
# image set, whereas thresholds following brightness normalization tend to pull out materils that stand
# out from the background more reliably. See the following demonstration:
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_1[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_2[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
print(np.min(bnorm79_2_img[:,:,0]))
print(np.max(bnorm79_2_img[:,:,0]))
print(np.mean(bnorm79_2_img[:,:,0]))
print(np.std(bnorm79_2_img[:,:,0]))
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm79_2_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow((bnorm79_2_img[:,:,0] > 0.9999) & \
           (bnorm79_2_img[:,:,1] < 0.9999) & \
           (bnorm79_2_img[:,:,2] < 0.9999))
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.995)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(bnorm_img[2000, 1000, :])
plt.subplot(122)
plt.plot(img79_1[2000, 1000, :])
from scipy import spatial

pixel = msi79_1.matrix[2000 * 1000, :]
np.shape(pixel)
# # Something's borked here
# 
# Think I'm gonna have to verify cosine similarity behavior for scipy here.
# 
# ```python
# def spectral_angle_mapper(pixel):
#     return lambda p2: spatial.distance.cosine(pixel, p2)
# 
# match_pixel = np.apply_along_axis(spectral_angle_mapper(pixel), 1, msi79_1.matrix)
# 
# plt.figure(figsize=(10,6))
# plt.imshow(msi79_1.to_flat_img(match_pixel < 0.0000001))
# 
# def summary(mat):
#     print("Max: ", np.max(mat),
#           "Min: ", np.min(mat),
#           "Std: ", np.std(mat),
#           "Mean: ", np.mean(mat))
#     
# summary(match_pixel)
# ```
# # Rudimentary Transforms, Edge Detection, Texture
set144 = [MSImage(plt.imread(smj)) for smj in smjpegs if "set144" in smj]
plt.imshow(set144[0].image)
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
# # Sobel Edge Detection
# 
# A Sobel filter is one means of getting a basic edge magnitude/gradient image. Can be useful to
# threshold and find prominent linear features, etc. Several other similar filters in skimage.filters
# are also good edge detectors: `roberts`, `scharr`, etc. and you can control direction, i.e. use
# an anisotropic version.
# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel(set144[0].image[:750,:750,2]))
from skimage.filters import sobel_h

# can also apply sobel only across one direction.
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(set144[0].image[:750,:750,2]), cmap='BuGn')
from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(set144[0].matrix)
set144_0_pca = pca.transform(set144[0].matrix)
set144_0_pca_img = set144[0].to_matched_img(set144_0_pca)
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,1], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,2], cmap='BuGn')
# # GLCM Textures
# 
# Processing time can be pretty brutal so we subset the image. We'll create texture images so
# we can characterize each pixel by the texture of its neighborhood.
# 
# GLCM is inherently anisotropic but can be averaged so as to be rotation invariant. For more on GLCM, see [the tutorial](http://www.fp.ucalgary.ca/mhallbey/tutorial.htm).
# 
# A good article on use in remote sensing is [here](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4660321&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4660321):
# 
# Pesaresi, M., Gerhardinger, A., & Kayitakire, F. (2008). A robust built-up area presence index by anisotropic rotation-invariant textural measure. Selected Topics in Applied Earth Observations and Remote Sensing, IEEE Journal of, 1(3), 180-192.
sub = set144[0].image[:150,:150,2]
def glcm_image(img, measure="dissimilarity"):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    texture = np.zeros_like(sub)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(sub.shape[1] ):  
          
            # don't calculate at edges
            if (i < 3) or \
               (i > (img.shape[0])) or \
               (j < 3) or \
               (j > (img.shape[0] - 4)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-3: i+4, j-3 : j+4]  
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            texture[i,j] = greycoprops(glcm, measure)  
    return texture
dissimilarity = glcm_image(sub, "dissimilarity")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(dissimilarity, cmap="bone")
plt.subplot(1,2,2)
plt.imshow(sub, cmap="bone")
# # HSV Transform
# 
# Since this contest is about time series ordering, I think it's possible there may be useful
# information in a transform to HSV color space. HSV is useful for identifying shadows and illumination, as well
# as giving us a means to identify similar objects that are distinct by color between scenes (hue), 
# though there's no guarantee the hue will be stable.
from skimage import color

hsv = color.rgb2hsv(set144[0].image)
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image, cmap="bone")
plt.subplot(2,2,2)
plt.imshow(hsv[:,:,0], cmap="bone")
plt.subplot(2,2,3)
plt.imshow(hsv[:,:,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,2], cmap='bone')
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image[:200,:200,:])
plt.subplot(2,2,2)
plt.imshow(hsv[:200,:200,0], cmap="PuBuGn")
plt.subplot(2,2,3)
plt.imshow(hsv[:200,:200,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:200,:200,2], cmap='bone')
fig = plt.figure(figsize=(8, 6))
plt.imshow(hsv[200:500,200:500,0], cmap='bone')
hsvmsi = MSImage(hsv)
# # Shadow Detection
# 
# We can apply a threshold to the V band now to find dark areas that are probably thresholds. Let's
# look at the distribution of all values then work interactively to find a good filter value.
import seaborn as sns
sns.distplot(hsvmsi.matrix[:,0], bins=12)
sns.distplot(hsvmsi.matrix[:,1], bins=12)
sns.distplot(hsvmsi.matrix[:,2], bins=12)
plt.imshow(hsvmsi.image[:,:,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(set144[0].image[:250,:250,:])
plt.subplot(1,2,2)
plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
img2 = plt.imshow(set144[0].image[:250,:250,:], interpolation='nearest')
img3 = plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap='binary_r', alpha=0.4)
plt.show()
# Could we glean something useful about sun position from shadow orientation if we could accurately
# reference the image?