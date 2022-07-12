"""

Animated Images with Outlined Nerve Area

Videos of ultrasounds are helpful for gaining a better sense of the 3D 
structure of the BP and surrounding tissues. 

Unfortunately, we were NOT given videos for this Kaggle challenge, 
only images from those videos.  Furthermore, we've been told that the images
for each patient are unordered. Therefore, one cannot just concatenate all 
the images together to easily reconstruct a video.

To to rectify this, this script attempts to create a reasonable reconstruction
of a patient's ultrasound video. It creates an animated GIF after finding
a sensible order for the patient's images.  The ordering assumes that
changes between adjacent frames in a video are generally smaller than
changes between randomly selected frames. Therefore, finding a sequence
that minimize the sum of changes between frames should approximate the
original video. 

The reconstruction also includes a red outline surrounding any nerve tissue, 
derived from masks that the ultrasound image annotators have provided. 

by Chris Hefele, May 2016

"""

import glob
import os.path 
import cv2
import numpy as np 
import collections
import matplotlib
import scipy.spatial.distance
import itertools
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


IMAGE_DIR        = '../input/train/'
MSEC_PER_FRAME   = 200  
MSEC_REPEAT_DELAY= 2000
ADD_MASK_OUTLINE = True
TILE_MIN_SIDE    = 50     # pixels; see tile_features()
SHOW_GIF         = False  # matplotlib popup of animation 


def get_image(f):
    # Read image file 
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print 'Read:', f
    return img


def grays_to_RGB(img):
    # Convert a 1-channel grayscale image into 3 channel RGB image
    return np.dstack((img, img, img))


def image_plus_mask(img, mask):
    # Returns a copy of the grayscale image, converted to RGB, 
    # and with the edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # chan 0 = bright red
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color


def to_mask_path(f_image):
    # Convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(f_image)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)


def add_masks(images):
    # Return copies of the group of images with mask outlines added
    # Images are stored as dict[filepath], output is also dict[filepath]
    images_plus_masks = {} 
    for f_image in images:
        img  = images[f_image]
        mask = cv2.imread(to_mask_path(f_image))
        images_plus_masks[f_image] = image_plus_mask(img, mask)
    return images_plus_masks


def get_patient_images(patient):
    # Return a dict of patient images, i.e. dict[filepath]
    f_path = IMAGE_DIR + '%i_*.tif' % patient 
    f_ultrasounds = [f for f in glob.glob(f_path) if 'mask' not in f]
    images = {f:get_image(f) for f in f_ultrasounds}
    return images


def image_features(img):
    return tile_features(img)   # a tile is just an image...


def tile_features(tile, tile_min_side = TILE_MIN_SIDE):
    # Recursively split a tile (image) into quadrants, down to a minimum 
    # tile size, then return flat array of the mean brightness in those tiles.
    tile_x, tile_y = tile.shape
    mid_x = tile_x / 2
    mid_y = tile_y / 2
    if (mid_x < tile_min_side) or (mid_y < tile_min_side):
        return np.array([tile.mean()]) # hit minimum tile size
    else:
        tiles = [ tile[:mid_x, :mid_y ], tile[mid_x:, :mid_y ], 
                  tile[:mid_x , mid_y:], tile[mid_x:,  mid_y:] ] 
        features = [tile_features(t) for t in tiles]
        return np.array(features).flatten()


def feature_dist(feats_0, feats_1):
    # Definition of the distance metric between image features
    return scipy.spatial.distance.euclidean(feats_0, feats_1)


def feature_dists(features):
    # Calculate the distance between all pairs of images (using their features)
    dists = collections.defaultdict(dict)
    f_img_features = features.keys()
    for f_img0, f_img1 in itertools.permutations(f_img_features, 2):
        dists[f_img0][f_img1] = feature_dist(features[f_img0], features[f_img1])
    return dists


def image_seq_start(dists, f_start):

    # Given a starting image (i.e. named f_start), greedily pick a sequence 
    # of nearest-neighbor images until there are no more unpicked images. 

    f_picked = [f_start]
    f_unpicked = set(dists.keys()) - set([f_start])
    f_current = f_start
    dist_tot = 0

    while f_unpicked:

        # Collect the distances from the current image to the 
        # remaining unpicked images, then pick the nearest one 
        candidates = [(dists[f_current][f_next], f_next) for f_next in f_unpicked]
        dist_nearest, f_nearest = list(sorted(candidates))[0]

        # Update the image accounting & make the nearest image the current image 
        f_unpicked.remove(f_nearest)
        f_picked.append(f_nearest)
        dist_tot += dist_nearest
        f_current = f_nearest 

    return (dist_tot, f_picked)


def image_sequence(dists):

    # Return a sequence of images that minimizes the sum of 
    # inter-image distances. This function relies on image_seq_start(), 
    # which requires an arbitray starting image. 
    # In order to find an even lower-cost sequence, this function
    # tries all possible staring images and returns the best result.

    f_starts = dists.keys()
    seqs = [image_seq_start(dists, f_start) for f_start in f_starts]
    dist_best, seq_best = list(sorted(seqs))[0]
    return seq_best


def grayscale_to_RGB(img):
    return np.asarray(np.dstack((img, img, img)), dtype=np.uint8)


def build_gif(imgs, fname, show_gif=True, save_gif=True, title=''):
    # Create an animated GIF file from a sequence of images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, 
                        wspace=None, hspace=None)  # removes white border
    #imgs = [(ax.imshow(img), ax.set_title(title)) for img in imgs] 
    imgs = [ (ax.imshow(img), 
              ax.set_title(title), 
              ax.annotate(n_img,(5,5))) for n_img, img in enumerate(imgs) ] 

    img_anim = animation.ArtistAnimation(fig, imgs, interval=MSEC_PER_FRAME, 
                                repeat_delay=MSEC_REPEAT_DELAY, blit=False)
    if save_gif:
        print('Writing:', fname)
        img_anim.save(fname, writer='imagemagick')
    if show_gif:
        plt.show()
    plt.clf() # clearing the figure when done prevents a memory leak 


def write_gif(f_seq, images, fname):
    imgs = [images[f] for f in f_seq] # get images indexed by their filenames
    build_gif(imgs, fname, show_gif=SHOW_GIF)


def write_patient_video(patient):
    # Given a patient number, create an animaged GIF of their ultrasounds
    # including an outline of any mask created that identifies nerve tissue.
    images       = get_patient_images(patient=patient)
    images_masks = add_masks(images)
    features     = { f : image_features(images[f]) for f in images }
    dists        = feature_dists(features)
    f_seq        = image_sequence(dists)
    write_gif(f_seq, images_masks, 'patient-%02i.gif' % patient)


def main():

    # Animations for patients 32 and 41 are particularly good examples. 
    write_patient_video(patient=41)
    write_patient_video(patient=32)

main()