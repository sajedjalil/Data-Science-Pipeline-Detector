import os
import glob
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.segmentation.boundaries import dilation
from skimage.morphology import square

dataset_root_path = '../input/stage1_train/'


# train image root path
image_path = os.path.join(dataset_root_path)
image_name_list = os.listdir(image_path)

for i, image_name in enumerate(image_name_list):
    print(i, image_name)
    image = Image.open(os.path.join(image_path, image_name, 'images', image_name + '.png')).convert('RGB')
    h, w, c = np.asarray(image).shape
    one_mask = np.zeros((h, w), dtype=bool)
    # mask image path
    masks_path = os.path.join(image_path, image_name, 'masks')
    masks_image_list = glob.glob(masks_path + '/*.png')

    boundary_image = np.zeros((h, w), dtype=np.uint16)
    for mask_path in masks_image_list:
        mask_image = Image.open(mask_path).convert('L')
        mask_image = np.asarray(mask_image, dtype=np.uint8)
        one_mask = one_mask | (mask_image > 128)

        b = find_boundaries(mask_image, mode='outer').astype(np.uint8)
        boundary_image += b

    one_mask = (one_mask * 255).astype(np.uint8)
    one_mask_image = Image.fromarray(one_mask)

    
    boundary_image[boundary_image >= 2] = 1  # set boundary pixel to 1, otherwise 0
    boundary_image = dilation(boundary_image, square(2))

    boundary_image = Image.fromarray(np.asarray(boundary_image * 255, dtype=np.uint8))
    # uncommnet in your local computer and set the path to save the generated images
    # one_mask_image.save(os.path.join(image_path, image_name, image_name+'_mask.png'))
    # boundary_image.save(os.path.join(image_path, image_name, image_name+'_boundary.png'))
