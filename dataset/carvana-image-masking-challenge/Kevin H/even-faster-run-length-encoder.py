# Import libraries
import numpy as np # linear algebra
from PIL import Image

# Fast run length encoding
def rle (img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    
    return starts_ix, lengths

mask = np.array(Image.open('../input/train_masks/00087a6bd4dc_01_mask.gif'), dtype=np.uint8)
mask_rle = rle(mask)
print(mask_rle)