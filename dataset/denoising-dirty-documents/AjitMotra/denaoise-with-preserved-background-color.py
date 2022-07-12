######################## Background Noise removal #########################
#Let's break image into back ground and foreground 
#At the same time, we want to keep the background look similar to original 
#book paper.

#Also lets remove small chunks of splattered ink by closing the gaps
#######################################################################

import numpy as np
from scipy import signal, ndimage
from PIL import Image

def load_im(path):
    return np.asarray(Image.open(path))/255.0

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_im_with_back(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 11)
    save('background.png', bg)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.1    
    save('foreground_mask.png', mask)
    back = np.average(bg);
    
    # Lets remove some splattered ink
    mod = ndimage.filters.median_filter(mask,2);
    mod = ndimage.grey_closing(mod, size=(2,2));
       
    # either return forground or average of background
       
    out = np.where(mod, inp, back)  ## 1 is pure white    
    return out;

inp_path = '../input/train/9.png'
out_path = 'output.png'

inp = load_im(inp_path)
out = denoise_im_with_back(inp)

save(out_path, out)