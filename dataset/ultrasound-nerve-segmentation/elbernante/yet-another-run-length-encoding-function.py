# Python 3
# A small 4-liner utility function that encodes image masks to run length encoding.
# Can be further improved by using vectorization instead of for-loops and generators.

from itertools import tee, chain # functions creating iterators for efficient looping
import numpy as np # linear algebra
import cv2 # image processing

def run_length(label, neg_class=0, enumerate=enumerate, next=next, zip=zip, tee=tee, chain=chain, np=np):
    """Returns generator of run length encoding as (position, length) pair of tuples.
    
    Parameters:
        label      - Required. numpy array with shape (H, W).
                     Expected to have binary values.
        neg_class  - Optional. Value to be classified as the negative class.
                     All other values are classified as positive class.
                     Defaults to 0.
    
    All other parameters are not strictly necessary but are declared as default values
    to avoid global lookups and thereby have performance gains.
    
    Returns:
        A generator for a list of tuples. Each tuple is a (position, length) pair.
    """
    
    padded = chain([neg_class], np.ravel(label, order='F'), [neg_class])
    a, b = tee(padded); next(b, None)
    switches = (i + 1 for i, (a, b) in enumerate(zip(a, b)) if a != b)
    return ((p, next(switches) - p) for p in switches)

# Utility to concatenate run length encoded tuples to string
string_rle = lambda rle: ' '.join(['{} {}'.format(p, l) for p, l in rle])

# Convenience method to convert from numpy.array format to string format
to_string_rle = lambda np_mask: string_rle(run_length(np_mask))


# Example usage:
mask = cv2.imread('../input/train/1_1_mask.tif', cv2.IMREAD_GRAYSCALE)
print(to_string_rle(mask))