# Here's some code for creating a benchmark that doesn't clean the data at all - just makes a submission out 
# of the original test set images.

# It doesn't run well in scripts right now due to time limit (10 minutes) & disk limitations (500mb). But maybe you 
# can make my code more efficient!

import os

from PIL import Image
import numpy as np
import pandas as pd
from functools import reduce
import gzip

def image_df_from_array(image_array, image_number):
    size = image_array.shape
    image_df = pd.DataFrame([(x, y) for x in range(size[0]) for y in range(size[1])], columns = ["row", "col"])
    image_df['pixel_value'] = image_array[image_df.row, image_df.col]
    image_df['value'] = image_df['pixel_value']/255
    # the solution file is 1-indexed.
    # this line is slow. is there a way to keep this operation vectorized in numpy?
    image_df['id'] = image_number + "_" + image_df.row.astype('str') + "_" + image_df.col.astype('str')
    return image_df[['id', 'value']]

def image_df_from_path(test_image_path):
    print("Working on image %s" % test_image_path)
    image_number = os.path.basename(test_image_path).split(".")[0]
    image_array = np.asarray(Image.open(test_image_path))
    return image_df_from_array(image_array, image_number)

def image_df_list_from_test_dir(test_images_dir, num_files_to_use=None):
    test_image_paths = [os.path.join(test_images_dir, relative_path) for relative_path in os.listdir(test_images_dir)]
    if num_files_to_use:
        test_image_paths = test_image_paths[0:num_files_to_use]
    return [image_df_from_path(test_image_path) for test_image_path in test_image_paths]

image_df_list = image_df_list_from_test_dir("../input/test/")
submission_df = reduce(lambda df1, df2: df1.append(df2), image_df_list)

submission_df.to_csv("submission.csv")
