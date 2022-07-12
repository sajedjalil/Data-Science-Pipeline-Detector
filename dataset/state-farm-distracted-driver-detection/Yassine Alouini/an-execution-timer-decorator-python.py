import datetime
import time
import cv2
import os
import glob
from joblib import Parallel, delayed

PATH_TO_C0_TRAIN = os.path.join('../input', 'train', 'c0', '*.jpg')
TRAIN_FILES = glob.glob(PATH_TO_C0_TRAIN)    
N_PROCS = 2

def time_function_execution(function_to_execute):
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        computation_time = datetime.timedelta(seconds=end_time - start_time)
        print('I am done!')
        print('Computation lasted: {}'.format(computation_time))
        return result
    return compute_execution_time
    


## Time the loading of all training files (normal and in parallel)

def load_image(img_file):
    return cv2.imread(img_file)

@time_function_execution
def load_images(img_files):
    imgs = []
    for img_file in img_files:
        imgs.append(load_image(img_file))
    return imgs



@time_function_execution
def load_images_parallel(img_files):
    return Parallel(n_jobs=N_PROCS)(delayed(load_image)(img_file) 
                                    for img_file in img_files)


print("Loading train images")
images = load_images(TRAIN_FILES)

print("_____________________")

print("Loading train images in parallel")
images_parallel = load_images_parallel(TRAIN_FILES)

