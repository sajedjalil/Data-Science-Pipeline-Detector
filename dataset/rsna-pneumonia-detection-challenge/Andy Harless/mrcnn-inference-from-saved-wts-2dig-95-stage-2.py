min_conf = 0.95
digits = 2

if digits==2:
    weight_files = [   # Files previously run with 2-digit output at 0.95 min_conf
        'mask_rcnn_pneumonia_0015randomly_higher.h5',
        'mask_rcnn_pneumonia_0018_v1a1f0.h5',
        'mask_rcnn_pneumonia_0018_v1a1f1.h5',
        'mask_rcnn_pneumonia_0018_v1a1f2.h5',
        'mask_rcnn_pneumonia_0018_v1a1f3.h5',
        'mask_rcnn_pneumonia_0018_v1a1f4.h5',
        'mask_rcnn_pneumonia_0018_v8a2f0.h5',  # lost the v1 version
        'mask_rcnn_pneumonia_0018_v1a2f1.h5',
        'mask_rcnn_pneumonia_0018_v1a2f2.h5',
        'mask_rcnn_pneumonia_0018_v1a2f3.h5',
        'mask_rcnn_pneumonia_0018_v1a2f4.h5'
        ]
elif digits==4:
    weight_files = [   # Files previously run with 4-digit output at 0.70 min_conf
        'mask_rcnn_pneumonia_0018_v8a1f0.h5',
        'mask_rcnn_pneumonia_0018_v8a1f1.h5',
        'mask_rcnn_pneumonia_0018_v8a1f2.h5',
        'mask_rcnn_pneumonia_0018_v8a1f3.h5',
        'mask_rcnn_pneumonia_0018_v8a1f4.h5',
        'mask_rcnn_pneumonia_0018_v8a2f0.h5',
        'mask_rcnn_pneumonia_0018_v8a2f1.h5',
        'mask_rcnn_pneumonia_0018_v8a2f2.h5',
        'mask_rcnn_pneumonia_0018_v8a2f3.h5',
        'mask_rcnn_pneumonia_0018_v8a2f4.h5'
        ]
else:
    print( 'BAD DIGITS NUMBER' )


URL_STEM = 'http://andy.harless.us/rsnaweights/'

import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold
from skimage import exposure
import hashlib
from subprocess import call



sys.stdout = open('log.txt', 'w')  # Log will not print correctly in kernel environment, so redirect output



DATA_DIR = '/kaggle/input'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'

call(["git", "clone", "https://www.github.com/matterport/Mask_RCNN.git"])
os.chdir('Mask_RCNN')

# Hack to suppress progress bar in output
os.chdir('mrcnn')
import shutil
shutil.move( 'model.py', 'model.py~' )
destination= open( 'model.py', 'w' )
source= open( 'model.py~', 'r' )
for line in source:
    destination.write( line )
    if 'workers=workers' in line: 
        destination.write( "            verbose=2,\n" )
source.close()
destination.close()
os.chdir('..')

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')


# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    VERBOSE = 2
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 100
    
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.25,
        "rpn_bbox_loss": 0.15,
        "mrcnn_class_loss": 0.3,
        "mrcnn_bbox_loss": 0.2,
        "mrcnn_mask_loss": 0.1
    }

config = DetectorConfig()
config.display()



class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
            image[:,:,0] = (exposure.equalize_adapthist(image[:,:,0])*255)
            image[:,:,1] = exposure.rescale_intensity(image[:,:,1], out_range=(0, 255))
            image[:,:,2] = (exposure.equalize_hist(image[:,:,2], mask=image[:,:,0])*255)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)
        
        
        

def get_fn(dicom_dir, patientId):
    return os.path.join(dicom_dir, patientId + '.dcm')

def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = get_fn(dicom_dir, row['patientId'])
        image_annotations[fp].append(row)
    return image_fps, image_annotations
    

ORIG_SIZE = 1024



class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=config.DETECTION_MIN_CONFIDENCE):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString\n")

        for image_id in image_fps:
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], digits))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str

            file.write(out_str+"\n")
            

inference_config = InferenceConfig()


with open(os.path.join(ROOT_DIR,'md5sums.log'), 'w') as md5file:

    for wf in weight_files:
    
        file_url = URL_STEM + wf
        call( ['wget', '-q', file_url] )
        md5file.write( hashlib.md5(open(wf,'rb').read()).hexdigest() + ' ' + wf + '\n' )
        
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode='inference', 
                                  config=inference_config,
                                  model_dir=ROOT_DIR)
        
        # Load trained weights (fill in path to trained weights here)
        print("Loading weights from ", wf)
        model.load_weights(wf, by_name=True)
        
        
        # Get filenames of test dataset DICOM images
        test_image_fps = get_dicom_fps(test_dicom_dir)
        
        
        version = wf.split('.')[-2].split('_')[-1]
        
        submission_fp = os.path.join(ROOT_DIR, f'submission_mrcnn_{version}.csv')
        predict(test_image_fps, filepath=submission_fp, min_conf=min_conf)
        output = pd.read_csv(submission_fp)
        
        call([ 'rm', wf ])

 
call(["rm", "-rf", "/kaggle/working/Mask_RCNN"])