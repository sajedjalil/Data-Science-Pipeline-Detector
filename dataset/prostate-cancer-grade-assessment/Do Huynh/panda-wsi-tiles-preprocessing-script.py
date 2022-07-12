# DEPENDANCIES ###########################################################################
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import openslide
from openslide import OpenSlideError
from IPython.display import Image
import seaborn as sns
import multiprocessing
import datetime
# / DEPENDANCIES #########################################################################
print("Dependencies loaded")

# UTILITIES ##############################################################################
class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
# / UTILITIES ############################################################################
print('Utilities loaded')

# PARAMETERS #############################################################################
BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'
OUTPUT_DIR = './'
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TRAIN_EXT = ".tiff"
MASK_DIR = os.path.join(BASE_DIR, "train_label_masks")
MASK_EXT = "_mask.tiff"

print("Parameters loaded")
# /PARAMETERS ############################################################################

# DATASET ################################################################################
# Get train/label slides ID
train = glob.glob1(TRAIN_DIR, "*" + TRAIN_EXT)
label = glob.glob1(MASK_DIR, "*" + MASK_EXT)

# Keep only image_id
train = [x[:-len(TRAIN_EXT)] for x in train]
label = [y[:-len(MASK_EXT)] for y in label]

# Add filenames to dataframe
train_df = pd.read_csv(BASE_DIR + 'train.csv')

# Add train file column for each existing file in train folder
train_df['train_file'] = list(map(lambda x : x + TRAIN_EXT if x in set(train) else '', 
                              train_df['image_id']))
# Add label file column for each existing file in mask folder
train_df['label_file'] = list(map(lambda y : y + MASK_EXT if y in set(label) else '', 
                              train_df['image_id']))

# Split dataframe by provider / we keep radboud scoring because their mask labels are more details
print('Dataframe original:', len(train_df))
train_radboud = train_df[train_df['data_provider'] == 'radboud'].copy()
print('Dataframe after provider select:', len(train_radboud))
# Keep only row with both train and label file
train_radboud = train_radboud[train_radboud['train_file'] != '']
print('Dataframe after file select:', len(train_radboud))
train_radboud = train_radboud[train_radboud['label_file'] != '']
print('Dataframe after label select:', len(train_radboud))

# Release memory
train_df = None
# / DATASET ##############################################################################


# FUNCTIONS ##############################################################################
# Open a slide
def open_slide(filename):
    """
    Open a whole-slide image (*.svs, etc).
    :filename : Name of the slide file.
    return: an OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide    

# Generate score tiles from files and masks
def generate_tiles_labels(file):
    """
    Generate a list of tiles with coordonnate and label from file/mask whole-slide image .tiff
    :file : Name of the slide file (must start and end with define directory and extension)
    return: a list of dictionnary tiles with gleason labels
    """
    interval = 32
    tiles = []
    filepath = os.path.join(TRAIN_DIR, file)
    image_id = file[:-len(TRAIN_EXT)]
    maskpath = os.path.join(MASK_DIR, image_id + MASK_EXT)

    # Open files
    biopsy = open_slide(filepath)
    mask = open_slide(maskpath)
    
    # Read lowest definition image
    level = biopsy.level_count - 1
    dimensions = biopsy.level_dimensions[level]

    # Get number of gridsquares in x and y direction
    nx=int(dimensions[0]/interval)
    ny=int(dimensions[1]/interval)
    #tiles = np.zeros((nx, ny))

    # Browse each tiles
    level = 1
    scale = 4
    size = interval*scale # Tile size depend on scale factor
    dimensions = (size, size)
    num_pixels = dimensions[0]*dimensions[1]

    for i in range(nx):
        for j in range(ny):  
            x, y = i*interval*16, j*interval*16 #Localization from the level 0 => * max scale interval to get coordinate
            
            # Read biopsy file
            sample = biopsy.read_region((x, y), level, dimensions)
            sample = sample.convert("1") #Convert to black and white
            score = 1-np.count_nonzero(sample)/num_pixels #Normalize the value between 0 and 1 (0=white, 1=black)

            # Keep only not empty tiles
            if score > 0.1:
                # Read mask file
                sample = mask.read_region((x, y), level, dimensions)
                sample = np.array(sample.convert('RGB'))
                
                key, value = np.unique(sample[:,:,0], return_counts=True) # Count by pixel score present on the first color channel
                scores = dict(zip(key, value)) #Create a score dict
                
                PREFIX = 'gleason_'
                labels = {PREFIX+str(k) : 0 for k in range(6)} #Create an empty score list from 0 to 5
                for k in scores.keys():
                    labels[PREFIX+str(k)] = scores[k]/num_pixels #Update score list
                
                # Add tile
                tile = {'image_id':file[:-len('.tiff')], 'tile':(i,j), 'x':x, 'y':y, 'level':level, 'size':size,}   
                tiles.append({**tile, **labels})

    # Close
    biopsy.close()
    mask.close()
    sample = None
    return tiles
print('Functions loaded')
# / FUNCTIONS ############################################################################

t = Time() # Launch timer

# EXTRACTION #############################################################################
# Extract label tiles for all slides (with multiprocessing)
print('Start tiles generation...')
files = train_radboud['train_file'].values

# Processes available
num_processes = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_processes)

# Image per process split
num_files = len(files)
if num_processes > num_files:
    num_processes = num_files
files_per_process = num_files / num_processes

print("Number of processes: " + str(num_processes))
print("Number of files: " + str(num_files))

# start tasks pooling
tiles = []

def get_tiles(result):
    tiles.append(result)
    
for file in files:
    result = pool.apply_async(generate_tiles_labels, args = (file,), callback = get_tiles)

pool.close()
pool.join()

tiles = np.concatenate(tiles)
print('Extract',len(tiles),'tiles from', len(files),'slides')

# OBSERVATION: ~30m to extract and score 603602 tiles from 5060 slides on 4 process
# / EXTRACTION ###########################################################################

# OUTPUT #################################################################################
# Export tiles for further usage
tiles_final_df = pd.DataFrame(tiles.tolist())
tiles_final_df.to_csv(OUTPUT_DIR + 'PANDA_tiles_labels_final.csv')
print('Tiles exported')

# OBSERVATION: output csv file of size 72,4Mb
# / OUTPUT ###############################################################################

t.elapsed_display() # Print timer