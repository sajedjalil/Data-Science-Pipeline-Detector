# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import keras.backend as K
import imageio

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

'''
# `mask2rle` is taken from this notebook: https://www.kaggle.com/bigkizd/se-resnext50-89
'''

test_data_folder = "../input/severstal-steel-defect-detection/test_images"

import glob
test_image_paths = glob.glob( f'{test_data_folder}/*.jpg' )
test_image_paths.sort()

from keras.models import load_model
model_folder = '/kaggle/input/kg-defect-detection/' 
#model_folder = '/data/structured_folders/tmp/steel_defect_pretrained_models/'
model_paths = [ f'{model_folder}new_model_0.model',
                f'{model_folder}new_model_1.model',
                f'{model_folder}new_model_2.model',
                f'{model_folder}new_model_3.model' ]

K.set_learning_phase( 1 )
models = []
for path in model_paths:
    models.append( load_model( path ) )
    #models.append( load_model( path, custom_objects={'dice_loss_se':dummy_function} ) )

predictions = []
thresholds = [300, 300, 500, 300]


for test_image_path in test_image_paths:
    #print( f'determining image {test_image_path}' )
    input_image = np.asarray( imageio.imread( test_image_path ), dtype='float32' ) / 255.0
    if len(input_image.shape) == 3 :
        input_image = ( input_image[:,:,0] + input_image[:,:,1] + input_image[:,:,2] ) / 3.0
    input_image = input_image.reshape( (1,) + input_image.shape + (1,) )
    for model_index in range( 4 ) :
        output = models[model_index].predict( input_image )
        if type(output) is list: # some are MCNN models, producing multiple outputs
            output, *_ = output
        output[output>=0.5] = 1.0
        output[output<0.5] = 0.0
        if np.sum(output) < thresholds[model_index]:
            output = np.zeros( output.shape )
        rle = mask2rle( output )
        name = test_image_path[-13:]+f'_{model_index+1}'
        predictions.append( [name, rle] )


df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("./submission.csv", index=False)

