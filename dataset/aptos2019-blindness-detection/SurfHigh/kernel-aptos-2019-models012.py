# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mxnet as mx
from PIL import Image
import glob
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/aptos-models-20190731"))
print(os.listdir("../input/aptos2019-blindness-detection"))
image_dir = '../input/aptos2019-blindness-detection/test_images/'
models_dir = '../input/aptos-models-20190731/models_20190731/'
channels, dim_height, dim_length = 3, 224, 224

# Any results you write to the current directory are saved as output.
def process_image(img_code):

    image_path = image_dir + img_code + '.png'
    current_image = Image.open(image_path)
    
    ### crop image to make it square
    width, height = current_image.size
    left, right = max(int(width/2 - height/2), 0), min(int(width/2 + height/2), width)
    current_image = current_image.crop((left, 0, right, height ))
    current_image_resized = current_image.resize((dim_length, dim_height))

    big_array_now = np.zeros((channels, dim_height, dim_length), dtype=np.float32)

    img_array = np.array(current_image_resized)  ## 128*128*3
    big_array_now[0, :, :] = img_array[:, :, 0] # red
    big_array_now[1, :, :] = img_array[:, :, 1] # green
    big_array_now[2, :, :] = img_array[:, :, 2] # blue
        
    return np.array(big_array_now)


def parse_log_file():
        
        models = []
        flag = False
        with open(models_dir + 'models_log.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line.find('Prediction') >= 0:
                    fold = line.rpartition(' ')[2]
                if line.find('model') >= 0:
                    l = line.split(' ')
                    model_name = l[3]
                    file_prefix = model_name.split('-')[0]
                    meaner = float(l[6].replace('[','').replace('],',''))
                    epoch_to_load = l[11].replace(',','')
                    flag = True
                if flag:
                    models.append([fold, model_name, file_prefix, epoch_to_load, meaner])
                    flag = False

        
        return models
#         return [models[0]]

   
test = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
# test = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv', nrows=50)
predictions = []

# iterate over folders of training cross validation results
models = parse_log_file()
# print('{}'.format(models))
for m in models:
      
    print('Started prediction on model {} from fold {} at {}'.format(m[1], m[0], time.strftime('%x %X')))
    set_pred = []
    try: 
        devices = mx.gpu() #[mx.gpu(x) for x in param_dict['gpu']]
        print('device=GPU')
    except:
        devices = mx.cpu() #[mx.gpu(x) for x in param_dict['gpu']]
        print('device=CPU')        
        
    model_prefix = models_dir + 'fold_' + m[0] + '/' + m[2] 
    print(model_prefix)

    model = mx.mod.Module.load(model_prefix, int(m[3]), context=devices)
    print('Model loaded.')
        
    for i, img in enumerate(test['id_code']): 
        
        if i % 100 == 0:  # print every 10th row
            print('Processed {} rows... '.format(i))
        
        try:
            proceesed_image = process_image(img)
            val_img = np.zeros((1, 1, channels, dim_height, dim_length), dtype=np.float32)
            val_img[0, 0, :, :, :] = proceesed_image.astype(np.float32) / 255

    #         print('mean val before: ' + str(np.mean((val_img))))
            val_img -= m[4]
    #         print('mean val after: ' + str(np.mean((val_img))))

            val = mx.io.NDArrayIter(val_img, np.zeros((1), dtype=np.int8), 1)

            model.bind(data_shapes=val.provide_data, label_shapes=val.provide_label, for_training=False)
    #         print('Model binded.')
            img_pred = model.predict(val).asnumpy()[0]
            set_pred.append(img_pred)
    #         print(img_pred)
        except:
            print('Failed at image {}'.format(img))
            pass
        
    predictions.append(np.array(set_pred))
    print(np.array(set_pred).shape)
    print('Finished prediction on model {} from fold {} at {}'.format(m[1], m[0], time.strftime('%x %X')))
    
    
predictions =  np.dstack(list(predictions))
avg_prd = np.mean(predictions, axis=2)
arg_pred = np.argmax(avg_prd, axis=1)
test['diagnosis'] = arg_pred.astype(int)
test['diagnosis'].fillna(0, inplace=True) 
print(test.head(5))
test.to_csv('submission.csv',index=False)
    
# try: 
#     predictions =  np.dstack(list(predictions))
#     avg_prd = np.mean(predictions, axis=2)
#     arg_pred = np.argmax(avg_prd, axis=1)
#     test['diagnosis'] = arg_pred.astype(int)
#     test['diagnosis'].fillna(0, inplace=True) 
#     print(test.head(5))
#     test.to_csv('submission.csv',index=False)

# except:
#     submission_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
#     submission_df.to_csv('submission.csv', index=False)
    
    


