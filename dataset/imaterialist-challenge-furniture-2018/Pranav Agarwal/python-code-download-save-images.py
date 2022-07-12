import json
import urllib
import os

def Download_And_Save_Image(data, data_path):
    
    for i in range(len(data['images'])):
        img_url = data['images'][i]['url'][0]
        img_id = data['images'][i]['image_id']
        try:
            urllib.request.urlretrieve(img_url,data_path+'/'+str(img_id)+'.jpg')
        except Exception as e:
            continue
    


test_data = json.load(open('../input/test.json'))
train_data = json.load(open('../input/train.json'))
validation_data = json.load(open('../input/validation.json'))

test_data_path = 'test_set'
train_data_path = 'train_set'
validate_data_path =  'validation_set'

os.makedirs(test_data_path)
os.makedirs(train_data_path)
os.makedirs(validate_data_path)

Download_And_Save_Image(test_data, test_data_path)
Download_And_Save_Image(train_data, train_data_path)
Download_And_Save_Image(validation_data, validate_data_path)

