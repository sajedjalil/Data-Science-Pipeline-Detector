# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Any results you write to the current directory are saved as output.

import tensorflow as tf
import csv
import os
import collections

datadir = '../input'

###########################################
# gender process
print('gender process')

gender_dict = {'F': 0, 'M': 1}
group_dict = {
    'F23-': 0,
    'F24-26': 1,
    'F27-28': 2,
    'F29-32': 3,
    'F33-42': 4,
    'F43+': 5,
    'M22-': 6,
    'M23-26': 7,
    'M27-28': 8,
    'M29-31': 9,
    'M32-38': 10,
    'M39+': 11
}

device_gender_dict = {}
device_group_dict = {}

with open(os.path.join(datadir,'gender_age_train.csv'), newline='', encoding="utf8") as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'device_id':
            device_gender_dict[row[0]] = gender_dict[row[1]]
            device_group_dict[row[0]] = group_dict[row[3]]
csvfile.close()

####################################
# brand and model process
print('brand and mo process')

# indexing device
device_index = {}

# indexing brand
brand_index = {}

# indexing mo
model_index = {}

# device with indexed brand and mo
device_indexed_brand_model = {}

dindex = 0
bindex = 0
mindex = 0

with open(os.path.join(datadir,'phone_brand_device_model.csv'), newline='', encoding="utf8") as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'device_id':
            # deal with device index
            device_index[row[0]] = dindex
            dindex += 1

            # deal with brand index
            if row[1] not in brand_index:
                brand_index[row[1]] = bindex
                bindex += 1

            # deal with mo index
            if row[2] not in model_index:
                model_index[row[2]] = mindex
                mindex += 1

            # deal with device brand model
            device_indexed_brand_model[row[0]] = [
                brand_index[row[1]],
                model_index[row[2]]
            ]
csvfile.close()

############################################################
# app process
# in naive version, we do not use app as a feature
print('app process')
print('in naive version, we do not use app as a feature')

############################################################
# label process
print('label process')
label_index = {}
idx = 0
with open(os.path.join(datadir,'label_categories.csv'), newline='', encoding="utf8") as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'label_id':
            label_index[row[0]] = idx
            idx += 1
csvfile.close()

app_index = {}
idx = 0
with open(os.path.join(datadir,'app_labels.csv'), newline='', encoding='utf8') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'app_id' and row[0] not in app_index:
            app_index[row[0]] = idx
            idx += 1
csvfile.close()

naive_nn_disp = {}
naive_nn_disp['brand'] = 0
naive_nn_disp['model'] = len(brand_index)
naive_nn_disp['label'] = naive_nn_disp['model'] + len(model_index)
naive_nn_disp['active_label'] = naive_nn_disp['label'] + len(label_index)
naive_nn_disp['event'] = naive_nn_disp['active_label'] + len(label_index)
naive_nn_disp['longitude'] = naive_nn_disp['event'] + 1
naive_nn_disp['latitude'] = naive_nn_disp['longitude'] + 1
naive_nn_disp['total'] = naive_nn_disp['latitude'] + 1

event_idx = naive_nn_disp['event']
longitude_idx = naive_nn_disp['longitude']
latitude_idx = naive_nn_disp['latitude']

device_feature_dict = {}
with open(os.path.join(datadir,'phone_brand_device_model.csv'), newline='', encoding="utf8") as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'device_id':
            device_feature_dict[row[0]] = {}
            brand = row[1]
            model = row[2]
            brand_id = brand_index[brand] + naive_nn_disp['brand']
            model_id = model_index[model] + naive_nn_disp['model']
            device_feature_dict[row[0]][brand_id] = 1
            device_feature_dict[row[0]][model_id] = 1
csvfile.close()

# generate event-app dict
event_app_dict = {}
active_event_app_dict = {}
with open(os.path.join(datadir,'app_events.csv'), newline='', encoding='utf8') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'event_id':
            if row[0] not in event_app_dict:
                event_app_dict[row[0]] = [row[1]]
            else:
                event_app_dict[row[0]].append(row[1])
            if row[3] == '1':
                if row[0] not in active_event_app_dict:
                    active_event_app_dict[row[0]] = [row[1]]
                else:
                    active_event_app_dict[row[0]].append(row[1])
csvfile.close()

# generate app-label dict
app_label_dict = {}
with open(os.path.join(datadir,'app_labels.csv'), newline='', encoding='utf8') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'app_id':
            if row[0] not in app_label_dict:
                app_label_dict[row[0]] = [row[1]]
            else:
                app_label_dict[row[0]].append(row[1])
csvfile.close()

# generate device features
non_zero_event_count = {}
with open(os.path.join(datadir,'events.csv'), newline='', encoding='utf8') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'event_id' and row[1] in device_feature_dict:
            # deal with event count
            if event_idx not in device_feature_dict[row[1]]:
                device_feature_dict[row[1]][event_idx] = 1
            else:
                device_feature_dict[row[1]][event_idx] += 1

            # deal with longitude and latitude
            if row[3] != '0.00' and row[4] != '0.00':
                if row[1] not in non_zero_event_count:
                    non_zero_event_count[row[1]] = 1
                    device_feature_dict[row[1]][longitude_idx] = float(row[3])
                    device_feature_dict[row[1]][latitude_idx] = float(row[4])
                else:
                    non_zero_event_count[row[1]] += 1
                    device_feature_dict[row[1]][longitude_idx] += float(row[3])
                    device_feature_dict[row[1]][latitude_idx] += float(row[4])

            # deal with app, label and active app and active label
            event_id = row[0]
            if event_id in event_app_dict:
                app_list = event_app_dict[event_id]
                for app in app_list:
                    label_list = app_label_dict[app]
                    for label in label_list:
                        label_id = label_index[label] + naive_nn_disp['label']
                        if label_id not in device_feature_dict[row[1]]:
                            device_feature_dict[row[1]][label_id] = 1
                        else:
                            device_feature_dict[row[1]][label_id] += 1

            if event_id in active_event_app_dict:
                active_app_list = active_event_app_dict[event_id]
                for app in active_app_list:
                    active_label_list = app_label_dict[app]
                    for label in active_label_list:
                        label_id = label_index[label] + naive_nn_disp['active_label']
                        if label_id not in device_feature_dict[row[1]]:
                            device_feature_dict[row[1]][label_id] = 1
                        else:
                            device_feature_dict[row[1]][label_id] += 1

csvfile.close()

for device in device_feature_dict:
    if device in non_zero_event_count:
        device_feature_dict[device][longitude_idx] = device_feature_dict[device][longitude_idx] / \
                                                     non_zero_event_count[device]
        device_feature_dict[device][latitude_idx] = device_feature_dict[device][latitude_idx] / \
                                                     non_zero_event_count[device]

################################################
# release space from unused structure
del device_indexed_brand_model
del non_zero_event_count
del app_label_dict
del event_app_dict
del active_event_app_dict
del device_gender_dict
del gender_dict
del model_index
del brand_index


################################################
# generate train test list
print('generate train test list')
count = 0
A_train_list = []
B_train_list = []
train_device_list = []
with open(os.path.join(datadir,'gender_age_train.csv'), newline='', encoding='utf8') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',')
    for row in train_reader:
        if row[0] != 'device_id':
            train_device_list.append(row[0])
            if count % 2 == 0:
                A_train_list.append(row[0])
            else:
                B_train_list.append(row[0])
            count += 1
csvfile.close()

#######################################################
# generate tensorflow format input
print('generate tensorflow format input')

Dataset = collections.namedtuple('Dataset', ['data', 'target'])

A_train_label = []
A_train_feature = []
for device in A_train_list:
    A_train_label.append(device_group_dict[device])
    feature_list = [0]*naive_nn_disp['total']
    if device in device_feature_dict:
        for idx in device_feature_dict[device]:
            feature_list[idx] = device_feature_dict[device][idx]
        # release space
        del device_feature_dict[device]
    else:
        print('wow')
    A_train_feature.append(np.asarray(feature_list))
A_train_set = Dataset(data=np.array(A_train_feature), 
                        target=np.array(A_train_label).astype(np.int))

del A_train_label
del A_train_feature
del A_train_list

print('A train done')
 
B_train_label = []
B_train_feature = []
for device in B_train_list:
    B_train_label.append(device_group_dict[device])
    feature_list = [0]*naive_nn_disp['total']
    if device in device_feature_dict:
        for idx in device_feature_dict[device]:
            feature_list[idx] = device_feature_dict[device][idx]
        del device_feature_dict[device]
    else:
        print('wow')
    B_train_feature.append(np.asarray(feature_list))
B_train_set = Dataset(data=np.array(B_train_feature), 
                        target=np.array(B_train_label).astype(np.int))  

del B_train_label
del B_train_feature
del B_train_list

print('B train done')

# test_device_list = []
# with open(os.path.join(datadir,'gender_age_test.csv'), newline='', encoding="utf8") as csvfile:
#     test_reader = csv.reader(csvfile, delimiter=',')
#     for row in test_reader:
#         if row[0] != 'device_id':
#             test_device_list.append(row[0])
# csvfile.close()
                        
# test_device_label = []
# test_device_feature = []
# for device in test_device_list:
#     test_device_label.append(0)
#     feature_list = [0]*naive_nn_disp['total']
#     if device in device_feature_dict:
#         for idx in device_feature_dict[device]:
#             feature_list[idx] = device_feature_dict[device][idx]
#         del device_feature_dict[device]
#     else:
#          print('wow')
#     test_device_feature.append(np.asarray(feature_list))
# test_device_set = Dataset(data=np.array(test_device_feature), 
#                         target=np.array(test_device_label).astype(np.int)) 
                        
# del test_device_label
# del test_device_feature
# del test_device_list

# print('test done')

# train_device_label = []
# train_device_feature = []
# for device in train_device_list:
#     train_device_label.append(device_group_dict[device])
#     feature_list = [0]*naive_nn_disp['total']
#     for idx in device_feature_dict[device]:
#         feature_list[idx] = device_feature_dict[device][idx]
#     train_device_feature.append(np.asarray(feature_list))
# train_device_set = Dataset(data=np.array(train_device_feature), 
#                         target=np.array(train_device_label).astype(np.int))
# del train_device_label
# del train_device_feature
# del train_device_list                        

# release space 
del device_feature_dict
del device_group_dict

#####################################################
# start naive NN training using tensorflow
print('start naive NN training using tensorflow')

tf.logging.set_verbosity(tf.logging.INFO)                        
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    B_train_set.data,
    B_train_set.target,
    every_n_steps=50)
    
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=12)

classifier.fit(x=A_train_set.data,
               y=A_train_set.target,
               steps=100,
               monitors=[validation_monitor])
               
y = classifier.predict_proba(B_train_set.data)

print(y)

print('Done')



