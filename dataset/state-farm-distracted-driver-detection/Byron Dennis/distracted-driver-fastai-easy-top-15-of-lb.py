# Distracted Driver Predictions Using Fast AI
# Use the Fast.AI library to quickly create an image recognition model with performance in the top 25% of the private leaderboard.

# 1 - Import Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# import fast ai vision library
from fastai.vision import *

# 2 - Create Validation Set

# create validation set by removing drivers from the training set and putting their pictures in a validation set.
path = '/kaggle/input/state-farm-distracted-driver-detection/'
img_list = pd.read_csv(path + 'driver_imgs_list.csv')

# select a subset of the subjects for validation
valid_subjects = img_list.subject.sort_values().unique()[-4:]
# create new column identifying the subjects for validation
img_list['is_valid'] = img_list['subject'].isin(valid_subjects)

print("valid subjects: ", valid_subjects)
print(img_list[img_list['is_valid']==True].subject.count())

# create new column with class folder and image name combined
img_list['img_path'] = img_list.classname + '/' + img_list.img

valid_names = img_list[img_list['subject'].isin(valid_subjects)].img
valid_names = valid_names.to_list()

# 3 - Create Data Bunch

# apply standard image transformations except flipping the pictures.  
# The categories we are trying to predict can be specific to left hand / right hand

tfms = get_transforms(do_flip=False)

# create the data bunch

data = (ImageList.from_df(df=img_list, path = path + 'imgs/train/', cols='img_path' )
        #.split_by_valid_func(lambda o: os.path.basename(o) in valid_names)
        .split_by_rand_pct(.1)
        .label_from_df(1)
        .transform(tfms=tfms)
        .add_test_folder(path + 'imgs/test/')
        .databunch(bs=64))

# output description of data

data

# review images from a batch

data.show_batch(3)

# 4 - Define CNN Model w/ Transfer Learning

# Used resnet34 because of memory errors.  Would have liked to try resnet50 or VGG16.

learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/')

# For this step we are not going to train the resnet34 layers.  We are going to use the pretrained resnet34 weights and apply the standard model head defined by Fast AI.  We will train the model head.  Once that is complete we will unfreeze the resnet layers and train them.  
# Predefined FastAI layers for the model head are Adaptive Pooling (max & avg), BN -> Dropout -> Dense w/ Relu -> BN -> Dropout -> Dense (predictions). 
# You can review the layers of the model by calling "learn.layer_groups"

# fit the model
# this takes several minutes to run on the kaggle GPU ~12 mins per cycle.
# You can run additional cycles until the validation error stops improving.

learn.fit_one_cycle(6, max_lr=1e-02)

# If you use a random split the train / validtion error look similar because a person may appear in both sets.  Although the model is overfitting, the performance on the test set is still in the top 25%.
# When you create a validation sample that removes individual subjects completely from the training set the validation error is more closely aligned with the leaderboard.  However, you have to make sure to train on all of the data before submitting test predictions because the information from the validation subjects is valuable (subjects have different ethnicity, size, color, etc.).

# 5 - Unfreeze and train pretrained layers

learn.unfreeze()
learn.fit_one_cycle(6, max_lr=(1e-4, 1e-3, 1e-2))

# 6 - Review Errors

# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(6)

# 7 - Create Predictions for Test Set
# To get the best results you would want to retrain the model using all of the training data.

preds = learn.get_preds(DatasetType.Test)

labels = pd.DataFrame(data.test_ds.x.items, columns=['img'])
labels.img = labels.img.astype(str)
labels = labels.img.str.rsplit('/', 1, expand=True)
labels.drop(0, axis=1, inplace=True)
labels.rename(columns={1: 'img'}, inplace=True)

columns = data.classes

submission = pd.DataFrame(preds[0].numpy(), columns=columns, index=labels.img)
submission.reset_index(inplace=True)
submission.to_csv('submission.csv', index=False)