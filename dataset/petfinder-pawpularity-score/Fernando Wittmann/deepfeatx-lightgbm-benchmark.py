"""
In this code I will be using the code from a library that I created (deepfeatx - https://github.com/WittmannF/deepfeatx) 
for helping extracting features from images from pretrained models. Since internet is not available, I pasted the full
library here instead of installing from pip install deepfeatx. For training the model, I will be using the lightgbm
implementation available at: https://www.kaggle.com/rohanrao/ashrae-half-and-half
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from PIL import ImageFile
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True #https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, EfficientNetB7
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torchvision.transforms as transforms
import lightgbm as lgb


def train_lightgbm(X_train, y_train, categorical_features=[]):
    X_half_1 = X_train[:int(X_train.shape[0] / 2)]
    X_half_2 = X_train[int(X_train.shape[0] / 2):]

    y_half_1 = y_train[:int(X_train.shape[0] / 2)]
    y_half_2 = y_train[int(X_train.shape[0] / 2):]

    d_half_1 = lgb.Dataset(X_half_1, 
                           label=y_half_1, 
                           categorical_feature=categorical_features, 
                           free_raw_data=False)
    d_half_2 = lgb.Dataset(X_half_2, 
                           label=y_half_2, 
                           categorical_feature=categorical_features, 
                           free_raw_data=False)

    watchlist_1 = [d_half_1, d_half_2]
    watchlist_2 = [d_half_2, d_half_1]

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse"
    }

    print("Building model with first half and validating on second half:")
    model_half_1 = lgb.train(params, 
                             train_set=d_half_1, 
                             num_boost_round=1000, 
                             valid_sets=watchlist_1,
                             verbose_eval=200, 
                             early_stopping_rounds=200)

    print("Building model with second half and validating on first half:")
    model_half_2 = lgb.train(params, 
                             train_set=d_half_2, 
                             num_boost_round=1000, 
                             valid_sets=watchlist_2, 
                             verbose_eval=200, 
                             early_stopping_rounds=200)
    return model_half_1, model_half_2

class ImageFeatureExtractor():
    def __init__(self, 
                 model_name='resnet', 
                 weights='imagenet',
                 target_shape=(224, 224, 3)):
        self.target_shape = target_shape
        self.model = self._get_model(model_name,
                                     weights)
        self.model_name = model_name

    def _center_crop_img(self, img, size=224): #using pytorch as it gives more freedom in the transformations
        tr = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])
        return tr(img)

    def _preprocess_img(self, img):
        img=self._center_crop_img(img, size=self.target_shape[0])

        # Convert to a Numpy array
        img_np = np.asarray(img)

        # Reshape by adding 1 in the beginning to be compatible as input of the model
        img_np = img_np[None] # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis

        # Prepare the image for the model
        img_np = self.preprocess_input(img_np)

        return img_np

    def _get_model(self, model_name, weights):
        if model_name=='resnet':
            self.preprocess_input = resnet50.preprocess_input

            base_model = resnet50.ResNet50(include_top=False,
                                           input_shape=self.target_shape,
                                           weights=weights
                                          )

            for layer in base_model.layers:
                layer.trainable=False

            model = Sequential([base_model,
                                GlobalAveragePooling2D()])

            return model

        elif model_name=='efficientnetb0':
            self.preprocess_input = efficientnet.preprocess_input
            base_model = EfficientNetB0(include_top=False,
                                        input_shape=self.target_shape,
                                        weights=weights
                                       )

            for layer in base_model.layers:
                layer.trainable=False

            model = Sequential([base_model,
                                GlobalAveragePooling2D()])

            return model

        elif model_name=='efficientnetb4':
            self.preprocess_input = efficientnet.preprocess_input
            base_model = EfficientNetB4(include_top=False,
                                        input_shape=self.target_shape,
                                        weights=weights
                                       )

            for layer in base_model.layers:
                layer.trainable=False

            model = Sequential([base_model,
                                GlobalAveragePooling2D()])

            return model

        elif model_name=='efficientnetb7':
            self.preprocess_input = efficientnet.preprocess_input
            base_model = EfficientNetB7(include_top=False,
                                        input_shape=self.target_shape,
                                        weights=weights
                                       )

            for layer in base_model.layers:
                layer.trainable=False

            model = Sequential([base_model,
                                GlobalAveragePooling2D()])

            return model


        return None

    def _get_img_gen_from_df(self, dataframe, batch_size=32):

        datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)

        gen = datagen.flow_from_dataframe(dataframe,
                                          batch_size=batch_size,
                                          target_size=self.target_shape[:2],
                                          class_mode=None,
                                          shuffle=False)
        return gen

    def _get_img_gen(self, folder_path, batch_size=32):
        datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
        gen = datagen.flow_from_directory(folder_path,
                                          batch_size=batch_size,
                                          target_size=self.target_shape[:2],
                                          class_mode='sparse',
                                          shuffle=False)
        return gen

    def _assert_df_size(self, dataframe):
        assert len(dataframe)>0, "Folder not found or does not have images. If there's one folder per class, please make sure to set classes_as_folders to True"

    def read_img_url(self, url, center_crop=True):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        if center_crop:
            img = self._center_crop_img(img, size=self.target_shape[0])
        return img

    def read_img_path(self, img_path, center_crop=True):
        img = image.load_img(img_path)
        if center_crop:
            img = self._center_crop_img(img, size=self.target_shape[0])
        return img

    def url_to_vector(self, url):
        img = self.read_img_url(url)
        vector = self.img_to_vector(img)
        return vector

    def img_path_to_vector(self, img_path):
        img = self.read_img_path(img_path)
        vector = self.img_to_vector(img)
        return vector

    def img_to_vector(self, img):
        img_np = self._preprocess_img(img)
        vector = self.model.predict(img_np)
        return vector

    def _get_gen(self, classes_as_folders, directory, batch_size):
        if classes_as_folders:
            gen = self._get_img_gen(directory, batch_size)
        else:
            filepaths = glob.glob(directory+'/*.*')
            self.dataframe=pd.DataFrame(filepaths,
                                        columns=['filename'])
            self._assert_df_size(self.dataframe)
            gen = self._get_img_gen_from_df(self.dataframe,
                                            batch_size)
        return gen

    def _vectors_to_df(self, all_vectors, classes_as_folders, export_class_names):
        vectors_df=pd.DataFrame(all_vectors)
        vectors_df.insert(loc=0, column='filepaths', value=self.gen.filepaths)
        if classes_as_folders and export_class_names:
            vectors_df.insert(loc=1, column='classes', value=self.gen.classes)
            id_to_class = {v: k for k, v in self.gen.class_indices.items()}
            vectors_df.classes=vectors_df.classes.apply(lambda x: id_to_class[x])
        return vectors_df

    def extract_features_from_directory(self,
                                        directory,
                                        batch_size=32,
                                        classes_as_folders=True,
                                        export_class_names=False,
                                        export_vectors_as_df=True):
        # Get image generator
        self.gen = self._get_gen(classes_as_folders, directory, batch_size)

        # Extract features into vectors
        self.all_vectors=self.model.predict(self.gen, verbose=1)

        # Either return vectors or everything as dataframes
        if not export_vectors_as_df:
            return self.all_vectors
        else:
            vectors_df = self._vectors_to_df(self.all_vectors, classes_as_folders, export_class_names)
            return vectors_df

    def vectors_from_folder_list(self, folder_list):
        df_list = []
        for folder_path in folder_list:
            df=self.img_folder_to_vectors(folder_path)
            df_list.append(df)
        return pd.concat(df_list)
    
    
if __name__ == '__main__':    
    # Initialize Image Feature Extractor
    fe=ImageFeatureExtractor('efficientnetb7', 
                             target_shape=(600, 600, 3),
                             weights='../input/tfkerasefficientnetimagenetnotop/efficientnetb7_notop.h5'
                            )
    
    # Prepare Training Data
    train_features = fe.extract_features_from_directory('../input/petfinder-pawpularity-score/train',
                                                        classes_as_folders=False)
    train_features['filepaths']=train_features.filepaths.apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
    train = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
    train_data = train.join(train_features.set_index('filepaths'), on='Id')
    X = train_data.drop(['Id', 'Pawpularity'], axis=1)
    y = train_data.Pawpularity
    
    # Train Model 
    model_half_1, model_half_2 = train_lightgbm(X, y)

    cv_score=(model_half_1.best_score['valid_1']['rmse']+
              model_half_2.best_score['valid_1']['rmse'])/2
    print('CV Score:', cv_score)
    
    # Prepare test data
    test_features = fe.extract_features_from_directory('../input/petfinder-pawpularity-score/test',
                                                        classes_as_folders=False)
    test_features['filepaths']=test_features.filepaths.apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
    test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')
    test_data = test.join(test_features.set_index('filepaths'), on='Id')
    X_test = test_data.drop(['Id'], axis=1)

    # Predict and create submission
    y_pred = model_half_1.predict(X_test).clip(0, 100)/2
    y_pred += model_half_2.predict(X_test).clip(0, 100)/2
    submission = pd.DataFrame({
        'Id' : test_data.Id,
        'Pawpularity' : y_pred
    })
    submission.to_csv('submission.csv', index=False)
