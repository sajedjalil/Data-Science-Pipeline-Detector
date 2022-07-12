# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import scipy as sp
from functools import partial
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

import json
from textblob import TextBlob

from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
import math

import category_encoders

#Predict images as dogs or cats, then use only the correctly classified images
img_width, img_height = 150, 150
batch_size = 100

#To load fine_tune weights (top layers different than full_model, and fine tune has frozen layers
# build the VGG16 network
base_model = applications.VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,  input_shape = (img_width,img_height,3))
for layer in base_model.layers[:15]:
    layer.trainable = False
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(base_model)
model.add(top_model)
model.load_weights('../input/vgg-fine-tunig-dogsvscats/fineTune_model.h5')

#Predict training and test images as Dogs or Cats
path = "../input/petfinder-adoption-prediction/"
whichData='train'
train = pd.read_csv(os.path.join(path,whichData+'/'+whichData+'.csv'))
train_pet_ids_labels = train[['PetID','Type']] #1=dog, 2=cat
    
whichData='test'
test = pd.read_csv(os.path.join(path,whichData+'/'+whichData+'.csv'))
test_pet_ids_labels = test[['PetID','Type']] #1=dog, 2=cat

#Imaging Data
train_dir = os.path.join(path,'train_images')
test_dir = os.path.join(path,'test_images')

#Function to read a process the images to the format required for the model
def read_and_process_image(list_of_images,pet_ids_labels):
    # x = [] #resized images
    y = [] #labels

    for im in list_of_images:
        # x.append(cv2.cvtColor(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(img_width,img_height),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB) #Read the image
        # x.append(image.img_to_array(image.load_img(im, target_size=(img_width,img_height))))
        _, file = os.path.split(im)
        pet_id = file.split('-')[0]
        label = pet_ids_labels.iloc[list(np.where(pet_ids_labels["PetID"] == pet_id)[0])]['Type'].values
        #get the labels
        if label == 2 :
            y.append(0) #cat
        elif label == 1:
            y.append(1) #dog
    return y

# predicting images
train_datagen = image.ImageDataGenerator(rescale=1. / 255)
test_datagen = image.ImageDataGenerator(rescale=1. / 255)
##
import shutil

original_train_dir = os.path.join(path,'train_images')
original_test_dir = os.path.join(path,'test_images')
train_images = ['{}'.format(i) for i in os.listdir(original_train_dir)] 
test_images = ['{}'.format(i) for i in os.listdir(original_test_dir)] 


os.mkdir('train_images')
os.mkdir('train_images/train_images')
os.mkdir('test_images')
os.mkdir('test_images/test_images')

for file in train_images:
    shutil.copyfile(os.path.join(original_train_dir, file), os.path.join('train_images/train_images', file))
    
for file in test_images:
    shutil.copyfile(os.path.join(original_test_dir, file), os.path.join('test_images/test_images', file))
##

i = 0
train_pred = []
train_prob = []
genTrain = train_datagen.flow_from_directory('train_images', target_size=(img_height, img_width), batch_size=batch_size, shuffle=False, class_mode=None)
train_images = genTrain.filenames
y_train = read_and_process_image(train_images, train_pet_ids_labels)
for batch in genTrain:
    pred = model.predict_classes(batch)
    prob = model.predict(batch)
    train_pred.extend(pred)
    train_prob.extend(prob)
    i+=1
    if i==math.ceil(len(train_images)/batch_size):
        break

i = 0
test_pred = []
test_prob = []
genTest = test_datagen.flow_from_directory('test_images', target_size=(img_height, img_width), batch_size=batch_size, shuffle=False, class_mode=None)
test_images = genTest.filenames
y_test = read_and_process_image(test_images, test_pet_ids_labels)
for batch in genTest: 
    pred = model.predict_classes(batch)
    prob = model.predict(batch)
    test_pred.extend(pred)
    test_prob.extend(prob)
    i += 1
    if i == math.ceil(len(test_images) / batch_size):
        break

train_preds = [item[0] for item in train_pred]
test_preds = [item[0] for item in test_pred]
from sklearn.metrics import accuracy_score
print('Accuracy in train set =', accuracy_score(y_train,train_preds))
print('Accuracy in test set =', accuracy_score(y_test,test_preds))

train_imageIDs = []
for im in train_images:
    _, file = os.path.split(im)
    train_imageIDs.append(file.split('.')[0])
probs = [item[0] for item in train_prob]
train_probs = pd.DataFrame({'ImageID': train_imageIDs,
                         'Label': y_train,
                         'Pred': train_preds,
                         'Prob': probs})

test_imageIDs = []
for im in test_images:
    _, file = os.path.split(im)
    test_imageIDs.append(file.split('.')[0])
probs = [item[0] for item in test_prob]
test_probs = pd.DataFrame({'ImageID': test_imageIDs,
                        'Label': y_test,
                        'Pred': test_preds,
                        'Prob': probs})

# Any results you write to the current directory are saved as output.
def preprocess (whichData, sent=True, meta=True):
    path = "../input/petfinder-adoption-prediction/"
    colors = pd.read_csv(os.path.join(path,'color_labels.csv'))
    breeds = pd.read_csv(os.path.join(path,'breed_labels.csv'))
    if whichData != 'train' and whichData != 'test':
        return []
    dataset = pd.read_csv(os.path.join(path,whichData+'/'+whichData+'.csv'))
    if whichData == 'train':
        dataset_probs = train_probs #Label 0 dog, label 1 cat.
    elif whichData =='test':
        dataset_probs = test_probs #Label 0 dog, label 1 cat.
    sentiment_dir = os.path.join(path,whichData+'_sentiment')
    metadata_dir = os.path.join(path,whichData+'_metadata')
    '''
        #Names:
        - Names are in fact, in some cases, a short description of the pet.
            Those profiles with information in the name feature may have a faster adoption speed.
        - Names with Puppy, Kitty or any aditional description can help
        - Sentiment analysis may help. I presume that those with negatives name description will adopt faster.
    '''

    cat_cols = ['Type','Name','Breed1','Gender','Color1','Color2','Color3','MaturitySize','FurLength',
                'Vaccinated','Dewormed','Sterilized','Health','State', 'RescuerID','Description','PetID']
    #Names
    dataset.insert(dataset.columns.get_loc('Name')+1, 'Name_length', 0)  # Add feature Name_length with default value 0
    dataset['Name_length'] = np.where(dataset['Name'].str.split().str.len() > 0, dataset['Name'].str.split().str.len(), 0) #Add Name_length (in words)
    dataset['Name'].fillna('unnamed', inplace=True)  # Fill NaN with unnamed

    #Sentiment on Name
    dataset.insert(dataset.columns.get_loc('Name_length') + 1, 'NameSent',0)  # Add feature Name_length with default value 0
    dataset['NameSent'] = dataset['Name'].astype(str).apply(lambda x: TextBlob(x).sentiment[0])

    dataset[['Name', 'NameSent']].head()
    # Assign dummy values to Name -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
    dataset['Name'] = np.select([dataset['Name'].str.contains('puppy', case=False),
                                 dataset['Name'].str.contains('dog', case=False),
                                 dataset['Name'].str.contains('kitty', case=False),
                                 dataset['Name'].str.contains('cat', case=False),
                                dataset['Name'].str.contains('no name|no-name|not name|yet|unnamed',case=False)],
                                [1, 2, 3, 4, -1],0)

    #Description
    dataset.insert(dataset.columns.get_loc('Description')+1, 'Description_length', 0)  # Add feature Description_length with default value 0
    dataset['Description_length'] = np.where(dataset['Description'].str.split().str.len() > 0, dataset['Description'].str.split().str.len(), 0) #Add Description_length (in words)

    dataset.insert(dataset.columns.get_loc('Description_length') + 1, 'LexicalDensity',0)  # Add feature Name_length with default value 0
    dataset['LexicalDensity'] = dataset['Description'].str.lower().str.split().apply(lambda x: np.unique(x)).str.len() / dataset['Description_length']
    dataset['LexicalDensity'].replace([np.inf, -np.inf],0, inplace=True)

    dataset['Description'].fillna('nothing', inplace=True)  # Fill NaN with nothing

    # Assign dummy values to Description -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
    dataset['Description'] = np.select([dataset['Description'].str.contains('puppy', case=False),
                                        dataset['Description'].str.contains('dog', case=False),
                                        dataset['Description'].str.contains('kitty', case=False),
                                        dataset['Description'].str.contains('cat', case=False),
                                        dataset['Description'].str.contains('nothing', case=False)],
                                       [1, 2, 3, 4, -1], 0)
    # Breed2 = 0 if Breed1 == Breed2
    dataset['Breed2'] = np.where((dataset['Breed1'] == dataset['Breed2']), 0, dataset['Breed2'])
    # Breed1 = Breed2 and set Breed2 = 0 if Breed1 = 0 and Breed2 != 0
    zeroBreed1 = (dataset['Breed1'] == 0)
    dataset.loc[zeroBreed1, 'Breed1'] = dataset['Breed2']
    dataset.loc[zeroBreed1, 'Breed2'] = 0
    #If Breed1 and Breed 2 change Breed1 for 307 (Mixed Breed)
    dataset.loc[dataset['Breed2'] != 0,'Breed1'] = 307
    #Drop Breed2
    dataset.drop('Breed2', axis=1, inplace=True)
    #Set incorrect breed types to 307
    datasetBreed = dataset.merge(breeds, how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_br'))
    dataset.loc[datasetBreed.Type != datasetBreed.Type_br,'Breed1'] = 307
    del datasetBreed

    # RescuerID doesn't help because there is no overlap between training and testing

    dataset_id = dataset['PetID']
    # Add sentiment data
    if sent:
        doc_sent_mag = []
        doc_sent_score = []
        for pet in dataset_id:
            try:
                with open(os.path.join(sentiment_dir, pet + '.json'), 'r', encoding="utf8") as f:
                    sentiment = json.load(f)
                doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
                doc_sent_score.append(sentiment['documentSentiment']['score'])
            except FileNotFoundError:
                doc_sent_mag.append(-1)
                doc_sent_score.append(-1)

        dataset.loc[:, 'doc_sent_mag'] = doc_sent_mag
        dataset.loc[:, 'doc_sent_score'] = doc_sent_score

    # Add image metadata
    if meta:
        dataset.loc[:,'PhotoAmtGood'] = 0
        image_id = dataset_probs['ImageID']
        im_cnt = 0
        for n,im in enumerate(image_id):
            if dataset_probs.loc[n,'Label']==0:
                dataset_probs.loc[n,'Prob'] = 1-float(dataset_probs.loc[n,'Prob'])
            if dataset_probs.loc[n,'Label']==dataset_probs.loc[n,'Pred'] and dataset_probs.loc[n,'Prob'] >= 0.99:
                im_cnt+=1
                with open(os.path.join(metadata_dir, im + '.json'), 'r', encoding="utf8") as f:
                    data = json.load(f)
                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
                dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
                dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
                dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
                dataset.loc[dataset['PetID'] == im.split('-')[0], 'PhotoAmtGood'] += 1
                if data.get('labelAnnotations'):
                    label_description = data['labelAnnotations'][0]['description']
                    label_score = data['labelAnnotations'][0]['score']
                else:
                    label_description = 'nothing'
                    label_score = -1
            else:
                vertex_x = -1
                vertex_y = -1
                bounding_confidence = -1
                bounding_importance_frac = -1
                dominant_blue = -1
                dominant_green = -1
                dominant_red = -1
                dominant_pixel_frac = -1
                dominant_score = -1
                label_description = 'nothing'
                label_score = -1
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'vertex_x-' + im.split('-')[1]] = vertex_x
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'vertex_y''vertex_y-' + im.split('-')[1]] = vertex_y
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'bounding_confidence-' + im.split('-')[1]] = bounding_confidence
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'bounding_importance-' + im.split('-')[1]] = bounding_importance_frac
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_blue-' + im.split('-')[1]] = dominant_blue
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_green-' + im.split('-')[1]] = dominant_green
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_red-' + im.split('-')[1]] = dominant_red
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_pixel_frac-' + im.split('-')[1]] = dominant_pixel_frac
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_score-' + im.split('-')[1]] = dominant_score
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'label_description-' + im.split('-')[1]] = label_description
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'label_score-' + im.split('-')[1]] = label_score
        dataset['Name'].fillna('unnamed', inplace=True)  # Fill NaN with unnamed
        for col,descN in enumerate(pd.Series(list(dataset)).str.contains('label_description')):
            if descN:
                dataset.iloc[:, col].fillna('nothing', inplace=True) #Fill Na label_descriptions with 'nothing'
                # Assign dummy values to label_description -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
                dataset.iloc[:,col]= np.select([dataset.iloc[:, col].str.contains('puppy', case=False),
                                             dataset.iloc[:, col].str.contains('dog', case=False),
                                             dataset.iloc[:, col].str.contains('kitty', case=False),
                                             dataset.iloc[:, col].str.contains('cat', case=False),
                                             dataset.iloc[:, col].str.contains('nothing', case=False)],
                                            [1, 2, 3, 4, -1], 0)
                cat_cols.append(list(dataset)[col])
            else:
                dataset.iloc[:, col].fillna(-1, inplace=True)  # Fill other Na metadata with -1
        print("Number of correctly classified images for " + whichData + ": %s" % im_cnt)
        dataset.loc[:,'PhotoAmtFrac'] = dataset['PhotoAmtGood']/dataset['PhotoAmt']
        dataset['PhotoAmtFrac'].fillna(-1, inplace=True) #These are pets with no photos PhotoAmt=0

    dataset[cat_cols] = dataset[cat_cols].apply(lambda x: x.astype('category'))
    # Breeds, check if they have only Breed1 and is mixed or mixed is written in name or description
    # FurLength, check if it is written in name or description. If yes, correct the FurLength accordingly
    return dataset
    
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)    

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

#Preprocess data
train_full=preprocess('train')
test_full=preprocess('test')

#We drop 'RescuerID' as there is not overlap between train and test
#We drop 'PhotoAmt' as we already have 'PhotoAmtGood' and 'PhotoAmtFrac=PhotoAmtGood/PhotoAmt'
toDrop = ['PetID','RescuerID','PhotoAmt']
train_full.drop(toDrop, axis=1, inplace=True)

cat_cols = ['Type', 'Name','Breed1', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
            'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Description'] + \
           list(train_full.columns[train_full.columns.str.contains('label_description')])
train_full[cat_cols] = train_full[cat_cols].apply(lambda x: x.astype('category'))
test_full[cat_cols] = test_full[cat_cols].apply(lambda x: x.astype('category'))

#Features to keep
# features1 = ['Type','Age','Breed1','Gender','Quantity', 'Description_length',
#              'doc_sent_mag', 'doc_sent_score', 'PhotoAmtGood','PhotoAmtFrac','AdoptionSpeed']
features1 = ['Type','Name','Name_length','NameSent','Age','Breed1','Gender','Color1','Color2','Color3','MaturitySize',
             'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
             'VideoAmt', 'Description','Description_length','LexicalDensity',
             'doc_sent_mag', 'doc_sent_score', 'PhotoAmtGood','PhotoAmtFrac','AdoptionSpeed']
features2 = features1 + ['vertex_x-1', 'vertex_yvertex_y-1', 'bounding_confidence-1',
        'bounding_importance-1', 'dominant_blue-1', 'dominant_green-1',
        'dominant_red-1', 'dominant_pixel_frac-1', 'dominant_score-1',
        'label_description-1', 'label_score-1']
features3 = features2 + ['vertex_x-2', 'vertex_yvertex_y-2', 'bounding_confidence-2',
        'bounding_importance-2', 'dominant_blue-2', 'dominant_green-2',
        'dominant_red-2', 'dominant_pixel_frac-2', 'dominant_score-2',
        'label_description-2', 'label_score-2']
features4 = features3 + ['vertex_x-3', 'vertex_yvertex_y-3', 'bounding_confidence-3',
        'bounding_importance-3', 'dominant_blue-3', 'dominant_green-3',
        'dominant_red-3', 'dominant_pixel_frac-3', 'dominant_score-3',
        'label_description-3', 'label_score-3']
features5 = features4 + ['vertex_x-4', 'vertex_yvertex_y-4', 'bounding_confidence-4',
        'bounding_importance-4', 'dominant_blue-4', 'dominant_green-4',
        'dominant_red-4', 'dominant_pixel_frac-4', 'dominant_score-4',
        'label_description-4', 'label_score-4']
features6 = features5 + ['vertex_x-5', 'vertex_yvertex_y-5', 'bounding_confidence-5',
        'bounding_importance-5', 'dominant_blue-5', 'dominant_green-5',
        'dominant_red-5', 'dominant_pixel_frac-5', 'dominant_score-5',
        'label_description-5', 'label_score-5']
features7 = train_full.columns.values
features = []
features.append(features1)
features.append(features2)
features.append(features3)
features.append(features4)
features.append(features5)
features.append(features6)
features.append(features7)

################################
#LGb with best params
params = {#'application': 'multiclass',
          'application': 'regression',
          #'num_class': 5,
          'boosting': 'gbdt',
          #'metric': 'multi_logloss',
          'metric': 'rmse',
          'num_leaves': 70,
          #'max_depth': 9,
          'max_depth': 25,
          #'learning_rate': 0.01,
          'learning_rate': 0.005,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.02,
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'data_random_seed': 17}
# Additional parameters:
early_stop = 500
verbose_eval = 1000
num_rounds = 10000
n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, random_state=11, shuffle=True)

results = []
n = 1
for feature in features:
    train=train_full.loc[:,feature]

    qwk_scores = []
    feature_importance_df = pd.DataFrame()
    i=1
    for train_index, valid_index in kfold.split(train, train['AdoptionSpeed'].values):
#######################################################################################################
             #Profiles with Good images
            X_tr = train.iloc[train_index, :]
            X_tr = X_tr[X_tr.PhotoAmtGood != 0].reset_index(drop=True)  # We keep rows with Good images

            X_val = train.iloc[valid_index, :]
            X_val = X_val[X_val.PhotoAmtGood != 0].reset_index(drop=True)  # We keep rows with Good images

            y_tr = X_tr['AdoptionSpeed'].values
            X_tr.drop('AdoptionSpeed', axis=1,inplace=True)

            y_val1 = X_val['AdoptionSpeed'].values
            X_val.drop('AdoptionSpeed', axis=1,inplace=True)

            # get the categorical features
            cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
            te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
            te = te.fit(X_tr,y_tr)
            X_tr = te.transform(X_tr)
            X_val = te.transform(X_val)

            d_train = lgb.Dataset(X_tr, label=y_tr)
            d_valid = lgb.Dataset(X_val, label=y_val1)
            watchlist = [d_train,d_valid]


            model = lgb.train(params,
                              train_set=d_train,
                              num_boost_round=num_rounds,
                              valid_sets=watchlist,
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stop)

            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            tr_pred = model.predict(X_tr, num_iteration=model.best_iteration)
            # val_predictions = []
            # for x in val_pred:
            #     val_predictions.append(np.argmax(x))

            optR = OptimizedRounder()
            optR.fit(tr_pred, y_tr)
            coefficients = optR.coefficients()
            val_predictions1 = optR.predict(val_pred, coefficients)

            #Profiles without Good images
            X_tr = train.iloc[train_index, :]
            X_tr = X_tr[X_tr.PhotoAmtGood == 0].reset_index(drop=True)  # We keep rows with no Good images

            X_val = train.iloc[valid_index, :]
            X_val = X_val[X_val.PhotoAmtGood == 0].reset_index(drop=True)  # We keep rows with no Good images

            y_tr = X_tr['AdoptionSpeed'].values
            X_tr.drop('AdoptionSpeed', axis=1,inplace=True)

            y_val2 = X_val['AdoptionSpeed'].values
            X_val.drop('AdoptionSpeed', axis=1,inplace=True)

            # get the categorical features
            cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
            te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
            X_tr = te.fit_transform(X_tr,y_tr)
            X_val = te.transform(X_val)

            d_train = lgb.Dataset(X_tr, label=y_tr)
            d_valid = lgb.Dataset(X_val, label=y_val2)
            watchlist = [d_train,d_valid]

            # get the categorical features
            cat_feature_names = X_tr.dtypes[X_tr.dtypes == "category"]
            cat_features = [X_tr.columns.get_loc(c) for c in X_tr.columns if c in cat_feature_names]
            model = lgb.train(params,
                              train_set=d_train,
                              num_boost_round=num_rounds,
                              valid_sets=watchlist,
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stop)

            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            tr_pred = model.predict(X_tr, num_iteration=model.best_iteration)
            # val_predictions = []
            # for x in val_pred:
            #     val_predictions.append(np.argmax(x))

            optR = OptimizedRounder()
            optR.fit(tr_pred, y_tr)
            coefficients = optR.coefficients()
            val_predictions2 = optR.predict(val_pred, coefficients)
#########OVERALL RESULTS###########
            qwk = quadratic_weighted_kappa(np.concatenate((y_val1,y_val2),axis=0),
                                              np.concatenate((val_predictions1, val_predictions2),axis=0))
            qwk_scores.append(qwk)
            # importances = model.feature_importance()
            # fold_importance_df = pd.DataFrame()
            # fold_importance_df['feature'] = train.drop(['AdoptionSpeed'],axis=1).columns.values
            # fold_importance_df['importance'] = importances
            # fold_importance_df['fold'] = i
            # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            i+=1

    print('Results for LGB classification - model',n)
    print('QWK scores : {}'.format(qwk_scores))
    print('mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('std QWK score : {}'.format(np.std(qwk_scores)))
    results.append(np.std(qwk_scores))
    # print(feature_importance_df.groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False).head(50))
    n+=1

# Predict test set
X_tr = train_full.loc[:, features[results.index(min(results))]]
X_tr = X_tr[X_tr.PhotoAmtGood != 0].reset_index(drop=True)

y_tr = X_tr['AdoptionSpeed'].values
X_tr.drop('AdoptionSpeed', axis=1, inplace=True)

X_test = test_full.loc[:, X_tr.columns.values]
X_test = X_test[X_test.PhotoAmtGood != 0].reset_index(drop=True)
petIDs = test_full.loc[test_full.PhotoAmtGood != 0,'PetID']
# get the categorical features
cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
X_tr = te.fit_transform(X_tr,y_tr)
X_test = te.transform(X_test)

d_train = lgb.Dataset(X_tr, label=y_tr)

model = lgb.train(params, train_set=d_train,
                  num_boost_round=num_rounds,
                  verbose_eval=verbose_eval)

test_pred = model.predict(X_test, num_iteration=model.best_iteration)
train_pred = model.predict(X_tr, num_iteration=model.best_iteration)

optR = OptimizedRounder()
optR.fit(train_pred, y_tr)
coefficients = optR.coefficients()

submission1 = pd.DataFrame()
submission1.loc[:,'PetID']=petIDs
submission1.loc[:,'AdoptionSpeed'] = optR.predict(test_pred, coefficients).astype(int)

X_tr = train_full.loc[:, features[0]] #We don't use metadata here
X_tr = X_tr[X_tr.PhotoAmtGood == 0].reset_index(drop=True)

y_tr = X_tr['AdoptionSpeed'].values
X_tr.drop('AdoptionSpeed', axis=1, inplace=True)

X_test = test_full.loc[:, X_tr.columns.values]
X_test = X_test[X_test.PhotoAmtGood == 0].reset_index(drop=True)
petIDs = test_full.loc[test_full.PhotoAmtGood == 0,'PetID']
# get the categorical features
cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
X_tr = te.fit_transform(X_tr,y_tr)
X_test = te.transform(X_test)


d_train = lgb.Dataset(X_tr, label=y_tr)

model = lgb.train(params, train_set=d_train,
                  num_boost_round=num_rounds,
                  verbose_eval=verbose_eval)

test_pred = model.predict(X_test, num_iteration=model.best_iteration)
train_pred = model.predict(X_tr, num_iteration=model.best_iteration)

optR = OptimizedRounder()
optR.fit(train_pred, y_tr)
coefficients = optR.coefficients()

submission2 = pd.DataFrame()
submission2.loc[:,'PetID']=petIDs
submission2.loc[:,'AdoptionSpeed'] = optR.predict(test_pred, coefficients).astype(int)

submission=pd.concat([submission1,submission2])
submission.to_csv("submission.csv", index=False)

if os.path.isdir('train_images'):
    shutil.rmtree('train_images')
if os.path.isdir('test_images'):
    shutil.rmtree('test_images')