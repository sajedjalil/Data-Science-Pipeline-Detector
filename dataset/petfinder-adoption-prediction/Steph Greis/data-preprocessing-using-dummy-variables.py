# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

# data contains the following columns: 
# 'Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 
# 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt', 
# 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed'

# of these Name and Description have NaNs (the rest doesn't)
# the following fields are going to be ignored (for now):
# Name, RescuerID, Description, PetID, State
breed = pd.read_csv('../input/breed_labels.csv')
col = pd.read_csv('../input/color_labels.csv')

def dataprep(data_file_path, col, breed):
    """
    function to read in data files for PetFinder data, and output X and y (with dummy variables)
    """
    df = pd.read_csv(data_file_path)
    df = df.drop(['Name', 'RescuerID', 'Description', 'State'], axis=1)
    df.Type = df.Type.replace({1: 'dog', 2: 'cat'})
    df.Gender = df.Gender.replace({1:'male', 2:'female', 3:'mixed'})
    df.MaturitySize = df.MaturitySize.replace({1:'S', 2:'M', 3:'L', 4:'XL', 0:'unsure'})
    df.FurLength = df.FurLength.replace({1:'S', 2:'M', 3:'L', 0:'unsure'})
    df.Vaccinated = df.Vaccinated.replace({1:'y', 2:'n', 3:'unsure'})
    df.Dewormed = df.Dewormed.replace({1:'y', 2:'n', 3:'unsure'})
    df.Sterilized = df.Sterilized.replace({1:'y', 2:'n', 3:'unsure'})
    df.Health = df.Health.replace({1:'healthy', 2: 'MinorInjury', 3:'SeriousInjury', 0: 'unsure'})
    
    df1 = df.merge(col, left_on='Color1', right_on='ColorID')
    df1 = df1.rename(columns={'ColorName': 'ColName1'})
    df1 = df1.drop(['Color1', 'ColorID'], axis=1)

    df2 = df1.merge(col, left_on='Color2', right_on='ColorID')
    df2 = df2.rename(columns={'ColorName': 'ColName2'})
    df2 = df2.drop(['Color2', 'ColorID'], axis=1)

    df3 = df2.merge(col, left_on='Color3', right_on='ColorID')
    df3 = df3.rename(columns={'ColorName': 'ColName3'})
    df3 = df3.drop(['Color3', 'ColorID'], axis=1)

    df4 = df3.merge(breed, left_on='Breed1', right_on='BreedID')
    df4 = df4.rename(columns={'BreedName': 'BreedName1'})
    df4 = df4.drop(['Breed1', 'BreedID'], axis=1)

    df5 = df4.merge(breed, left_on='Breed2', right_on='BreedID')
    df5 = df5.rename(columns={'BreedName': 'BreedName2'})
    df5 = df5.drop(['Breed2', 'BreedID'], axis=1)
    del df1, df2, df3, df4
    
    df5 = df5.set_index('PetID')

    # set dummy variables for everything BUT: Age, Quantity, Fee, VideoAmt, PhotoAmt, AdoptionSpeed 
    df_final = pd.get_dummies(df5)

    if 'AdoptionSpeed' in df_final.columns:
        y = df_final['AdoptionSpeed']
        X = df_final.drop('AdoptionSpeed', axis=1)
    else:
        X = df_final
        y = None
    return X, y
    
X_train, y_train = dataprep('../input/train/train.csv', col, breed)
X_test, y_test = dataprep('../input/test/test.csv', col, breed)

X_train.to_pickle('X_train.pkl')
y_train.to_pickle('y_train.pkl')
X_test.to_pickle('X_test.pkl')