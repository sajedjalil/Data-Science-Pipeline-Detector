# Packages 
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


# Any results you write to the current directory are saved as output.
traindf = pd.read_csv('../input/train.csv')
testdf = pd.read_csv('../input/test.csv')


#Make X and y for training
X = traindf.drop(columns=['Id','Cover_Type'])
y = traindf['Cover_Type']
Xtest = testdf.drop(columns=['Id'])

#Changing soil_type and wilderness_area one-hot variables into categoricals
#from skillsmuggler on Kaggle

def wilderness_feature(df):
    df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].sum(axis=1)
    return df

def soil_features(df):
    soil_types = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', \
                  'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', \
                  'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', \
                  'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', \
                  'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    df[soil_types] = df[soil_types].multiply([i for i in range(1, 41)], axis=1)
    df['soil_type'] = df[soil_types].sum(axis=1)
    return df

##Convert to categoricals
X=wilderness_feature(X)
X=soil_features(X)
X['Wilderness_Area']=X['Wilderness_Area'].astype('str')
X['soil_type']=X['soil_type'].astype('str')
##and remove onehots
X=X.drop(columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3','Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8','Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12','Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16','Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20','Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24','Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28','Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32','Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36','Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])

##Same for test
Xtest=wilderness_feature(Xtest)
Xtest=soil_features(Xtest)
Xtest['Wilderness_Area']=Xtest['Wilderness_Area'].astype('str')
Xtest['soil_type']=Xtest['soil_type'].astype('str')
Xtest=Xtest.drop(columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3','Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8','Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12','Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16','Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20','Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24','Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28','Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32','Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36','Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])

##Initial model
extree=ExtraTreesClassifier(n_estimators=300)
extree.fit(X,y)

##Remove less important features
X=X[list(X.columns[extree.feature_importances_ > 0.05])]
Xtest=Xtest[list(Xtest.columns[extree.feature_importances_ > 0.05])]
##Fit again
extree.fit(X,y)

#Predict
preds=extree.predict(Xtest)

#Submission
pd.DataFrame(list(zip(testdf['Id'],preds)),columns=["Id","Cover_Type"]).to_csv("pred1.csv",index=False)