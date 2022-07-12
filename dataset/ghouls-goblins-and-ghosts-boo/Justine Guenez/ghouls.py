# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


from matplotlib import pyplot as plt
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/forest-cover-type-kernels-only/train.csv.zip")
test = pd.read_csv("../input/forest-cover-type-kernels-only/test.csv.zip")

submission = pd.read_csv("../input/forest-cover-type-kernels-only/sample_submission.csv.zip")
submission["Cover_Type"] = "Unknown"


#Changer chiffres en noms pour Cover_Type
def forest(x):
    if x==1:
        return 'Spruce/Fir'
    elif x==2:
        return 'Lodgepole Pine'
    elif x==3:
        return 'Ponderosa Pine'
    elif x==4:
        return 'Cottonwood/Willow'
    elif x==5:
        return 'Aspen'
    elif x==6:
        return 'Douglas-fir'
    elif x==7:
        return 'Krummholz'
    
train['Cover_Type'] = train['Cover_Type'].apply(lambda x: forest(x))

#Décommentez et modifiez le numéro(entre 1 et 7) dans hue_order pour changer de Cover_Type
#sns.pairplot(train.drop("Id",axis=1), hue="Cover_Type", hue_order=[1], vars=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"], diag_kind="kde")

#Décommentez et modifiez le numéro(entre 1 et 40) de Soil_TypeX et (1 et 4) de Wilderness_AreaX
#train[['Soil_Type4', 'Cover_Type']].groupby(['Cover_Type'], as_index=False).mean().sort_values(by='Soil_Type4', ascending=False)
#train[['Wilderness_Area1', 'Cover_Type']].groupby(['Cover_Type'], as_index=False).mean().sort_values(by='Wilderness_Area1', ascending=False)

#On ramène les 40 colonnes Soil_TypeX à une seule, de même pour les 4 colonnes Wilderness_AreaX
train['Wilderness_Area'] =((train['Wilderness_Area1'])+(train['Wilderness_Area2']*2)
                           +(train['Wilderness_Area3']*3)+(train['Wilderness_Area4']*4))

train['Soil_Type']=((train['Soil_Type1'])+(train['Soil_Type2']*2)+(train['Soil_Type3']*3)
                    +(train['Soil_Type4']*4)+(train['Soil_Type5']*5)+(train['Soil_Type6']*6)
                    +(train['Soil_Type7']*7)+(train['Soil_Type8']*8)+(train['Soil_Type9']*9)
                    +(train['Soil_Type10']*10)+(train['Soil_Type11']*11)+(train['Soil_Type12']*12)
                    +(train['Soil_Type13']*13)+(train['Soil_Type14']*14)+(train['Soil_Type15']*15)
                    +(train['Soil_Type16']*16)+(train['Soil_Type17']*17)+(train['Soil_Type18']*18)
                    +(train['Soil_Type19']*19)+(train['Soil_Type20']*20)+(train['Soil_Type21']*21)
                    +(train['Soil_Type22']*22)+(train['Soil_Type23']*23)+(train['Soil_Type24']*24)
                    +(train['Soil_Type25']*25)+(train['Soil_Type26']*26)+(train['Soil_Type27']*27)
                    +(train['Soil_Type28']*28)+(train['Soil_Type29']*29)+(train['Soil_Type30']*30)
                    +(train['Soil_Type31']*31)+(train['Soil_Type32']*32)+(train['Soil_Type33']*33)
                    +(train['Soil_Type34']*34)+(train['Soil_Type35']*35)+(train['Soil_Type36']*36)
                    +(train['Soil_Type37']*37)+(train['Soil_Type38']*38)+(train['Soil_Type39']*39)+(train['Soil_Type40']*40))

train.drop(['Wilderness_Area1'],inplace = True, axis=1)
train.drop(['Wilderness_Area2'],inplace = True, axis=1)
train.drop(['Wilderness_Area3'],inplace = True, axis=1)
train.drop(['Wilderness_Area4'],inplace = True, axis=1)
train.drop(['Soil_Type1'],inplace = True, axis=1)
train.drop(['Soil_Type2'],inplace = True, axis=1)
train.drop(['Soil_Type3'],inplace = True, axis=1)
train.drop(['Soil_Type4'],inplace = True, axis=1)
train.drop(['Soil_Type5'],inplace = True, axis=1)
train.drop(['Soil_Type6'],inplace = True, axis=1)
train.drop(['Soil_Type7'],inplace = True, axis=1)
train.drop(['Soil_Type8'],inplace = True, axis=1)
train.drop(['Soil_Type9'],inplace = True, axis=1)
train.drop(['Soil_Type10'],inplace = True, axis=1)
train.drop(['Soil_Type11'],inplace = True, axis=1)
train.drop(['Soil_Type12'],inplace = True, axis=1)
train.drop(['Soil_Type13'],inplace = True, axis=1)
train.drop(['Soil_Type14'],inplace = True, axis=1)
train.drop(['Soil_Type15'],inplace = True, axis=1)
train.drop(['Soil_Type16'],inplace = True, axis=1)
train.drop(['Soil_Type17'],inplace = True, axis=1)
train.drop(['Soil_Type18'],inplace = True, axis=1)
train.drop(['Soil_Type19'],inplace = True, axis=1)
train.drop(['Soil_Type20'],inplace = True, axis=1)
train.drop(['Soil_Type21'],inplace = True, axis=1)
train.drop(['Soil_Type22'],inplace = True, axis=1)
train.drop(['Soil_Type23'],inplace = True, axis=1)
train.drop(['Soil_Type24'],inplace = True, axis=1)
train.drop(['Soil_Type25'],inplace = True, axis=1)
train.drop(['Soil_Type26'],inplace = True, axis=1)
train.drop(['Soil_Type27'],inplace = True, axis=1)
train.drop(['Soil_Type28'],inplace = True, axis=1)
train.drop(['Soil_Type29'],inplace = True, axis=1)
train.drop(['Soil_Type30'],inplace = True, axis=1)
train.drop(['Soil_Type31'],inplace = True, axis=1)
train.drop(['Soil_Type32'],inplace = True, axis=1)
train.drop(['Soil_Type33'],inplace = True, axis=1)
train.drop(['Soil_Type34'],inplace = True, axis=1)
train.drop(['Soil_Type35'],inplace = True, axis=1)
train.drop(['Soil_Type36'],inplace = True, axis=1)
train.drop(['Soil_Type37'],inplace = True, axis=1)
train.drop(['Soil_Type38'],inplace = True, axis=1)
train.drop(['Soil_Type39'],inplace = True, axis=1)
train.drop(['Soil_Type40'],inplace = True, axis=1)

#Idem Test
test['Wilderness_Area'] =((test['Wilderness_Area1'])+(test['Wilderness_Area2']*2)
                           +(test['Wilderness_Area3']*3)+(test['Wilderness_Area4']*4))

test['Soil_Type']=((test['Soil_Type1'])+(test['Soil_Type2']*2)+(test['Soil_Type3']*3)
                    +(test['Soil_Type4']*4)+(test['Soil_Type5']*5)+(test['Soil_Type6']*6)
                    +(test['Soil_Type7']*7)+(test['Soil_Type8']*8)+(test['Soil_Type9']*9)
                    +(test['Soil_Type10']*10)+(test['Soil_Type11']*11)+(test['Soil_Type12']*12)
                    +(test['Soil_Type13']*13)+(test['Soil_Type14']*14)+(test['Soil_Type15']*15)
                    +(test['Soil_Type16']*16)+(test['Soil_Type17']*17)+(test['Soil_Type18']*18)
                    +(test['Soil_Type19']*19)+(test['Soil_Type20']*20)+(test['Soil_Type21']*21)
                    +(test['Soil_Type22']*22)+(test['Soil_Type23']*23)+(test['Soil_Type24']*24)
                    +(test['Soil_Type25']*25)+(test['Soil_Type26']*26)+(test['Soil_Type27']*27)
                    +(test['Soil_Type28']*28)+(test['Soil_Type29']*29)+(test['Soil_Type30']*30)
                    +(test['Soil_Type31']*31)+(test['Soil_Type32']*32)+(test['Soil_Type33']*33)
                    +(test['Soil_Type34']*34)+(test['Soil_Type35']*35)+(test['Soil_Type36']*36)
                    +(test['Soil_Type37']*37)+(test['Soil_Type38']*38)+(test['Soil_Type39']*39)+(test['Soil_Type40']*40))

test.drop(['Wilderness_Area1'],inplace = True, axis=1)
test.drop(['Wilderness_Area2'],inplace = True, axis=1)
test.drop(['Wilderness_Area3'],inplace = True, axis=1)
test.drop(['Wilderness_Area4'],inplace = True, axis=1)
test.drop(['Soil_Type1'],inplace = True, axis=1)
test.drop(['Soil_Type2'],inplace = True, axis=1)
test.drop(['Soil_Type3'],inplace = True, axis=1)
test.drop(['Soil_Type4'],inplace = True, axis=1)
test.drop(['Soil_Type5'],inplace = True, axis=1)
test.drop(['Soil_Type6'],inplace = True, axis=1)
test.drop(['Soil_Type7'],inplace = True, axis=1)
test.drop(['Soil_Type8'],inplace = True, axis=1)
test.drop(['Soil_Type9'],inplace = True, axis=1)
test.drop(['Soil_Type10'],inplace = True, axis=1)
test.drop(['Soil_Type11'],inplace = True, axis=1)
test.drop(['Soil_Type12'],inplace = True, axis=1)
test.drop(['Soil_Type13'],inplace = True, axis=1)
test.drop(['Soil_Type14'],inplace = True, axis=1)
test.drop(['Soil_Type15'],inplace = True, axis=1)
test.drop(['Soil_Type16'],inplace = True, axis=1)
test.drop(['Soil_Type17'],inplace = True, axis=1)
test.drop(['Soil_Type18'],inplace = True, axis=1)
test.drop(['Soil_Type19'],inplace = True, axis=1)
test.drop(['Soil_Type20'],inplace = True, axis=1)
test.drop(['Soil_Type21'],inplace = True, axis=1)
test.drop(['Soil_Type22'],inplace = True, axis=1)
test.drop(['Soil_Type23'],inplace = True, axis=1)
test.drop(['Soil_Type24'],inplace = True, axis=1)
test.drop(['Soil_Type25'],inplace = True, axis=1)
test.drop(['Soil_Type26'],inplace = True, axis=1)
test.drop(['Soil_Type27'],inplace = True, axis=1)
test.drop(['Soil_Type28'],inplace = True, axis=1)
test.drop(['Soil_Type29'],inplace = True, axis=1)
test.drop(['Soil_Type30'],inplace = True, axis=1)
test.drop(['Soil_Type31'],inplace = True, axis=1)
test.drop(['Soil_Type32'],inplace = True, axis=1)
test.drop(['Soil_Type33'],inplace = True, axis=1)
test.drop(['Soil_Type34'],inplace = True, axis=1)
test.drop(['Soil_Type35'],inplace = True, axis=1)
test.drop(['Soil_Type36'],inplace = True, axis=1)
test.drop(['Soil_Type37'],inplace = True, axis=1)
test.drop(['Soil_Type38'],inplace = True, axis=1)
test.drop(['Soil_Type39'],inplace = True, axis=1)
test.drop(['Soil_Type40'],inplace = True, axis=1)


#Diagrammes
#cmap = sns.color_palette("Set2")
#sns.countplot(x='Cover_Type', data=train, palette=cmap)
#sns.countplot(x='Wilderness_Area',data=train, palette=cmap)
#sns.countplot(x='Wilderness_Area',data=test, palette=cmap)

#Types d'arbres en fonction de Wilderness_Area
#plt.figure(figsize=(12,6))
#sns.countplot(x='Cover_Type',data=train, hue='Wilderness_Area')

#Fréquences Soil_Type
#plt.figure(figsize=(12,5))
#sns.countplot(x='Soil_Type',data=train)
#plt.figure(figsize=(12,5))
#sns.countplot(x='Soil_Type',data=test)


#Fusion de certaines colonnes, on pourra en ajouter d'autres
train['HHydrology_HFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Neg_HHydrology_HFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HHydrology_HRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HHydrology_HRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['HFire_Points_HRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HFire_HRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']

train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] ) / 3

train['Mean_HHydrology_HFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2
train['Mean_HHydrology_HRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2
train['Mean_HFire_HRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2

train['MeanNeg_Mean_HHydrology_HFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2
train['MeanNeg_HHydrology_HRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HFire_HRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2

train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3
train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology']) / 2 

train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])

train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2


test['HHydrology_HFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Neg_HHydrology_HFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HHydrology_HRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HHydrology_HRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['HFire_HRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HFire_HRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']
test['Elevation_Vertical'] = test['Elevation'] + test['Vertical_Distance_To_Hydrology']

test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] ) / 3

test['Mean_HHydrology_HFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2
test['Mean_HHydrology_HRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2
test['Mean_HFire_HRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2

test['MeanNeg_Mean_HHydrology_HFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2
test['MeanNeg_HHydrology_HRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HFire_HRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2

test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3 
test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology']) / 2


test['Vertical_Distance_To_Hydrology'] = abs(test["Vertical_Distance_To_Hydrology"])

test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2


#ICI C'EST POUR TEST, CA DONNE 92,99% DE REUSSITE, SACHANT QUE SANS LES MODIFICATIONS PRECEDANTES, ON AVAIT 89,15%

#On enlève le cover type pour vérifier que ça marche
x = train.drop(['Cover_Type'], axis = 1)
y = train['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.05, random_state=42 )
unique, count= np.unique(y_train, return_counts=True)
#print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )

#Voir l'impact de chaque donnée sur Cover_Type

# Charger le dataset iris
dataset = datasets.load_iris()

# Fit un modèle Extra Trees dans le data
clf = ExtraTreesClassifier()
clf.fit(x_train,y_train)
# Montrer l'importance de chaque attribut
z = clf.feature_importances_
#Faire un dataframe pour montrer chaque valeur et le nom de sa colonne
df = pd.DataFrame()
print(len(z))
print(len(list(x.columns.values)))

df["values"] = z
df['column'] = list(x.columns.values)
# Trier en descente pour avoir 
df.sort_values(by='values', ascending=False, inplace = True)
df.head(100)

#Laisser, ça augmente le taux de réussite
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn import decomposition

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


clf = ExtraTreesClassifier(n_estimators=950, random_state=0)

clf.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(x_test, y_test) * 100))
