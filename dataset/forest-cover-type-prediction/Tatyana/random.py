from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train=train.drop('Id',axis=1)
ids=test['Id']
test=test.drop('Id',axis=1)
y=train['Cover_Type']
xtrain=train.drop('Cover_Type',axis=1)
#расстояние до водных источников;
def distance(myData):
    myData['Distance_To_Hydrology']=(myData['Vertical_Distance_To_Hydrology']**2.0+
                                     myData['Horizontal_Distance_To_Hydrology']**2.0)**0.5

distance(xtrain)
distance(test)                                     

clf =RandomForestClassifier(random_state=0)
clf.set_params(max_features='sqrt')
clf.set_params(n_estimators=200)
clf.fit(xtrain,y)
with open("forest2.csv", "w") as outfile: 
    outfile.write('Id,Cover_Type\n') 
    for e, val in enumerate(list(clf.predict(test))): 
        outfile.write('%s,%s\n'%(ids[e],val))