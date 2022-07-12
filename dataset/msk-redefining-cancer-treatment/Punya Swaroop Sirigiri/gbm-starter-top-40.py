# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

#Read Data
train_variant = pd.read_csv("../input/training_variants")
test_variant = pd.read_csv("../input/test_variants")
train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#Train data number of observations in each class
##Class1:568#Class2:452#Class3:89#Class4:686
##Class6:275#Class7:953#Class8:19#Class9:37
##Class5:242
train = pd.merge(train_variant, train_text, how='left', on='ID')
x_train = train.drop('Class', axis=1)
# number of train data : 3321

x_test = pd.merge(test_variant, test_text, how='left', on='ID')
test_index = x_test['ID'].values
# number of test data : 5668

data = np.concatenate((x_train, x_test), axis=0)
data=pd.DataFrame(data)
data.columns = ["ID", "Gene", "Variation", "Text"]

#TFIDF
mod=TfidfVectorizer(min_df=5, max_features=500, stop_words='english')
mod_TD=mod.fit_transform(data.Text)

#SVD features
SVD=TruncatedSVD(200,random_state=41)
SVD_FIT=SVD.fit_transform(mod_TD)
yet_to_complete=pd.DataFrame(SVD_FIT)
#data.drop(data.columns[[0,3]],inplace=True, axis=1)
#as Gene and Variation data values are just scattered like IDS, i dont think these give u great info about the prediction
encoder = LabelEncoder()
y_train = train['Class'].values
encoder.fit(y_train)
encoded_y = encoder.transform(y_train)

#gbm algorithm with random parameters
gbm1=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
gbm1.fit(yet_to_complete[:3321],encoded_y)

#predictions
y_pred=gbm1.predict_proba(yet_to_complete[3321:])

#tweaking the submission file as required
subm_file = pd.DataFrame(y_pred)
subm_file['id'] = test_index
subm_file.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
subm_file.to_csv("submission.csv",index=False)

