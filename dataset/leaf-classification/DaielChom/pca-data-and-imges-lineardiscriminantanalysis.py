# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io
from skimage import transform as tf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def ruta(num):
    return "../input/images/"+str(num+1)+".jpg"

set_images = np.array([io.imread(ruta(i)) for i in range(1584)])

#Extraccion de las clases
clases = np.unique(train['species'].values)

#Auxiliares para obtener y_train en base a clases
clases_num = np.array(range(99))
aux_y = np.array(train['species'].values)

data_y_train = np.ones(len(aux_y))
data_X_train = train.drop(['id', 'species'], axis = 1).values
data_X_test = test.drop(['id'], axis=1).values

#Convierte los string de las clases a entero y los guarda en y_train
for i,n in enumerate(aux_y):
    for j,l in enumerate(clases):
        if (n == l):
            data_y_train[i] = j
            
id_train = train['id'].values
id_test = test['id'].values

aux_img_train = np.array([set_images[i-1] for i in id_train])
aux_img_test = np.array([set_images[i-1] for i in id_test])

resize_train = []
resize_test = []

for i in aux_img_train:
    resize_train.append(tf.resize(i,(300,300)).flatten())
    

for i in aux_img_test:

    resize_test.append(tf.resize(i,(300,300)).flatten())

pca = PCA(n_components=60)
# PCA Datos
Xp = pca.fit_transform(data_X_train)

# PCA Imágenes
Xp_img = pca.fit_transform(resize_train)


Xp_data_img = np.concatenate((Xp,Xp_img),axis = 1)


# test
# PCA Datos
Xp_test = pca.fit_transform(data_X_test)

# PCA Imágenes
Xp_img_test = pca.fit_transform(resize_test)

# PCA datos e Imágenes

Xp_data_img_test = np.concatenate((Xp_test,Xp_img_test),axis = 1)


favorite = LinearDiscriminantAnalysis()

favorite.fit(Xp_data_img, data_y_train)
test_predictions = favorite.predict_proba(Xp_data_img_test)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=clases)
submission.insert(0, 'id', id_test)
submission.reset_index()

# Export Submission
#submission.to_csv('submission.csv', index = False)
submission.tail()