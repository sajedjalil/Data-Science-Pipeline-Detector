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

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.optimizers import Adam

from keras import regularizers
from keras.layers import Input
from keras.models import Model

np.random.seed(1)  # for reproducibility

print('Loading data...')
train_in = pd.read_csv('../input/train.csv')
test_in = pd.read_csv('../input/test.csv')

### Create dummy variables and prepare training/test data frames...
print('Creating dummies...')
# Save our target variable column before we drop it
y_train = train_in.iloc[:, -1]

# Stack training and test data into one big data frame so we cover all categories found in both
data = pd.concat((train_in.iloc[:, :-1], test_in))

# Get dummy variables for all categorical columns
colnames = []
X_data = []

# For each categorical column...
for c in [i for i in data.columns[1:-1] if 'cat' in i]:
    # Get dummy variables for that column
    dummies = pd.get_dummies(data[c])
    # Drop the last dummy, as its value is implied by the others (all 0's = 1)
    dummies = dummies.iloc[:, :-1].values.astype(np.bool)
    X_data.append(dummies)
    # Create column names for those dummy variables
    colnames += [c + '_' + str(i) for i in range(dummies.shape[1])]
    
# Stack all dummy variables into big dataframe with the colnames
X_data = pd.DataFrame(np.hstack(X_data), columns=colnames)

# Drop any columns with only 1 value (so drop unused categories, if any)
X_data = X_data.iloc[:, [len(pd.unique(X_data.loc[:,c]))>1 for c in X_data.columns]]

## Get the other (continuous) columns
#X_data_cont = np.vstack([data[c].values.astype(np.float32) \
#                         for c in data.columns[1:-1] if 'cat' not in c]).T
#
## Final data frame is the dummy variables + the continuous variables
#X_data = X_data.join(pd.DataFrame(X_data_cont, 
#                    columns=[c for c in data.columns[1:-1] if 'cat' not in c]))

# Create X train and y train arrays for input to NN
Xt = X_data.values

in_dim = Xt.shape[1]

input_img = Input(shape=(in_dim,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dropout(0.5)(encoded)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dropout(0.5)(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dropout(0.5)(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(in_dim, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(loss='mse', optimizer=Adam())


history = autoencoder.fit(Xt, Xt,
                          batch_size=96*4,
                          nb_epoch=2,
                          verbose=2, 
                          shuffle=True)

encoder = Model(input=input_img, output=encoded)
pd.DataFrame(encoder.predict(Xt)).to_csv('all_data_encoded.csv', index=False)