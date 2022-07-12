import csv
import gzip
###
import numpy as np
###
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


def get_data(fn):
  data = []
  with open(fn) as f:
    reader = csv.DictReader(f)
    data = [row for row in reader]
  return data


def get_fields(data, fields):
  extracted = []
  for row in data:
    extract = []
    for field, f in sorted(fields.items()):
      info = f(row[field])
      if type(info) == list:
        extract.extend(info)
      else:
        extract.append(info)
    extracted.append(np.array(extract, dtype=np.float32))
  return extracted


def shuffle(X, y, seed=1337):
  np.random.seed(seed)
  shuffle = np.arange(len(y))
  np.random.shuffle(shuffle)
  X = X[shuffle]
  y = y[shuffle]
  return X, y


def preprocess_data(X, scaler=None):
  if not scaler:
    scaler = StandardScaler()
    scaler.fit(X)
  X = scaler.transform(X)
  return X, scaler


def dating(x):
  date, time = x.split(' ')
  y, m, d = map(int, date.split('-'))
  time = time.split(':')[:2]
  time = int(time[0]) * 60 + int(time[1])
  return [y, m, d, time]

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
districts = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
labels = 'ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'.split(',')
data_fields = {
    'X': lambda x: float(x),
    'Y': lambda x: float(x),
    'DayOfWeek': lambda x: days.index(x) / float(len(days)),
    'Address': lambda x: [1 if 'block' in x.lower() else 0],
    'PdDistrict': lambda x: [1 if x == d else 0 for d in districts],
    'Dates': dating,  # Parse 2015-05-13 23:53:00
}
label_fields = {'Category': lambda x: labels.index(x.replace(',', ''))}

print('Loading training data...')
raw_train = get_data('../input/train.csv')
print('Creating training data...')
X = np.array(get_fields(raw_train, data_fields), dtype=np.float32)
print('Creating training labels...')
y = np.array(get_fields(raw_train, label_fields))
del raw_train

X, y = shuffle(X, y)
X, scaler = preprocess_data(X)
Y = np_utils.to_categorical(y)

input_dim = X.shape[1]
output_dim = len(labels)
print('Input dimensions: {}'.format(input_dim))


def build_model(input_dim, output_dim, hn=32, dp=0.5, layers=1):
    model = Sequential()
    model.add(Dense(input_dim, hn, init='glorot_uniform'))
    model.add(PReLU((hn,)))
    model.add(Dropout(dp))

    for i in range(layers):
      model.add(Dense(hn, hn, init='glorot_uniform'))
      model.add(PReLU((hn,)))
      model.add(BatchNormalization((hn,)))
      model.add(Dropout(dp))

    model.add(Dense(hn, output_dim, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

EPOCHS = 1
BATCHES = 128
HN = 64
RUN_FOLDS = False
nb_folds = 4
kfolds = KFold(len(y), nb_folds)
av_ll = 0.
f = 0
if RUN_FOLDS:
  for train, valid in kfolds:
      print('---' * 20)
      print('Fold', f)
      print('---' * 20)
      f += 1
      X_train = X[train]
      X_valid = X[valid]
      Y_train = Y[train]
      Y_valid = Y[valid]
      y_valid = y[valid]

      print("Building model...")
      model = build_model(input_dim, output_dim, HN)

      print("Training model...")

      model.fit(X_train, Y_train, nb_epoch=EPOCHS, batch_size=BATCHES, validation_data=(X_valid, Y_valid), verbose=0)
      valid_preds = model.predict_proba(X_valid)
      ll = metrics.log_loss(y_valid, valid_preds)
      print("LL:", ll)
      av_ll += ll
  print('Average LL:', av_ll / nb_folds)

print("Generating submission...")

model = build_model(input_dim, output_dim, HN)
model.fit(X, Y, nb_epoch=EPOCHS, batch_size=BATCHES, verbose=0)

print('Loading testing data...')
raw_test = get_data('../input/test.csv')
print('Creating testing data...')
X_test = np.array(get_fields(raw_test, data_fields), dtype=np.float32)
del raw_test
X_test, _ = preprocess_data(X_test, scaler)

print('Predicting over testing data...')
preds = model.predict_proba(X_test, verbose=0)

with gzip.open('sf-nn.csv.gz', 'wt') as outf:
  fo = csv.writer(outf, lineterminator='\n')
  fo.writerow(['Id'] + labels)

  for i, pred in enumerate(preds):
    fo.writerow([i] + list(pred))