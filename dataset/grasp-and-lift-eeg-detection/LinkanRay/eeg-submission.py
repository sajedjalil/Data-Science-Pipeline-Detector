from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def load_file_train(file):
    data = pd.read_csv(file, skiprows=range(0))
    event_file = file.replace('data', 'events')
    event = pd.read_csv(event_file)
    return data, event


def load_file_test(file):
    data = pd.read_csv(file, skiprows=range(0))
    return data


subjs = range(1, 3)
user_id = ['id']
event_header = ['HandStart', 'FirstDigitTouch',
                'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']
columns = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
           'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']

data_train_clean = pd.DataFrame(columns=columns)
data_test_clean = pd.DataFrame(columns=columns)
event_train_clean = pd.DataFrame(columns=event_header)
user_id_clean = pd.DataFrame(columns=user_id)

for sub in subjs:
    files_train = glob('../input/train/subj%d_series*_data.csv' % (sub))

    for file in files_train:
        data_train, events_train = load_file_train(file)
        data_train_inter = data_train.drop(['id'], axis=1)
        data_train_clean = data_train_clean.append(data_train_inter, ignore_index=True)
        event_train_inter = events_train.drop(['id'], axis=1)
        event_train_clean = event_train_clean.append(event_train_inter, ignore_index=True)


    files_test = glob('../input/test/subj%d_series*_data.csv' % (sub))
    for file in files_test:
        data_test = load_file_test(file)
        user_id_inter = data_test.drop(columns, axis=1)
        user_id_clean = user_id_clean.append(user_id_inter, ignore_index=True)
        data_test_inter = data_test.drop(['id'], axis=1)
        data_test_clean = data_test_clean.append(data_test_inter, ignore_index=True)

data_train_fn = data_train_clean.values
data_test_fn = data_test_clean.values


pca = PCA(n_components=0.8,whiten=True)
data_train_fn = pca.fit_transform(data_train_fn)
data_test_fn = pca.transform(data_test_fn)


#model = svm.SVC(kernel='rbf',C=10)
model = LogisticRegression()
model.fit(data_train_fn,event_train_clean['HandStart'].values)
predicted_HandStart = model.predict(data_test_fn)

model.fit(data_train_fn,event_train_clean['FirstDigitTouch'].values)
predicted_FirstDigitTouch = model.predict(data_test_fn)

model.fit(data_train_fn,event_train_clean['BothStartLoadPhase'].values)
predicted_BothStartLoadPhase = model.predict(data_test_fn)

model.fit(data_train_fn,event_train_clean['LiftOff'].values)
predicted_LiftOff = model.predict(data_test_fn)

model.fit(data_train_fn,event_train_clean['Replace'].values)
predicted_Replace = model.predict(data_test_fn)

model.fit(data_train_fn,event_train_clean['BothReleased'].values)
predicted_BothReleased = model.predict(data_test_fn)

output = pd.DataFrame(columns=['Id', 'HandStart', 'FirstDigitTouch',
                'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased'])
output['Id'] = user_id_clean
output['HandStart'] = predicted_HandStart.astype(int)
output['FirstDigitTouch'] = predicted_FirstDigitTouch.astype(int)
output['BothStartLoadPhase'] = predicted_BothStartLoadPhase.astype(int)
output['LiftOff'] = predicted_LiftOff.astype(int)
output['Replace'] = predicted_Replace.astype(int)
output['BothReleased'] = predicted_BothReleased.astype(int)

output.to_csv('logisticRegressionSubmit.csv', index=False)


