import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
type_mapping = {'Ghoul': 0, 'Goblin': 1, 'Ghost': 2}
reverse_type_mapping = {0: 'Ghoul', 1: 'Goblin', 2: 'Ghost'}
train_df['type'] = train_df['type'].copy().map(type_mapping)


comb = list(itertools.combinations(train_df.drop(["id", "color", "type"], axis=1).columns, 2))
try_comb = pd.DataFrame()
for c in comb:
    try_comb[c[0]+"_x_"+c[1]] = train_df[c[0]].values * train_df[c[1]].values

try_comb["type"] = train_df.type

for i in [1,2,-1]:
    train_df[comb[i][0]+"_x_"+comb[i][1]] = train_df[comb[i][0]].values * train_df[comb[i][1]].values
    test_df[comb[i][0]+"_x_"+comb[i][1]] = test_df[comb[i][0]].values * test_df[comb[i][1]].values
    
train_df['hair_x_soul_x_bone'] = train_df['hair_length'].values * train_df['has_soul'].values * train_df['bone_length'].values
test_df['hair_x_soul_x_bone'] = test_df['hair_length'].values * test_df['has_soul'].values * test_df['bone_length'].values


labels_df = train_df['type'].copy()
features_df = train_df.copy().drop(['id', 'type', 'color'], axis=1)

test_ids = test_df['id'].copy()
test_features_df = test_df.copy().drop(['id', 'color'], axis=1)

all_features = pd.concat([features_df, test_features_df])
all_labels = pd.concat([labels_df, pd.DataFrame([-1]*len(test_features_df))])

NR_FOLDS=5
skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=1)
accuracies = []
for fold, (train_idx, test_idx) in enumerate(skf):
    X_train = features_df.iloc[train_idx, :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)
    X_train = pd.concat([X_train, test_features_df])
    y_train = pd.concat([y_train, pd.DataFrame([-1]*len(test_features_df))])
    
    #print(X_train.count())
    #print(X_train)
    #print(y_train.count())
    
    params = {'n_neighbors':[3,5,7,11], 'gamma':[1,5,10,20,50], 'kernel': ['rbf', 'knn'],
              'alpha': [0.05, 0.1, 0.2, 0.5], 'max_iter': [10, 50, 100, 250]}
    label_model = LabelSpreading(gamma=100, n_neighbors=15, kernel='knn', max_iter=10)
    label_model.fit(X_train, y_train)
    
    predictions = label_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    #print('SVM:\n', conf_matrix)
    accuracy = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])/np.sum(conf_matrix)
    #print('accuracy:', accuracy)
    accuracies.append(accuracy)
print('Avg acc:', np.mean(accuracies))
    
label_model = LabelSpreading(gamma=50)#, n_neighbors=30, kernel='knn')
label_model.fit(all_features, all_labels)
prediction_vectors = []
for (i,j) in zip(test_ids, label_model.predict(test_features_df)):
    prediction_vectors.append([i,reverse_type_mapping[j]])
predictions_df = pd.DataFrame(prediction_vectors)
predictions_df.columns=['id', 'type']
predictions_df.to_csv('submission_rbf.csv', index=False)

label_model = LabelSpreading(n_neighbors=15, kernel='knn', max_iter=10)
label_model.fit(all_features, all_labels)
prediction_vectors = []
for (i,j) in zip(test_ids, label_model.predict(test_features_df)):
    prediction_vectors.append([i,reverse_type_mapping[j]])
predictions_df = pd.DataFrame(prediction_vectors)
predictions_df.columns=['id', 'type']
predictions_df.to_csv('submission_knn.csv', index=False)