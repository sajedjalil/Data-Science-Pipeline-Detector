import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split

loc_train = "../input/train.csv"
loc_test = "../input/test.csv"
loc_submission = "submission.csv"

df = pd.read_csv(loc_train)
output = pd.read_csv(loc_test)

feature_cols = [col for col in df.columns if col not in ['Cover_Type','Id']]

df_train, df_test = train_test_split(df, test_size = 0.2)

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
y_train = df_train['Cover_Type']
y_test = df_test['Cover_Type']
output_ids = output['Id']

results = []
sample_leaf_options = list(range(1,100,4))
n_estimators_options = list(range(1,200,5))

for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        alg = ensemble.RandomForestClassifier(min_samples_leaf = leaf_size, n_estimators =\
        n_estimators_size, n_jobs=-1, random_state=0)
        alg.fit(X_train, y_train)
        predict = alg.predict(X_test)
        results.append((leaf_size, n_estimators_size, (y_test == predict).mean()))
        print((y_test == predict).mean())
        
parameter = max(results, key = lambda x: x[2])
print(parameter)

alg = ensemble.RandomForestClassifier(min_samples_leaf = parameter[0], n_estimators =\
parameter[1], n_jobs=-1, random_state=0)
alg.fit(X_train, y_train)

with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(alg.predict(output[feature_cols]))):
        outfile.write("%s,%s\n"%(output_ids[e],val))