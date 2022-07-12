import os
from time import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, log_loss

DATA_PATH = "dataset"
# noinspection SpellCheckingInspection
ZIP_NAME = "lish-moa.zip"
# noinspection SpellCheckingInspection
CSV_FILES = ["sample_submission.csv",
             "test_features.csv",
             "train_drug.csv",
             "train_features.csv",
             "train_targets_nonscored.csv",
             "train_targets_scored.csv"]

OUT_LOCATION = "/kaggle/working"
IN_LOCATION = "/kaggle/input/lish-moa"

random_state = 42


class ModelWrapper(BaseEstimator):
    def __init__(self, model, pipeline, remove_perts=False):
        self.model = model
        self.pipeline = pipeline
        self.remove_perts = remove_perts

    def fit(self, x, y):
        return self.model.fit(self.pipeline.fit_transform(x), y)

    def predict_proba(self, x):
        if self.remove_perts:
            to_zero = x['cp_type'] == 'ctl_vehicle'
        else:
            to_zero = []
        x = self.pipeline.fit_transform(x)
        proba = self.model.predict_proba(x)
        proba[to_zero, :] = 0.
        return proba

    def predict(self, x):
        if self.remove_perts:
            to_zero = x['cp_type'] == 'ctl_vehicle'
        else:
            to_zero = []
        x = self.pipeline.fit_transform(x)
        pred = self.model.predict(x)
        pred[to_zero, :] = 0.
        return pred


# ------------------ PERFORMANCE METRICS ------------------
def measure_performance(truth_data, predictions):
    return log_loss(np.ravel(truth_data), np.ravel(predictions))


# ------------------ DATA FUNCTIONS ------------------
def load_data(pathname=IN_LOCATION, csv_files=None):
    if csv_files is None:
        csv_files = CSV_FILES
    data = []
    for file in csv_files:
        data.append(pd.read_csv(os.path.join(pathname, file)))
    return tuple(data)


# ------------------ PIPELINE FUNCTIONALITY ------------------
def get_pipeline(numeric_attributes, ordinal_attributes, one_hot_attributes=None):
    if one_hot_attributes is None:
        # Forward attributes to proper pipeline
        return ColumnTransformer([
            ('numeric', StandardScaler(), numeric_attributes),
            ('ordinal', OrdinalEncoder(), ordinal_attributes)
        ])
    else:
        # Forward attributes to proper pipeline
        return ColumnTransformer([
            ('numeric', StandardScaler(), numeric_attributes),
            ('ordinal', OrdinalEncoder(), ordinal_attributes),
            ('one_hot', OneHotEncoder(), one_hot_attributes)
        ])


# ------------------ MODEL FUNCTIONALITY ------------------
def get_models():
    def make_mlp_entry(alpha, hidden_layer_sizes=(600,), max_iterations=500):
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            random_state=random_state,
                            alpha=alpha,
                            #early_stopping=True,
                            max_iter=max_iterations)
        return "MLP (" + ','.join([str(size) for size in mlp.hidden_layer_sizes]) + "), α = " + str(mlp.alpha), mlp
    return [
        make_mlp_entry(0.75),
        make_mlp_entry(1.0),
        make_mlp_entry(1.25),
        make_mlp_entry(1.5),
        make_mlp_entry(1.75),
        make_mlp_entry(2.0),
        make_mlp_entry(2.25),
        make_mlp_entry(2.5),
        make_mlp_entry(2.75)
    ]


# ------------------ FULL RUNNER ------------------
def run_final_project():
    np.random.seed(random_state)
    [submission, x_test, _, x_train, _, y_train] = load_data()

    # Remove sig_id
    assert(np.array_equal(x_train['sig_id'], y_train['sig_id']))
    test_sig_id = x_test['sig_id']
    for data in [x_train, y_train, x_test]:
        data.drop(columns='sig_id', inplace=True)

    categorical_attributes = ['cp_dose', 'cp_type', 'cp_time']
    numeric_attributes = x_train.drop(columns=categorical_attributes).columns
    pipeline = get_pipeline(numeric_attributes, categorical_attributes[0:2], categorical_attributes[2:])
    #categorical_attributes = ['cp_dose', 'cp_type']
    #numeric_attributes = x_train.drop(columns=categorical_attributes).columns
    #pipeline = get_pipeline(numeric_attributes, categorical_attributes)
    best_score = float('inf')
    best_model = None
    best_model_name = None
    for model_name, model_in in get_models():
        print("--- Model --- ", model_name)
        model = ModelWrapper(model_in, pipeline, True)
        start = time()
        scores = cross_val_score(model, x_train, y_train, scoring=make_scorer(measure_performance, needs_proba=True), cv=5)
        score = np.mean(scores)
        print("Spent %.2f seconds" % (time() - start))
        print("Mean: ", score)
        print("STD: ", np.std(scores))
        print()

        if score < best_score:
            best_score = score
            best_model = model
            best_model_name = model_name

    print('Best Model: ', best_model_name)
    #best_model.fit(x_train, y_train)
    #submission[submission.columns[1:]] = best_model.predict_proba(x_test)
    #submission.to_csv(os.path.join(OUT_LOCATION, "submission.csv"), index=False)

    print('Done!')


if __name__ == "__main__":
    run_final_project()
