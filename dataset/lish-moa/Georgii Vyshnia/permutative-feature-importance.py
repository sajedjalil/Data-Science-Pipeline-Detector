import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import warnings
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder

from skmultilearn.model_selection import IterativeStratification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# permutation importance
from sklearn.inspection import permutation_importance

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score, log_loss


# read data
in_kaggle = True


def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str]:
    train_path = ''
    test_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/lish-moa/train_features.csv'
        train_targets_path = '../input/lish-moa/train_targets_scored.csv'
        test_path = '../input/lish-moa/test_features.csv'
        sample_submission_path = '../input/lish-moa/sample_submission.csv'
    else:
        # running locally
        train_path = 'data/train_features.csv'
        train_targets_path = 'data/train_targets_scored.csv'
        test_path = 'data/test_features.csv'
        sample_submission_path = 'data/sample_submission.csv'

    return train_path, train_targets_path, test_path, sample_submission_path


# Here's how you use permutation importance

def get_permutation_importance(training_set, target, model) -> pd.DataFrame:
    model.fit(training_set, target)

    result = permutation_importance(model, training_set, target, n_repeats=1,
                                    random_state=0)

    # permutational importance results
    result_df = pd.DataFrame(colnames, columns=['Feature'])
    result_df['permutation_importance'] = result.get('importances')

    return result_df


if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Started at ", start_time)

    # Import data
    train_set_path, train_set_targets_path, test_set_path, sample_subm_path = get_data_file_path(in_kaggle)

    train_features = pd.read_csv(train_set_path)
    test_features = pd.read_csv(test_set_path)
    train_targets_scored = pd.read_csv(train_set_targets_path)

    # one-hot encoding
    train_features['cp_type'] = train_features.cp_type.map(lambda x: 0 if x == 'trt_cp' else 1)
    test_features['cp_type'] = test_features.cp_type.map(lambda x: 0 if x == 'trt_cp' else 1)

    train_features['cp_dose'] = train_features.cp_dose.map(lambda x: 0 if x == 'D1' else 1)
    test_features['cp_dose'] = test_features.cp_dose.map(lambda x: 0 if x == 'D1' else 1)

    # binning cp_time
    replace_values = {24: 1, 48: 2, 72: 3}
    train_features['cp_time'] = train_features['cp_time'].map(replace_values)
    test_features['cp_time'] = test_features['cp_time'].map(replace_values)

    # now we are going to exclude control samples from the training and testing sets
    # as we know test labels for the control samples in the test set will always predict to 0

    # Datasets for treated and control experiments
    train_features_treated = train_features[train_features['cp_type'] == 0]
    test_features_treated = test_features[test_features['cp_type'] == 0]

    # new, cleaned feature set
    train_features_new = train_features.iloc[:, 1:]

    # remove sig_id from  the targets dataset for computing in a later stage
    train_targets_new = train_targets_scored.iloc[:, 1:]
    test_features_new = test_features_treated.iloc[:, 1:]

    colnames = train_features_new.columns

    # normalize dataset ((x-min)/(max-min))

    normalized_train_features = (train_features_new - train_features_new.min()) / (
            train_features_new.max() - train_features_new.min())
    normalized_test_features = (test_features_new - test_features_new.min()) / (
            test_features_new.max() - test_features_new.min())

    X = normalized_train_features.values
    X_test = normalized_test_features.values

    y = train_targets_new.values

    classifier = LogisticRegression(solver='lbfgs', penalty='l2')

    # baseline
    n_folds = 5
    model_name = "LogReg Baseline"
    model_description = "LogReg all features"
    # baseline_df = evaluate_the_model(X, y, classifier, model_name, model_description, n_folds)
    # print(baseline_df.head())

    clf = OneVsRestClassifier(classifier)

    print("Started the permutate feature importance probe")
    permutate_df = get_permutation_importance(X, y, clf)

    print(permutate_df.head())

    permutate_df.to_csv("/kaggle/working/permutate_feature_importance.csv", index=False)

    print('We are done. That is all, folks!')
    finish_time = dt.datetime.now()
    print("Finished at ", finish_time)
    elapsed = finish_time - start_time
    print("Elapsed time: ", elapsed)
