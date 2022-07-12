import pandas as pd
import os
import os.path as osp
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load Yandex/CERN's evaluation Python script from the input data
exec(open("../input/evaluation.py").read())


def read_training_data(data_folder):
    """
    Read the training data file & separate features from signal column
    :param data_folder: path to folder containing the challenge datasets
    :return (pandas.DataFrame, pandas.Series) : (x, y) ie features dataframe and label series.
    """
    full_train = pd.read_csv(data_folder + 'training.csv', index_col='id')
    feature_cols = full_train.columns.tolist()
    feature_cols.remove("signal")
    return full_train[feature_cols], full_train['signal']


def split_data(x, y):
    """
    split the training dataset: set apart 10% of data for validation
    x_train, x_valid, y_train, y_valid = splitted_set
    :param x: feature columns
    :param y: signal column
    :return:
    """
    splitted_set = train_test_split(x,  # training features
                                    y,  # training labels
                                    test_size=0.1,
                                    random_state=0)

    return splitted_set


def train_model(x_train, y_train):
    """
    Change this function to get the best model you can
    :param x_train: training features
    :param y_train: training labels ('signal' column)

    :return: a trained model & the names of the variable columns it takes as input
    """

    variables = ['LifeTime', 'FlightDistance', 'pt', ]
    baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                          min_samples_leaf=10, max_depth=7, random_state=11)
    baseline.fit(x_train[variables], y_train)
    return baseline, variables


def _check_agreement(model, variables, data_folder):
    check_agreement = pd.read_csv(data_folder + 'check_agreement.csv', index_col='id')
    agreement_prob = model.predict_proba(check_agreement[variables])[:, 1]

    ks = compute_ks(
        agreement_prob[check_agreement['signal'].values == 0],
        agreement_prob[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print('KS metric', ks, ks < 0.09)
    return ks < 0.09


def _check_correlation(model, variables, data_folder):
    check_correlation = pd.read_csv(data_folder + 'check_correlation.csv', index_col='id')
    correlation_probs = model.predict_proba(check_correlation[variables])[:, 1]
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print('CvM metric', cvm, cvm < 0.002)
    return cvm < 0.002


def check_model(model, variables, data_folder):
    return _check_agreement(model, variables, data_folder) and _check_correlation(model, variables, data_folder)


def compute_AUC_on_valid(x, y, model, variables=None):
    samples_for_eval = x['min_ANNmuon'] > 0.4
    train_eval = x[samples_for_eval]
    train_probs = model.predict_proba(train_eval[variables])[:, 1]
    AUC = roc_auc_truncated(y[samples_for_eval], train_probs)
    print('AUC', AUC)
    return AUC

def prepare_submission_data(model, variables, x_full_train, y_full_train, data_folder):
    """
    Prepare a model with optimized parameters for submission:
        - fit on full training dataset
        - check agreement & correlation
        - compute predictions
    :param model: to fit
    :param variables: to use for training
    :param x_full_train: full training dataset
    :param y_full_train: corresponding labels
    :param data_folder: path to folder with test.csv file.
    :return: pandas.Series with predictions
    """
    # re-fit on full training data
    print('Training on full dataset for submission...')
    model.fit(x_full_train[variables], y_full_train)

    # re-check validity
    print('Re-check...')
    if not check_model(model, variables, data_folder):
        raise Exception("Sorry, your model doesn't pass the checks")

    # compute test predictions
    print('Computing predictions...')
    test = pd.read_csv(data_folder + 'test.csv', index_col='id')
    result = pd.DataFrame({'id': test.index})
    result['prediction'] = model.predict_proba(test[variables])[:, 1]
    return result


def generate_submission(model, variables, validation_auc, data_folder,
                        x_full_train, y_full_train,
                        output_folder, force_submit=False):
    """
    Prepare submission and keep track of results in a 'submissions.csv' file.
    """
    submission_list_file = osp.join(output_folder, 'submissions.csv')
    # not saving here
    # if not osp.exists(output_folder):
    #     os.makedirs(output_folder)
    if osp.exists(submission_list_file):
        submissions = pd.read_csv(submission_list_file, index_col=0)
    else:
        submissions = pd.DataFrame(columns=['id', 'auc', ])

    id_submission = 0
    if len(submissions) > 0:
        last_submission = submissions.tail(1)
        if last_submission['auc'].ix[0] > validation_auc:
            print("This doesn't improve on the previous submission.")
            if not force_submit:
                return
        id_submission = int(last_submission['id'].ix[0] + 1)

    submission_name = osp.join(output_folder, 'test_{0}.csv'.format(id_submission))
    result = prepare_submission_data(model, variables, x_full_train, y_full_train, data_folder)
    # not saving here
    #result.to_csv(submission_name, index=False, sep=',')

    submissions.ix[pd.pnow('s')] = [id_submission, validation_auc]
    # not saving here
    # submissions.to_csv(submission_list_file)
    print('Predictions written to {0}, you can submit!'.format(submission_name))


if __name__ == "__main__":
    folder = '../input/'
    x, y = read_training_data(folder)
    x_train, x_valid, y_train, y_valid = split_data(x, y)
    model_fitted, vars = train_model(x_train, y_train)
    if not check_model(model_fitted, vars, folder):
        raise Exception("Sorry, your model doesn't pass the checks")
    curr_auc = compute_AUC_on_valid(x_valid, y_valid, model_fitted, vars)
    generate_submission(model_fitted, vars,
                        curr_auc, folder,
                        x, y,
                        output_folder='outputs',
                        force_submit=False)
