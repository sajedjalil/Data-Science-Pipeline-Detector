import numpy as np
from sklearn.metrics import fbeta_score
from keras import backend as K


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


y_true, y_pred = np.round(np.random.rand(100)), np.random.rand(100)

fbeta_keras = fbeta(K.variable(y_true), K.variable(y_pred)).eval(session=K.get_session())
fbeta_sklearn = fbeta_score(y_true, np.round(y_pred), beta=2)

print('Scores are {:.3f} (sklearn) and {:.3f} (keras)'.format(fbeta_sklearn, fbeta_keras))
