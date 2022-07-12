import numpy as np 
import pandas as pd 
from keras.callbacks import Callback
from sklearn.metrics import fbeta_score
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.test_utils import get_test_data, keras_test



""" F2 metric implementation for Keras models. Inspired from this Medium
article: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
Before we start, you might ask: this is a classic metric, isn't it already 
implemented in Keras? 
The answer is: it used to be. It has been removed since. Why?
Well, since metrics are computed per batch, this metric was confusing 
(should be computed globally over all the samples rather than over a mini-batch).
For more details, check this: https://github.com/keras-team/keras/issues/5794.
In this short code example, the F2 metric will only be called at the end of 
each epoch making it more useful (and correct).
"""

# Notice that since this competition has an unbalanced positive class
# (fewer ), a beta of 2 is used (thus the F2 score). This favors recall
# (i.e. capacity of the network to find positive classes). 

# Some default constants

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
DEFAULT_THRESHOLDS = np.linspace(START, END, N_STEPS)
DEFAULT_BETA = 1
DEFAULT_LOGS = {}
FBETA_METRIC_NAME = "val_fbeta"

# Some unit test constants
input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20
SEED = 42
TEST_BETA = 2
EPOCHS = 5




# Notice that this callback only works with Keras 2.0.0


class FBetaMetricCallback(Callback):

    def __init__(self, beta=DEFAULT_BETA, thresholds=DEFAULT_THRESHOLDS):
        self.beta = beta
        self.thresholds = thresholds
        # Will be initialized when the training starts
        self.val_fbeta = None

    def on_train_begin(self, logs=DEFAULT_LOGS):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_fbeta = []

    def _score_per_threshold(self, predictions, targets, threshold):
        """ Compute the Fbeta score per threshold.
        """
        # Notice that here I am using the sklearn fbeta_score function.
        # You can read more about it here:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        thresholded_predictions = (predictions > threshold).astype(int)
        return fbeta_score(targets, thresholded_predictions, beta=self.beta)

    def on_epoch_end(self, epoch, logs=DEFAULT_LOGS):
        val_predictions = self.model.predict(self.validation_data[0])
        val_targets = self.validation_data[1]
        _val_fbeta = np.mean([self._score_per_threshold(val_predictions,
                                                        val_targets, threshold)
                              for threshold in self.thresholds])
        self.val_fbeta.append(_val_fbeta)
        print("Current F{} metric is: {}".format(str(self.beta), str(_val_fbeta)))
        return

    def on_train_end(self, logs=DEFAULT_LOGS):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history[FBETA_METRIC_NAME] = self.val_fbeta

"""
Here is how to use this metric: 
Create a model and add the FBetaMetricCallback callback (with beta set to 2).
f2_metric_callback = FBetaMetricCallback(beta=2)
callbacks = [f2_metric_callback]
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    nb_epoch=10, batch_size=64, callbacks=callbacks)
print(history.history.val_fbeta)
"""


# Here is a unit test


@keras_test
def test_fbeta_metric_callback():
    np.random.seed(SEED)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_classes)
    # Simple classification model definition
    # TODO: Refactor this into a function.
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    fbeta_metric_callback = FBetaMetricCallback(beta=TEST_BETA)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[fbeta_metric_callback], 
                        epochs=EPOCHS)
    assert fbeta_metric_callback.val_fbeta is not None
    assert FBETA_METRIC_NAME in history.history.keys()
    assert history.history[FBETA_METRIC_NAME] is not None
    assert len(history.history[FBETA_METRIC_NAME]) == EPOCHS
    
    
if __name__ == "__main__":
    test_fbeta_metric_callback()
 
 # Enjoy!