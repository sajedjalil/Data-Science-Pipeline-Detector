import keras
from sklearn.metrics import fbeta_score

class Fbeta(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.fbeta = []
    def on_epoch_end(self, epoch, logs ={}):
        p_valid = self.model.predict(self.validation_data[0])
        y_val = self.validation_data[1]
        f_beta = fbeta_score(y_val, np.array(p_valid) > 0.2, beta=2, average='samples')
        self.fbeta.append(f_beta)
        return

fbeta = Fbeta()

print(fbeta.fbeta)