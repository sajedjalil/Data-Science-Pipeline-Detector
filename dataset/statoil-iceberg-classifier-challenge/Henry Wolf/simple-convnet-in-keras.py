import numpy as np
import pandas as pd
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import callbacks
from sklearn.model_selection import train_test_split


def get_callbacks(filepath, patience=2):
    """
    Sets up early stopping and model weight saving.

    :param filepath: location of saved model
    :param patience: number of epochs with no improvement to stop at
    :return:
    """
    es = callbacks.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = callbacks.ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


# create a keras model
def iceberg():
    model = models.Sequential()

    # conv layer one has 32 feature maps, 3x3 kernel, based on 75x75 input image
    model.add(layers.Conv2D(64, (3, 3), input_shape=(75, 75, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())  # default is 2x2
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))  # fully connected = Dense
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))  # sigmoid for binary classification
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    # determine optimizer and loss function
    adam_opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam_opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # display model architecture and return
    model.summary()
    return model


# assign file path and get callbacks
file_path = ".model_weights.hdf5"
callback = get_callbacks(filepath=file_path)

# read in the data
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

# reshape and concatenate the training and testing data
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis],
                          X_band_2[:, :, :, np.newaxis],
                          ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis],
                         X_band_test_2[:, :, :, np.newaxis],
                         ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)


# determine the output column and split the training set for validation
target_train = train['is_iceberg']
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, train_size=0.8)

# train the model
iceberg = iceberg()
iceberg.fit(X_train_cv, y_train_cv,
            batch_size=24,
            epochs=50,
            verbose=1,
            validation_data=(X_valid, y_valid),
            callbacks=callback)

# validate the final model
iceberg.load_weights(filepath=file_path)
score = iceberg.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# test the model and output to file
predicted_test = iceberg.predict_proba(X_test)
submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub.csv', index=False)
