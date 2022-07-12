"""Fine tuning ResNet50 with batch generator."""

import os
from collections import OrderedDict
import itertools as it
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import fbeta_score
import cv2

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import optimizers, Input
from keras import backend as K


def batch_generator(img_dir_path, df, label_map, batch_size=32, shuffle=True,
                    seed=2017, target_image_size=(224, 224),
                    process_target=True, number_of_batches=None,
                    add_seed_shuffle=True, data_gen_args={}, cv2_read=True,
                    preprocess_unit=False):
    """Batch generator for keras model."""
    if number_of_batches is None:
        number_of_batches = np.ceil(df.shape[0] / batch_size)
        print(number_of_batches)

    counter = 0

    if shuffle:
        np.random.seed(seed)
        df = df.sample(frac=1)

    while True:
        if process_target:
            y_batch = []

        idx_start = batch_size * counter
        idx_end = batch_size * (counter + 1)
        x_batch = []

        for f, tags in df.iloc[idx_start:idx_end].values:
            img_path = os.path.join(img_dir_path, '{}.jpg'.format(f))
            if cv2_read:
                img = cv2.imread(img_path)
                x = cv2.resize(img, target_image_size)
            else:
                img = image.load_img(img_path, target_size=target_image_size)
                x = image.img_to_array(img)

            x = np.expand_dims(x, axis=0)
            if preprocess_unit:
                x = preprocess_input(x)

            x_batch.append(x)

            if process_target:
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1
                y_batch.append(targets)

        x_batch = np.concatenate(x_batch)

        datagen = ImageDataGenerator(**data_gen_args)
        datagen.fit(x_batch)
        x_batch = next(datagen.flow(x_batch, shuffle=False))

        counter += 1
        if process_target:
            yield x_batch, np.array(y_batch)
        else:
            yield x_batch

        if (counter == number_of_batches):
            if shuffle:
                if add_seed_shuffle:
                    np.random.seed(seed + 1)
                df = df.sample(frac=1)
            counter = 0


def fbeta(y_true, y_pred, threshold_shift=0):
    """Calculate fbeta score for Keras metrics.

    from https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras
    """
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
    fbeta = ((beta_squared + 1) * (precision * recall) /
             (beta_squared * precision + recall + K.epsilon()))
    return fbeta


def get_train_test():
    """Read and return train test dataframes labels."""
    df_train = pd.read_csv('../input/train_v2.csv')
    df_test = pd.read_csv('../input/sample_submission_v2.csv')
    return df_train, df_test


def get_label_map(df_train):
    """Create and return label_map from df_train['tags']."""
    tags = sorted(set(it.chain(*df_train['tags'].str.split().values)))
    label_map = OrderedDict(zip(tags, range(len(tags))))
    return label_map


def get_ytrain(df_train, label_map):
    """Get y_train."""
    y_train = np.zeros((len(df_train), 17))
    for i, (_, tags) in enumerate(df_train.values):
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_train[i] = targets
    return y_train


def get_model(learn_rate=0.001, decay=0.1, trainable=False):
    """Get ResNet50 model."""
    main_model = ResNet50(include_top=False, input_tensor=Input(
        shape=(224, 224, 3)))

    main_model.trainable = trainable
    for layer in main_model.layers:
        layer.trainable = trainable

    top_model = Sequential()
    top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
    # top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(17, activation='sigmoid'))
    model = Model(inputs=main_model.input,
                  outputs=top_model(main_model.output))

    opt = optimizers.Adam(lr=learn_rate, decay=decay)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[fbeta, 'accuracy'])

    return model


def train_model(model, df_train, train_dir, fit_index, val_index, label_map,
                model_weights_path, epochs=50, batch_size=32,
                data_gen_args_fit={}, data_gen_args_val={}, seed=2017):
    """Train model with generators."""
    print('Training model...')
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1),
                 ModelCheckpoint(model_weights_path, monitor='val_loss',
                                 save_best_only=True, verbose=0)]

    steps_per_epoch_fit = np.ceil(len(fit_index) / batch_size)
    steps_per_epoch_val = np.ceil(len(val_index) / batch_size)

    fit_generator = batch_generator(train_dir,
                                    df_train.iloc[fit_index],
                                    label_map,
                                    batch_size=batch_size,
                                    number_of_batches=steps_per_epoch_fit,
                                    data_gen_args=data_gen_args_fit,
                                    seed=seed)

    val_generator = batch_generator(train_dir,
                                    df_train.iloc[val_index],
                                    label_map,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    number_of_batches=steps_per_epoch_val,
                                    data_gen_args=data_gen_args_val)
    try:
        model.fit_generator(generator=fit_generator,
                            steps_per_epoch=steps_per_epoch_fit,
                            epochs=epochs,
                            verbose=1,
                            validation_data=val_generator,
                            validation_steps=steps_per_epoch_val,
                            callbacks=callbacks)
    except KeyboardInterrupt:
        pass
    return model


def evaluate_model(model, df_train, train_dir, y_train, val_index, label_map,
                   batch_size=32, return_preds=False, data_gen_args_val={}):
    """Evaluate model."""
    print('Evaluating model...')
    steps_per_epoch_val = np.ceil(len(val_index) / batch_size)

    val_generator = batch_generator(train_dir,
                                    df_train.iloc[val_index],
                                    label_map,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    number_of_batches=steps_per_epoch_val,
                                    data_gen_args=data_gen_args_val)

    pred_val = model.predict_generator(generator=val_generator,
                                       steps=steps_per_epoch_val,
                                       verbose=1)

    f_valid_beta_score = fbeta_score(y_train[val_index],
                                     np.array(pred_val) > 0.2,
                                     beta=2, average='samples')
    print('Fbeta score:', f_valid_beta_score)
    if return_preds:
        return f_valid_beta_score, pred_val
    return f_valid_beta_score


def predict_model(model, df_test, test_dir, label_map, batch_size=32,
                  data_gen_args_test={}):
    """Make predictions."""
    print('Making predicions...')
    steps_per_epoch_test = np.ceil(len(df_test) / batch_size)
    test_generator = batch_generator(test_dir, df_test, label_map,
                                     batch_size=batch_size,
                                     process_target=False,
                                     shuffle=False,
                                     number_of_batches=steps_per_epoch_test,
                                     data_gen_args=data_gen_args_test)

    predictions = model.predict_generator(generator=test_generator,
                                          steps=steps_per_epoch_test,
                                          verbose=1)
    return predictions


def get_data_from_generator(generator, steps_per_epoch):
    X_full = []
    y_full = []
    for i, data in enumerate(generator):
        if i == steps_per_epoch:
            break
        X_full.append(data[0])
        y_full.append(data[1])

    X_full = np.concatenate(X_full)
    y_full = np.concatenate(y_full)
    return X_full, y_full


def create_submission(predictions, df_test, label_map, f_valid_beta_score):
    """Create submission with threshold 0.2."""
    print('Creating submission...')
    result = pd.DataFrame(predictions, columns=list(label_map.keys()))

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.iloc[[i]]
        a = a.apply(lambda x: x > 0.2, axis=1)
        a = a.transpose()
        a = a.loc[a[i]]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df_preds = df_test.copy()
    df_preds['tags'] = preds
    df_preds.to_csv(
        '../submissions/submission_keras_resnet50_{:.6f}_{}.csv'
        .format(f_valid_beta_score, dt.datetime.now().strftime('%Y_%m_%d_%H')),
        index=False)


def main():
    """Main function to train, validate, predicting and create submission."""

    # To use main GTX 1070
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    random_state = 25
    train_dir = '../input/train-jpg'
    test_dir = '../input/test-jpg'

    learn_rate = 0.001
    decay = 0.0001
    batch_size = 32
    epochs = 100
    trainable = True
    model_weights_path = os.path.join('model_weights', 'weights_resnet50.h5')

    data_gen_args_train = dict(
        rotation_range=90,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1 / 255)

    data_gen_args_val = dict(rescale=1 / 255)
    data_gen_args_test = dict(rescale=1 / 255)

    df_train, df_test = get_train_test()
    label_map = get_label_map(df_train)
    y_train = get_ytrain(df_train, label_map)

    fit_index, val_index = train_test_split(df_train.index, test_size=0.1,
                                            random_state=random_state)

    model = get_model(learn_rate=learn_rate, decay=decay, trainable=trainable)
    model = train_model(model, df_train, train_dir, fit_index, val_index,
                        label_map, model_weights_path, epochs=epochs,
                        batch_size=batch_size,
                        data_gen_args_fit=data_gen_args_train,
                        data_gen_args_val=data_gen_args_val,
                        seed=random_state)

    f_valid_beta_score, pred_val = evaluate_model(
        model, df_train,
        train_dir,
        y_train,
        val_index, label_map,
        batch_size=batch_size,
        return_preds=True,
        data_gen_args_val=data_gen_args_val)

    predictions = predict_model(model, df_test, test_dir, label_map,
                                batch_size=batch_size,
                                data_gen_args_test=data_gen_args_test)

    create_submission(predictions, df_test, label_map, f_valid_beta_score)


def main_kfold():
    """
    Function to train, validate, predicting and create submission with KFold.
    """

    # To use main GTX 1070
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    random_state = 20
    train_dir = '../input/train-jpg'
    test_dir = '../input/test-jpg'

    learn_rate = 0.001
    decay = 0.0001
    batch_size = 32
    epochs = 70
    nfolds = 5
    trainable = True

    df_train, df_test = get_train_test()

    label_map = get_label_map(df_train)
    y_train = get_ytrain(df_train, label_map)

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    data_gen_args_train = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1 / 255)

    data_gen_args_val = dict(rescale=1 / 255)
    data_gen_args_test = dict(rescale=1 / 255)

    f_scores = []
    preds = []
    pred_vals = []

    for num_fold, (fit_index, val_index) in enumerate(kf.split(df_train)):
        kfold_weights_path = os.path.join(
            'model_weights', 'weights_resnet50_{}.h5'.format(num_fold))

        model = get_model(learn_rate=learn_rate, decay=decay,
                          trainable=trainable)
        model = train_model(model, df_train, train_dir, fit_index, val_index,
                            label_map, kfold_weights_path, epochs=epochs,
                            batch_size=batch_size,
                            data_gen_args_fit=data_gen_args_train,
                            data_gen_args_val=data_gen_args_val,
                            seed=random_state)

        f_valid_beta_score, pred_val = evaluate_model(
            model, df_train,
            train_dir,
            y_train,
            val_index, label_map,
            batch_size=batch_size,
            return_preds=True,
            data_gen_args_val=data_gen_args_val)

        f_scores.append(f_valid_beta_score)
        pred_vals.append(pred_val)

        predictions = predict_model(model, df_test, test_dir, label_map,
                                    batch_size=batch_size,
                                    data_gen_args_test=data_gen_args_test)
        preds.append(predictions)


    f_valid_beta_score = np.mean(f_scores)
    create_submission(preds_final_test, df_test, label_map, f_valid_beta_score)

if __name__ == '__main__':
    # To use main GTX 1070
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main_kfold()
