from collections import Counter

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.applications.vgg19 import VGG19

import cv2
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score

input_size = 128
input_channels = 3

epochs = 50
batch_size = 128

n_folds = 5

training = True

ensemble_voting = False  # If True, use voting for model ensemble, otherwise use averaging

df_train_data = pd.read_csv('input/train_v2.csv')
df_test_data = pd.read_csv('input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

fold_count = 0

y_full_test = []
thres_sum = np.zeros(17, np.float32)

for train_index, test_index in kf.split(df_train_data):

    fold_count += 1
    print('Fold ', fold_count)


    def transformations(src, choice):
        if choice == 0:
            # Rotate 90
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        if choice == 1:
            # Rotate 90 and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            src = cv2.flip(src, flipCode=1)
        if choice == 2:
            # Rotate 180
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        if choice == 3:
            # Rotate 180 and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
            src = cv2.flip(src, flipCode=1)
        if choice == 4:
            # Rotate 90 counter-clockwise
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        if choice == 5:
            # Rotate 90 counter-clockwise and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            src = cv2.flip(src, flipCode=1)
        return src

    df_valid = df_train_data.ix[test_index]
    print('Validating on {} samples'.format(len(df_valid)))


    def valid_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for f, tags in df_valid_batch.values:
                    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, np.random.randint(6))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    df_train = df_train_data.ix[train_index]
    if training:
        print('Training on {} samples'.format(len(df_train)))


    def train_generator():
        while True:
            for start in range(0, len(df_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_train))
                df_train_batch = df_train[start:end]
                for f, tags in df_train_batch.values:
                    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, np.random.randint(6))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    base_model = VGG19(include_top=False,
                       weights='imagenet',
                       input_shape=(input_size, input_size, input_channels))

    model = Sequential()
    # Batchnorm input
    model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
    # Base model
    model.add(base_model)
    # Classifier
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))

    opt = Adam(lr=1e-4)

    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=4,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=2,
                                   cooldown=2,
                                   verbose=1),
                 ModelCheckpoint(filepath='weights/best_weights.fold_' + str(fold_count) + '.hdf5',
                                 save_best_only=True,
                                 save_weights_only=True)]

    if training:
        model.fit_generator(generator=train_generator(),
                            steps_per_epoch=(len(df_train) // batch_size) + 1,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=valid_generator(),
                            validation_steps=(len(df_valid) // batch_size) + 1)


    def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
        def mf(x):
            p2 = np.zeros_like(p)
            for i in range(17):
                p2[:, i] = (p[:, i] > x[i]).astype(np.int)
            score = fbeta_score(y, p2, beta=2, average='samples')
            return score

        x = [0.2] * 17
        for i in range(17):
            best_i2 = 0
            best_score = 0
            for i2 in range(resolution):
                i2 /= float(resolution)
                x[i] = i2
                score = mf(x)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)
        return x


    # Load best weights
    model.load_weights(filepath='weights/best_weights.fold_' + str(fold_count) + '.hdf5')

    p_valid = model.predict_generator(generator=valid_generator(),
                                      steps=(len(df_valid) // batch_size) + 1)

    y_valid = []
    for f, tags in df_valid.values:
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_valid.append(targets)
    y_valid = np.array(y_valid, np.uint8)

    # Find optimal f2 thresholds for local validation set
    thres = optimise_f2_thresholds(y_valid, p_valid, verbose=False)

    print('F2 = {}'.format(fbeta_score(y_valid, np.array(p_valid) > thres, beta=2, average='samples')))

    thres_sum += np.array(thres, np.float32)


    def test_generator(transformation):
        while True:
            for start in range(0, len(df_test_data), batch_size):
                x_batch = []
                end = min(start + batch_size, len(df_test_data))
                df_test_batch = df_test_data[start:end]
                for f, tags in df_test_batch.values:
                    img = cv2.imread('input/test-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, transformation)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                yield x_batch

    # 6-fold TTA
    p_full_test = []
    for i in range(6):
        p_test = model.predict_generator(generator=test_generator(transformation=i),
                                         steps=(len(df_test_data) // batch_size) + 1)
        p_full_test.append(p_test)

    p_test = np.array(p_full_test[0])
    for i in range(1, 6):
        p_test += np.array(p_full_test[i])
    p_test /= 6

    y_full_test.append(p_test)

result = np.array(y_full_test[0])
if ensemble_voting:
    for f in range(len(y_full_test[0])):  # For each file
        for tag in range(17):  # For each tag
            preds = []
            for fold in range(n_folds):  # For each fold
                preds.append(y_full_test[fold][f][tag])
            pred = Counter(preds).most_common(1)[0][0]  # Most common tag prediction among folds
            result[f][tag] = pred
else:
    for fold in range(1, n_folds):
        result += np.array(y_full_test[fold])
    result /= n_folds
result = pd.DataFrame(result, columns=labels)

preds = []
thres = (thres_sum / n_folds).tolist()

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)
