import pandas as pd
import numpy as np

data_dir = '../input/'

def load_data(data_dir):
    train = pd.read_json(data_dir + 'train.json')
    test = pd.read_json(data_dir + 'test.json')
    #Fill 'na' angles with mode
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    test.inc_angle = test.inc_angle.replace('na', 0)
    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
    return train, test

train, test = load_data(data_dir)

def color_composite(data):
    import cv2
    w,h = 197,197
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        #Add in to resize for resnet50 use 197 x 197
        rgb = cv2.resize(rgb, (w,h)).astype(np.float32)
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

rgb_train = color_composite(train)
rgb_test = color_composite(test)

y_train = np.array(train['is_iceberg'])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(rgb_train, y_train, random_state=420, train_size=0.75)

from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.models import Model

#Create the model
#model = simple_cnn()
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(197,197,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
#for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True

from keras.optimizers import SGD
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
batch_size = 3
#Lets define the image transormations that we want
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.2,
                         rotation_range=40)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_one_input(X1, y):
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=420)
    while True:
        X1i = genX1.next()
        yield X1i[0], X1i[1]

#Finally create out generator
gen_flow = gen_flow_for_one_input(X_train, y_train)

from keras.callbacks import EarlyStopping, ModelCheckpoint
epochs_to_wait_for_improve = 50
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
checkpoint_callback = ModelCheckpoint('BestKerasModelResNet50.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#fit the model
model.fit_generator(gen_flow, validation_data=(X_valid, y_valid), steps_per_epoch=int(np.ceil(len(X_train)/batch_size)), epochs=500, verbose=1, callbacks=[early_stopping_callback, checkpoint_callback])

# Predict on test data
from keras.models import load_model
model = load_model('BestKerasModelResNet50.h5')
test_predictions = model.predict(rgb_test)

# Create .csv
pred_df = test[['id']].copy()
pred_df['is_iceberg'] = test_predictions
pred_df.to_csv('predictionsResNet50Layers15.csv', index = False)