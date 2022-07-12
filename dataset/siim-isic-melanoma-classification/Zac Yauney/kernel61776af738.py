# %% [code]
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator

# %% [code]
# This is the code that sets up the dataframes to give us our data generators

train_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
df_train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df_test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
df_train['image_num'] = df_train.apply(lambda row: train_dir + row['image_name'] + '.jpg', axis=1)
df_test['image_num'] = df_test.apply(lambda row: test_dir + row['image_name'] + '.jpg', axis=1)
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

# %% [code]
train_generator = datagen.flow_from_dataframe(
dataframe=df_train,
x_col="image_num",
y_col="target",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

valid_generator = datagen.flow_from_dataframe(
dataframe=df_train,
x_col="image_num",
y_col="target",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))



# %% [code]
image_size = (180, 180)
batch_size = 32

#Actually make the model

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = data_augmentation(inputs)

    # Entry block
    #x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
#keras.utils.plot_model(model, show_shapes=True)

# %% [code]
#Train the model

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
#model.fit(
#    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
#)

STEP_SIZE_TRAIN=(100) #train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=(100) #valid_generator.n//valid_generator.batch_size


# %% [code]
#Train the model
model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=epochs, callbacks=callbacks )

# %% [code]
#Save the model
model.save('model.sav')

# %% [code]
#Evaluate the model

model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)