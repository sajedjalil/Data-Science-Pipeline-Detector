import pandas as pd 
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from typing import List
from collections import Counter

# EfficientNet
from tensorflow.keras.applications import EfficientNetB4, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Layers
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization

# Compiling and Callbacks
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# Sklearn
from sklearn.metrics import roc_auc_score
#-----------------------------------------------------------------------------------------------------
# Competition Directory
comp_dir="/kaggle/input/ranzcr-clip-catheter-line-classification/"

# Get Training Data Labels
df_train=pd.read_csv(comp_dir+"train.csv").sample(frac=1).reset_index(drop=True)

# Get Training/Testing Data Paths
test_files = os.listdir(comp_dir+"test")      

df_test = pd.DataFrame({"StudyInstanceUID": test_files})

image_size = 380
batch_size = 16
num_epochs = 14
learn_rate = 5e-01
df_train.StudyInstanceUID += ".jpg"

# Train-Val = [:21000], [21000:], test_files
# Train-Val-Test (for tuning model) = [:18000], [18000:24000], [24000:]

#-----------------------------------------------------------------------------------------------------
label_cols=df_train.columns.tolist()
label_cols.remove("StudyInstanceUID")
label_cols.remove("PatientID")
datagen=ImageDataGenerator()
test_datagen=ImageDataGenerator()

train_generator=datagen.flow_from_dataframe(
    dataframe=df_train[:18000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df_train[18000:24000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_train[24000:],
    directory=comp_dir+"train",    # Change this
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")
#-----------------------------------------------------------------------------------------------------
# base_model = ResNet50(include_top=False,
#                       weights=None,
#                       input_shape=(image_size, image_size, 3))
# base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

# for layer in base_model.layers[:-6]:
#     layer.trainable = False

base_model = EfficientNetB4(include_top=False,
                      weights="imagenet",
                      input_shape=(image_size, image_size, 3))
# base_model.load_weights("../input/tfkeras-22-pretrained-and-vanilla-efficientnet/TF2.2_EfficientNetB4_NoTop_ImageNet.h5", by_name=True)

for layer in base_model.layers[:-6]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

#-----------------------------------------------------------------------------------------------------
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)

output1 = Dense(1, activation = 'sigmoid')(x)
output2 = Dense(1, activation = 'sigmoid')(x)
output3 = Dense(1, activation = 'sigmoid')(x)
output4 = Dense(1, activation = 'sigmoid')(x)
output5 = Dense(1, activation = 'sigmoid')(x)
output6 = Dense(1, activation = 'sigmoid')(x)
output7 = Dense(1, activation = 'sigmoid')(x)
output8 = Dense(1, activation = 'sigmoid')(x)
output9 = Dense(1, activation = 'sigmoid')(x)
output10 = Dense(1, activation = 'sigmoid')(x)
output11 = Dense(1, activation = 'sigmoid')(x)

model = Model(inp,[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11])

sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(11)],
              metrics = ["accuracy"])


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(11)])


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size


# Fit Model
history = model.fit(generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(valid_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=num_epochs,
                    verbose=2)

test_generator.reset()

# Get Prediction
pred = model.predict(test_generator,
                     steps=STEP_SIZE_TEST,
                     verbose=1)

# Get AUC Score    # TEST THIS
y_pred = np.transpose(np.squeeze(pred))
y_true = df_train.loc[24000:, label_cols].to_numpy()
aucs = roc_auc_score(y_true, y_pred, average=None)

print("AUC Scores: ", aucs)
print("Average AUC: ", np.mean(aucs))

# Create Submission df
df_submission = pd.DataFrame()
df_submission["StudyInstanceUID"] = test_files
df_submission["StudyInstanceUID"] = df_submission["StudyInstanceUID"].map(lambda x: x.replace(".jpg",""))

df_preds = pd.DataFrame(np.squeeze(pred)).transpose()
df_preds = df_preds.rename(columns=dict(zip([i for i in range(11)], label_cols)))

df_submission = pd.concat([df_submission, df_preds], axis=1)
df_submission.to_csv("submission.csv", index=False)

epochs = range(1,num_epochs)
plt.plot(history.history['loss'], label='Training Set')
plt.plot(history.history['val_loss'], label='Validation Data)')
plt.title('Training and Validation loss')
plt.ylabel('MAE')
plt.xlabel('Num Epochs')
plt.legend(loc="upper left")
plt.show()
plt.savefig("loss.png")