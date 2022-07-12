import os
import gc
import shutil
import time
import warnings

import cv2
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.initializers import he_normal
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, ReLU)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

id_label = pd.read_csv("../input/train_labels.csv")
id_label["file_name"] = id_label["id"].apply(lambda x: x + ".tif")

sample_size = 80000
id_label_pos = id_label.loc[id_label["label"] == 1, :].sample(n=sample_size)
id_label_neg = id_label.loc[id_label["label"] == 0, :].sample(n=sample_size)
id_label = pd.concat([id_label_pos, id_label_neg], ignore_index=True)
del id_label_pos, id_label_neg

id_label_train, id_label_val = train_test_split(id_label, test_size=0.1, stratify=id_label["label"])
id_label_train.reset_index(drop=True, inplace=True)
id_label_val.reset_index(drop=True, inplace=True)
len_train = id_label_train.shape[0]
len_val = id_label_val.shape[0]
print("Training data size: {}.".format(len_train))
print("Validation data size: {}.".format(len_val))
del id_label
gc.collect()

img_gen_params = {
    "rescale": 1.0 / 255,
    "samplewise_center": True,
    "samplewise_std_normalization": True,
    "horizontal_flip": True,
    "vertical_flip": True
}
img_gen = ImageDataGenerator(**img_gen_params)

IMAGE_SHAPE = (96, 96, 3)
path_train = "../input/train"
batch_size = 32

img_flow_params_train = {
    "dataframe": id_label_train,
    "directory": path_train,
    "x_col": "file_name",
    "y_col": "label",
    "has_ext": True,
    "target_size": IMAGE_SHAPE[:2],
    "batch_size": batch_size
}
img_flow_train = img_gen.flow_from_dataframe(**img_flow_params_train)

img_flow_params_val = {
    "dataframe": id_label_val,
    "directory": path_train,
    "x_col": "file_name",
    "y_col": "label",
    "has_ext": True,
    "target_size": IMAGE_SHAPE[:2],
    "batch_size": 1,
    "shuffle": False
}
img_flow_val = img_gen.flow_from_dataframe(**img_flow_params_val)

kernel_size = (5, 5)
filters = (32, 64, 128)
drop_prob_conv = 0.3
drop_prob_dense = 0.3

model = keras.models.Sequential()

model.add(Conv2D(filters[0], kernel_size, padding="same", kernel_initializer=he_normal(), input_shape=IMAGE_SHAPE))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(filters[0], kernel_size, padding="same", kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob_conv))

model.add(Conv2D(filters[1], kernel_size, padding="same", kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(filters[1], kernel_size, padding="same", kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob_conv))

model.add(Conv2D(filters[2], kernel_size, padding="same", kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(filters[2], kernel_size, padding="same", kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob_conv))

model.add(Flatten())
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(drop_prob_dense))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(drop_prob_dense))
model.add(Dense(2, activation="softmax"))
model.compile(Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

lr_decay_params = {
    "monitor": "val_acc",
    "factor": 0.5,
    "patience": 1,
    "min_lr": 1e-5
}
lr_decay = ReduceLROnPlateau(**lr_decay_params)

early_stopping = EarlyStopping(monitor="val_acc", patience=3, verbose=1)

fit_params = {
    "generator": img_flow_train,
    "steps_per_epoch": len_train // batch_size,
    "epochs": 10,
    "verbose": 1,
    "validation_data": img_flow_val,
    "validation_steps": len_val,
    "callbacks": [lr_decay, early_stopping]
}
print("Training the model...")
model.fit_generator(**fit_params)
print("Done!")

y_val_pred = model.predict_generator(img_flow_val, steps=len_val)[:, 1]
y_val_true = img_flow_val.classes
acc_val = np.equal((y_val_pred > 0.5).astype("int"), y_val_true).sum() / y_val_pred.shape[0]
print("Validation accuracy: {:.3f}.".format(acc_val))

fpr, tpr, thresholds = roc_curve(y_val_true, y_val_pred)
auc_val = auc(fpr, tpr)
print("Validation AUC: {:.3f}.".format(auc_val))

plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr, label="ACC={:.4F}, AUC={:.4f}".format(acc_val, auc_val))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
plt.show()
plt.savefig("roc_curve.png")

del img_flow_params_train, img_flow_params_val
gc.collect()

path_test = "../input/test"
test_files = [x + ".tif" for x in os.listdir(path_test)]
test_files = pd.DataFrame({"file_name": test_files})
len_test = len(test_files)

img_flow_params_test = {
    "dataframe": test_files,
    "directory": path_test,
    "x_col": "file_name",
    "has_ext": True,
    "class_mode": None,
    "target_size": IMAGE_SHAPE[:2],
    "batch_size": 1,
    "shuffle": False
}
img_flow_test = img_gen.flow_from_dataframe(**img_flow_params_test)

pred_params = {
    "generator": img_flow_test,
    "steps": len_test,
    "verbose": 1
}

test_run = 10
y_pred = pd.read_csv("../input/sample_submission.csv")
y_pred["id"] = y_pred["id"].apply(lambda x: x + ".tif")

for i in range(test_run):

    preds = model.predict_generator(**pred_params)[:, 1]
    file_names = img_flow_test.filenames
    ind_y_pred = pd.DataFrame({"id": file_names, "label" + str(i + 1): preds})
    y_pred = y_pred.merge(ind_y_pred, on="id")

y_pred["label"] = y_pred[["label" + str(i + 1) for i in range(test_run)]].mean(axis=1)
y_pred.drop(["label" + str(i + 1) for i in range(test_run)], axis=1, inplace=True)
y_pred["id"] = y_pred["id"].apply(lambda x: x.split(".")[0])

y_pred.to_csv("y_pred.csv", index=False)