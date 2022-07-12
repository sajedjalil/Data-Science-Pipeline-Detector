#part of the code is from https://github.com/maciejkula/triplet_recommendations_keras

from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalMaxPooling2D
from keras.models import Model
import glob
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors  

class sample_gen(object):
    def __init__(self, file_class_mapping, other_class = "new_whale"):
        self.file_class_mapping= file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            if class_ == other_class:
                self.list_other_class.append(file)
            else:
                self.class_to_list_files[class_].append(file)

        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes= range(len(self.list_classes))
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_weight/np.sum(self.class_weight)

    def get_sample(self):
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)
        positive_example_1, positive_example_2 = \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]],\
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]


        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] == \
                self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example, positive_example_2




batch_size = 8
input_shape = (256, 256)
base_path = "../input/train/"
def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def get_base_model():
    latent_dim = 50
    base_model = ResNet50(include_top=False) # use weights='imagenet' locally

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    dense_1 = Dense(latent_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model

def build_model():
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

    print(model.summary())

    return model


model_name = "triplet_model"

file_path = model_name + "weights.best.hdf5"



def build_inference_model(weight_path=file_path):
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

    model.load_weights(weight_path)

    inference_model = Model(base_model.get_input_at(0), output=base_model.get_output_at(0))
    inference_model.compile(loss="mse", optimizer=Adam(0.000001))
    print(inference_model.summary())

    return inference_model

def read_and_resize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize(input_shape)
    im_array = np.array(im, dtype="uint8")[..., ::-1]
    return np.array(im_array / (np.max(im_array)+ 0.001), dtype="float32")


def augment(im_array):
    if np.random.uniform(0, 1) > 0.9:
        im_array = np.fliplr(im_array)
    return im_array

def gen(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            positive_example_1_img, negative_example_img, positive_example_2_img = read_and_resize(base_path+positive_example_1), \
                                                                       read_and_resize(base_path+negative_example), \
                                                                       read_and_resize(base_path+positive_example_2)

            positive_example_1_img, negative_example_img, positive_example_2_img = augment(positive_example_1_img), \
                                                                                   augment(negative_example_img), \
                                                                                   augment(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        list_positive_examples_1 = np.array(list_positive_examples_1)
        list_negative_examples = np.array(list_negative_examples)
        list_positive_examples_2 = np.array(list_positive_examples_2)
        yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(batch_size)



num_epochs = 10

# Read data
data = pd.read_csv('../input/train.csv')
train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=1337)
file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}
file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}
train_gen = sample_gen(file_id_mapping_train)
test_gen = sample_gen(file_id_mapping_test)



# Prepare the test triplets

model = build_model()



#model.load_weights(file_path)

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [checkpoint, early]  # early

history = model.fit_generator(gen(train_gen), validation_data=gen(test_gen), epochs=3, verbose=2, workers=4, use_multiprocessing=True,
                              callbacks=callbacks_list, steps_per_epoch=300, validation_steps=30)
                              
                              
model_name = "triplet_loss"
def data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = read_and_resize(path)
        imgs.append(img)
        fnames.append(os.path.basename(path))
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()

data = pd.read_csv('../input/train.csv')

file_id_mapping = {k: v for k, v in zip(data.Image.values, data.Id.values)}

inference_model = build_inference_model()

train_files = glob.glob("../input/train/*.jpg")
test_files = glob.glob("../input/test/*.jpg")

train_preds = []
train_file_names = []
i = 1
for fnames, imgs in data_generator(train_files, batch=32):
    print(i*32/len(train_files)*100)
    i += 1
    predicts = inference_model.predict(imgs)
    predicts = predicts.tolist()
    train_preds += predicts
    train_file_names += fnames

train_preds = np.array(train_preds)

test_preds = []
test_file_names = []
i = 1
for fnames, imgs in data_generator(test_files, batch=32):
    print(i * 32 / len(test_files) * 100)
    i += 1
    predicts = inference_model.predict(imgs)
    predicts = predicts.tolist()
    test_preds += predicts
    test_file_names += fnames

test_preds = np.array(test_preds)

neigh = NearestNeighbors(n_neighbors=6)
neigh.fit(train_preds)
#distances, neighbors = neigh.kneighbors(train_preds)

#print(distances, neighbors)

distances_test, neighbors_test = neigh.kneighbors(test_preds)

distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

preds_str = []

for filepath, distance, neighbour_ in zip(test_file_names, distances_test, neighbors_test):
    sample_result = []
    sample_classes = []
    for d, n in zip(distance, neighbour_):
        train_file = train_files[n].split(os.sep)[-1]
        class_train = file_id_mapping[train_file]
        sample_classes.append(class_train)
        sample_result.append((class_train, d))

    if "new_whale" not in sample_classes:
        sample_result.append(("new_whale", 0.1))
    sample_result.sort(key=lambda x: x[1])
    sample_result = sample_result[:5]
    preds_str.append(" ".join([x[0] for x in sample_result]))

df = pd.DataFrame(preds_str, columns=["Id"])
df['Image'] = [x.split(os.sep)[-1] for x in test_file_names]
df.to_csv("sub_%s.csv"%model_name, index=False)

