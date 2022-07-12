import os
import shutil
import pathlib
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 5, padding='same')
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        # self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(64, 3, padding='same')
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        # self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(64, 3, padding='same')
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        # self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(128, 3, padding='same')
        self.pool4 = MaxPool2D(pool_size=(2, 2))
        # self.bn4 = BatchNormalization()

        self.flatten = Flatten()
        self.d1 = Dense(128)
        # self.bn5 = BatchNormalization()

        self.d2 = Dense(1)

        self.activattion = Activation('relu')

    def call(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activattion(x)
        x = self.pool1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activattion(x)
        x = self.pool2(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.activattion(x)
        x = self.pool3(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.activattion(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.d1(x)
        # x = self.bn5(x)
        x = self.activattion(x)

        x = self.d2(x)

        return x

print("GPU: ", len(tf.config.list_physical_devices('GPU')) > 0)

BATCH_SIZE = 64
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
CLASSES = np.array(['cat', 'dog'])


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    label = tf.where(CLASSES == parts[-2])[0][0]
    return label


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def test_process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.expand_dims(img, 0)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, batch_size=BATCH_SIZE):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()  # repeat forever
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

# def train():
#     train_path = pathlib.Path('data/train')

#     no_of_samples = len(list(train_path.glob('**/*.jpg')))

#     steps_per_epoch = np.ceil(no_of_samples / BATCH_SIZE)
#     list_ds = tf.data.Dataset.list_files(str(train_path / '*/*'))
#     labeled_ds = list_ds.map(
#         process_path,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )

#     dataset = prepare_for_training(labeled_ds)

#     # image_batch, label_batch = next(iter(dataset))
#     # show_batch(image_batch.numpy(), label_batch.numpy())

#     # Split the dataset
#     train_size = np.ceil(no_of_samples / BATCH_SIZE * 0.7)
#     val_size = np.ceil(no_of_samples / BATCH_SIZE * 0.3)

#     train_ds = dataset.take(train_size)
#     val_ds = dataset.skip(train_size).take(val_size)
#     loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

#     val_loss = tf.keras.metrics.Mean(name='test_loss')
#     val_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

#     @tf.function
#     def train_step(b_images, b_labels):
#         with tf.GradientTape() as tape:
#             predictions = model(b_images, training=True)
#             loss = loss_fn(b_labels, predictions)
#             gradients = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         train_loss(loss)
#         train_accuracy(b_labels, predictions)

#     @tf.function
#     def val_step(b_images, b_labels):
#         predictions = model(b_images, training=True)
#         loss = loss_fn(b_labels, predictions)

#         val_loss(loss)
#         val_accuracy(b_labels, predictions)

#     # model.build()
#     # model.summary()
#     epochs = 10
#     print("Steps per epochs:", steps_per_epoch)
#     print("Training...")
#     prev_loss = np.inf

#     for epoch in range(epochs):
#         train_loss.reset_states()
#         train_accuracy.reset_states()
#         val_loss.reset_states()
#         val_accuracy.reset_states()
#         step = 0
#         for images, labels in train_ds:
#             train_step(images, labels)
#             step += 1
#             # print("Step: {}/{}".format(step, STEPS_PER_EPOCH))
#         vstep = 0
#         for test_images, test_labels in val_ds:
#             val_step(test_images, test_labels)
#             vstep += 1
#             # print("VStep: ", vstep)

#         template = 'Epoch: {}, Loss: {}, Accuracy; {}, Val Loss: {}, Val Accuracy: {}'
#         print(template.format(epoch + 1,
#                               train_loss.result(),
#                               train_accuracy.result() * 100,
#                               val_loss.result(),
#                               val_accuracy.result() * 100))
#         if val_loss.result() < prev_loss:
#             print("Saving weights! Loss decreased: {} ==> {}".format(prev_loss, val_loss.result()))
#             prev_loss = val_loss.result()
#             manager.save()


def kaggle_test(test_path):
    output_file = 'output.csv'
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'label'])
    test_path = pathlib.Path(test_path)

    for img_id in range(1, 12501):
        file = "{}/{}.jpg".format(test_path, img_id)
        image = tf.io.read_file(file)
        image = decode_img(image)
        input_image = tf.expand_dims(image, 0)
        prediction = model(input_image, training=False)

        prediction = int(prediction > 0)
        result = [img_id, prediction]
        # print(*result, sep=',')
        with open(output_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(result)
            if img_id % 1000 == 0:
                print("wrote ", img_id)


if __name__ == "__main__":
    model = MyModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory="/kaggle/input/dogsvscats-checkpoint", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # train()
    # test()
    os.chdir('/kaggle/working')
    with ZipFile('/kaggle/input/dogs-vs-cats/test1.zip') as zip:
        zip.extractall()
    kaggle_test('/kaggle/working/test1')
    print("Execution complete! Deleteing extra dirs.")
    shutil.rmtree('/kaggle/working/test1')
    print("Complete!")