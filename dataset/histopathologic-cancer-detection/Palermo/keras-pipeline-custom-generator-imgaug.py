import numpy as np
import pandas as pd
import time
import keras
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications.nasnet import NASNetMobile, preprocess_input

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback

import imgaug as ia
from imgaug import augmenters as iaa


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, images_paths, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
		self.labels       = labels              # array of labels
		self.images_paths = images_paths        # array of image paths
		self.dim          = image_dimensions    # image dimensions
		self.batch_size   = batch_size          # batch size
		self.shuffle      = shuffle             # shuffle bool
		self.augment      = augment             # augment data bool
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images_paths) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# selects indices of data for next batch
		indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

		# select data and load images
		labels = np.array([self.labels[k] for k in indexes])
		images = [cv2.imread(self.images_paths[k]) for k in indexes]
        
		# preprocess and augment data
		if self.augment == True:
		    images = self.augmentor(images)
		
		images = np.array([preprocess_input(img) for img in images])
		return images, labels
	
	
	def augmentor(self, images):
		'Apply data augmentation'
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		seq = iaa.Sequential(
				[
				# apply the following augmenters to most images
				iaa.Fliplr(0.5),  # horizontally flip 50% of all images
				iaa.Flipud(0.2),  # vertically flip 20% of all images
				sometimes(iaa.Affine(
					scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
					# scale images to 80-120% of their size, individually per axis
					translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
					# translate by -20 to +20 percent (per axis)
					rotate=(-10, 10),  # rotate by -45 to +45 degrees
					shear=(-5, 5),  # shear by -16 to +16 degrees
					order=[0, 1],
					# use nearest neighbour or bilinear interpolation (fast)
					cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
					mode=ia.ALL
					# use any of scikit-image's warping modes (see 2nd image from the top for examples)
				)),
				# execute 0 to 5 of the following (less important) augmenters per image
				# don't execute all of them, as that would often be way too strong
				iaa.SomeOf((0, 5),
				           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
						                                     n_segments=(20, 200))),
					           # convert images into their superpixel representation
					           iaa.OneOf([
							           iaa.GaussianBlur((0, 1.0)),
							           # blur images with a sigma between 0 and 3.0
							           iaa.AverageBlur(k=(3, 5)),
							           # blur image using local means with kernel sizes between 2 and 7
							           iaa.MedianBlur(k=(3, 5)),
							           # blur image using local medians with kernel sizes between 2 and 7
					           ]),
					           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
					           # sharpen images
					           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
					           # emboss images
					           # search either for all edges or for directed edges,
					           # blend the result with the original image using a blobby mask
					           iaa.SimplexNoiseAlpha(iaa.OneOf([
							           iaa.EdgeDetect(alpha=(0.5, 1.0)),
							           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
							                                  direction=(0.0, 1.0)),
					           ])),
					           iaa.AdditiveGaussianNoise(loc=0,
					                                     scale=(0.0, 0.01 * 255),
					                                     per_channel=0.5),
					           # add gaussian noise to images
					           iaa.OneOf([
							           iaa.Dropout((0.01, 0.05), per_channel=0.5),
							           # randomly remove up to 10% of the pixels
							           iaa.CoarseDropout((0.01, 0.03),
							                             size_percent=(0.01, 0.02),
							                             per_channel=0.2),
					           ]),
					           iaa.Invert(0.01, per_channel=True),
					           # invert color channels
					           iaa.Add((-2, 2), per_channel=0.5),
					           # change brightness of images (by -10 to 10 of original value)
					           iaa.AddToHueAndSaturation((-1, 1)),
					           # change hue and saturation
					           # either change the brightness of the whole image (sometimes
					           # per channel) or change the brightness of subareas
					           iaa.OneOf([
							           iaa.Multiply((0.9, 1.1), per_channel=0.5),
							           iaa.FrequencyNoiseAlpha(
									           exponent=(-1, 0),
									           first=iaa.Multiply((0.9, 1.1),
									                              per_channel=True),
									           second=iaa.ContrastNormalization(
											           (0.9, 1.1))
							           )
					           ]),
					           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
					                                               sigma=0.25)),
					           # move pixels locally around (with random strengths)
					           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
					           # sometimes move parts of the image around
					           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
				           ],
				           random_order=True
				           )
				],
				random_order=True
		)
		return seq.augment_images(images)


class NetModel():
	def __init__(self, image_dimensions=(96,96 ,3), n_classes=1):
		self.n_classes = n_classes  # number of classes to classify(1 for binary classification)
		self.input_dim = image_dimensions  # image input dimensions
		self.model = self.create_model()  # model

	def summary(self):
		self.model.summary()

	def create_model(self):
		input_layer = Input(self.input_dim)
        
		nas_mobile_model = NASNetMobile(include_top=False, input_tensor=input_layer, weights='imagenet')
		x = nas_mobile_model(input_layer)

		# output layers
		x1 = GlobalAveragePooling2D()(x)
		x2 = GlobalMaxPooling2D()(x)
		x3 = Flatten()(x)

		out = Concatenate(axis=-1)([x1, x2, x3])
		out = Dropout(0.5)(out)
		output_layer = Dense(self.n_classes, activation='sigmoid')(out)

		model = Model(inputs=input_layer, outputs=output_layer)

		model.compile(optimizer=Adam(lr=0.0005), loss="binary_crossentropy", metrics=['acc'])
		return model


	def train(self, train_data, val_data, plot_results=True):
		'Trains data on generators'
		print("Starting training")

		# reduces learning rate if no improvement are seen
		learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
		                                            patience=2,
		                                            verbose=1,
		                                            factor=0.5,
		                                            min_lr=0.0000001)

		# stop training if no improvements are seen
		early_stop = EarlyStopping(monitor="val_loss",
		                           mode="min",
		                           patience=5,
		                           restore_best_weights=True)

		# saves model weights to file
		checkpoint = ModelCheckpoint('./model_weights.hdf5',
		                             monitor='val_loss',
		                             verbose=1,
		                             save_best_only=True,
		                             mode='min',
		                             save_weights_only=True)

		# visualize training data
		# tensorboard = TensorBoard(log_dir='./logs',
		#                          histogram_freq=0,
		#                          batch_size=BATCH_SIZE,
		#                          write_graph=True,
		#                          write_grads=True,
		#                          write_images=False)

		# reduce resource usage(keeps laptop from melting)
		# idle = LambdaCallback(on_epoch_end=lambda batch, logs: time.sleep(30),
		#                      on_batch_end=lambda batch, logs: time.sleep(0.005))

		# train on data
		history = self.model.fit_generator(generator=train_data,
		                                   validation_data=val_data,
		                                   epochs=EPOCHS,
		                                   steps_per_epoch=len(train_data),
		                                   validation_steps =len(val_data),
		                                   callbacks=[learning_rate_reduction, early_stop, checkpoint],
		                                   verbose=2,
		                                   )
		# plot training history
		if plot_results:
			fig, ax = plt.subplots(2, 1, figsize=(6, 6))
			ax[0].plot(history.history['loss'], label="TrainLoss")
			ax[0].plot(history.history['val_loss'], label="ValLoss")
			ax[0].legend(loc='best', shadow=True)

			ax[1].plot(history.history['acc'], label="TrainAcc")
			ax[1].plot(history.history['val_acc'], label="ValAcc")
			ax[1].legend(loc='best', shadow=True)
			plt.show()

	def create_submit(self, test_data):
		'Create basic file submit'
		self.model.load_weights("./model_weights.hdf5")
		# predict on data
		results = self.model.predict_generator(test_data)

		# binarize prediction
		# rbin = np.where(results > 0.5, 1, 0)

		# save results to dataframe
		results_to_save = pd.DataFrame({"id": test_data.images_paths,
		                                "label": results[:,0]
		                                })

		results_to_save["id"] = results_to_save["id"].apply(lambda x: x.replace("../input/test/", "").replace(".tif", ""))

		# create submission file
		results_to_save.to_csv("./submission.csv", index=False)






def loadData(db, val_split=0.2, sub_sample_size=-1):
	'Loads data into generator object'
	if db == "train":
		data = pd.read_csv(f'../input/train_labels.csv')
		image_paths = data['id'].apply(lambda x: '../input/train/' + x + '.tif').values[:sub_sample_size]
		labels = data['label'].values[:sub_sample_size]
		if val_split > 0:
			X_train, X_test, Y_train, Y_test = train_test_split(image_paths, labels, test_size=val_split)
			train_data = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
			val_data = DataGenerator(X_test, Y_test, batch_size=BATCH_SIZE, augment=False, shuffle=False)
			return train_data, val_data
		else:
			return DataGenerator(image_paths, labels, batch_size=BATCH_SIZE, augment=True, shuffle=True), None

	else:
		data = pd.read_csv(f'../input/sample_submission.csv')
		image_paths = data['id'].apply(lambda x: '../input/test/' + x + '.tif').values
		labels = data['label'].values
		if sub_sample_size== -1 :
			return DataGenerator(image_paths, labels, batch_size=2), None
		else:
			return DataGenerator(image_paths[:sub_sample_size], labels[:sub_sample_size], batch_size=1), None





if __name__ == "__main__":
	EPOCHS = 10
	BATCH_SIZE = 64
	IMAGE_DIMENSIONS = (96,96 ,3)

	# create model
	model = NetModel(image_dimensions=IMAGE_DIMENSIONS, n_classes=1)
	model.summary()

	# train model
	train_data, val_data = loadData("train", val_split=0.2)
	model.train(train_data, val_data, plot_results=True)

	# submit model
	test_data, _ = loadData("test")
	model.create_submit(test_data)