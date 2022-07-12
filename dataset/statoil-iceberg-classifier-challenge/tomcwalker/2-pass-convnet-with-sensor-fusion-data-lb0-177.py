# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


####################################################################################
# Imports
####################################################################################
import pandas as pd
import numpy as np
import json

from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import backend
from keras.models import Model
from keras import layers
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


####################################################################################
# Utilities for n-fold
###################################################################################

# Get the index IDs for a given fold
def getFoldIds (foldNumber, foldCount, dfForIndex, seed=1811) :
    kf = KFold(foldCount, True, seed)
    fold = 1

    for train_index, cv_index in kf.split(dfForIndex):
        fold += 1
        if fold >= foldNumber:
            break

    return train_index, cv_index


# Get the content of the nth fold for each of the input data sets
def getNFoldTrainCv(X_train_full, y_train_full, fold, foldCount, seed=1811):
    print ("FoldNumber is ", fold, " of ", foldCount, " folds")

    trainIds, cvIds = getFoldIds(fold, foldCount, X_train_full, seed)

    X_cv = X_train_full.iloc[cvIds,:]
    y_cv = y_train_full.iloc[cvIds,:]
    X_train = X_train_full.iloc[trainIds, :]
    y_train = y_train_full.iloc[trainIds, :]

    return X_train, y_train, X_cv, y_cv


####################################################################################
# Data transformations
####################################################################################

#
# Transform to strip out na angle
#
class tr_dropna_angle:
    def __init__(self, dfs):
        self.name = "dropNaAngle"
        self.dfs = dfs

    def transform(self, df):
        if (df['inc_angle'].dtype == "object"):
            df = df[df['inc_angle'] != "na"]
        df.loc[:,'inc_angle'] = pd.to_numeric(df.loc[:,'inc_angle'])
        print (df['inc_angle'].dtype)
        return df

    def fit(self):
        return

#
# Transform to remove angle column from data set
#
class tr_drop_angle_col:
    def __init__(self, dfs):
        self.name = "dropAngleCol"
        self.dfs = dfs

    def transform(self, df):
        return df.drop('inc_angle', axis=1)

    def fit(self):
        return
#
# Transform to normalise inc_angle to sd of 1 and mean of zero
#
class tr_norm_angle:
    def __init__(self, dfs):
        self.name = "normAngle"
        self.dfs = dfs
        self.mean_angle = None
        self.angle_min = None
        self.angle_max = None

    def transform(self, df):
        df.loc[:,'inc_angle'] = (df.loc[:,'inc_angle'] - self.angle_mean) / (self.angle_max - self.angle_min)
        return df

    def fit(self):
        all = pd.Series()
        for df in self.dfs:
            all = all.append(df.loc[:,'inc_angle'])

        self.angle_mean = np.mean(all)
        self.angle_min = np.mean(all)
        self.angle_max = np.max(all)
        return

# Add scaled columns reflecting histogram of the pixel intensities of each image.
# This should give an idea of image size for the iceberg/ship (which would be the high intensity pixel bucket)
# and also caputure things like the lens flare which seems to occur in some images.
class tr_image_metadata():
    def __init__(self, dfs):
        self.name = "image_metadata"
        self.dfs = dfs

    def fit(self):
        return

    def transform(self,df, scaleFactor = 0.001):
        df_hb1 = pd.DataFrame(index=df.index)
        df_hb1['hist_vals'] = df['band_1'].apply(lambda x: np.histogram(x, 20)[0])
        df_hb1 = df_hb1 * scaleFactor
        df = df.assign(**pd.DataFrame(df_hb1.hist_vals.values.tolist(), index=df.index).add_prefix("b1_histbin"))
        print (df.columns)

        df_hb2 = pd.DataFrame(index=df.index)
        df_hb2['hist_vals'] = df['band_2'].apply(lambda x: np.histogram(x, 20)[0])
        df_hb2 = df_hb2 * scaleFactor
        df = df.assign(**pd.DataFrame(df_hb2.hist_vals.values.tolist(), index=df.index).add_prefix("b2_histbin_"))

        print ("Metadata df")
        df2 = df.drop(['band_1', 'band_2'], axis=1)
        print(df2.head())
        print(df2.describe())

        return df

#
# Return min, max, mean of a given column in a dataframe. Used to do normalisation.
#
def getMinMaxMean(dfs, col):

    firstTime = True
    total = 0
    totalLen = 0

    for df in dfs:
        listOfLists = df[col].values.tolist()
        flatList = [item for sublist in listOfLists for item in sublist]
        mnCur = min(flatList)
        mxCur = max(flatList)
        total = total + sum(flatList)
        totalLen = totalLen + len(flatList)

        if (firstTime):
            (mn, mx) = (mnCur, mxCur)
        else:
            mn = min(mn, mnCur)
            mx = max(mn, mnCur)

    av = total / totalLen

    return mn, mx, av

# Apply a list of transforms to the dataframes passed in
def applyTransforms(transformers, dfs):
    for transCls in transformers:
        trans = transCls(dfs)
        trans.fit()
        for d in range(0, len(dfs)):
            dfs[d] = trans.transform(dfs[d])

    return dfs

#
# Transform to scale the image data from 0->255, normalising across the whole
# image population, rather than just each individual image. Thus we preserve the
# difference in mean intensity between images while scaling the values to be
# consistent
#
#
class tr_minmax_scale_band_global:
    def __init__(self, dfs, lower=0, upper=255
                 ):
        self.name = "normMinMaxGlobal"
        self.dfs = dfs
        self.min_band1, self.max_band1, self.mean_band1 = getMinMaxMean(self.dfs, "band_1")
        self.min_band2, self.max_band2, self.mean_band2 = getMinMaxMean(self.dfs, "band_2")
        self.lower = lower
        self.upper = upper

    def fit(self):
        return

    def scaleList(self, lst, mn, mx, mean, lower=None, upper=None):
        if (lower == None or upper == None):
            upper = 1
            lower = 0

        scaleFactor = (upper - lower) / (mx - mn)
        offset = mx * scaleFactor - upper

        return [elem * scaleFactor - offset for elem in lst]

    def transform(self, df):
        df['band_1'] = (df['band_1'].apply(
            lambda lst: self.scaleList(lst, self.min_band1, self.max_band1, self.mean_band1, self.lower,
                                       self.upper)))
        df['band_2'] = (df['band_2'].apply(
            lambda lst: self.scaleList(lst, self.min_band2, self.max_band2, self.mean_band2, self.lower,
                                       self.upper)))

        mn, mx, av = getMinMaxMean([df], 'band_1')
        print("band_1 min max mean is ", mn, mx, av)
        mn, mx, av = getMinMaxMean([df], 'band_2')
        print("band_2 min max mean is ", mn, mx, av)

        return df

#
# Return a dataframe of images only with the same index as the input DF
#
def get_imgs(df, img_size=75):
    imgs = get_imgs_as_list(df, size=img_size)
    retVal = pd.DataFrame(index=df.index)
    retVal['imgs'] = imgs
    return retVal


#
# Utility to convert the 2 band data to an array of 3 channel images
#
# The third band will be discarded and is only used to enable use of standard image processing
# functions (in particular, keras image data generators random_transform doesn't work for 2 channel images
#
def get_imgs_as_list(df, size=75):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(size, size)
        band_2 = np.array(row['band_2']).reshape(size, size)
        band_3 = (band_1 + band_2) / 2

        # Rescale
        imgs.append(np.dstack((band_1, band_2, band_3)))

    return imgs
#
# Get inputs to 2 part model in np array form from Dataframe.
# First element is the images, second is the sensor fusion data
def getModelInputs(X, addCols, img_size=75,channels = 2):
    imgs = np.asarray(get_imgs_as_list(X, size=img_size))
    return [imgs[:, :, :, 0:channels], np.array(X[addCols])]



####################################################################################
# Model which implements merging of sensor fusion data
# 
# You can play with the shape of the model.
# 
# After the convnet bottleneck we merge the bottleneck data with the extra inputs
# (angle and histogram here) but you could add any set of metadata.
#
# On the first pass we have no extra inputs (so fusionlen is zero) as we just train on the image data.
#
# We then load the weights from this pass for the second pass and add the extra data. 
#
# The later layers are named based on the size of the inputs. Because Keras
# loads weights based on the layer names, this ensure weights are only load for layers which have
# identical input/output shapes.  This avoid errors with mismatching layer size when loading the weights.
# 
####################################################################################
def Convnet_exp_2(fusionLen, weights_path=None, batchNorm = False, ks=3, channels = 2, filters1 = 64, filters2=128, filters3 = 128, filters4=256, op1=1024, op2=512):

    print ("Channels is ", channels)
    image_model_input = Input(shape=(75,75,channels))

    x = Convolution2D(filters =filters1, kernel_size=(ks, ks), activation='relu', padding="same", name = "c1a")(image_model_input)
    x = Convolution2D(filters =filters1, kernel_size=(ks, ks), activation='relu', padding="same", name = "c1b")(x)
    x = Convolution2D(filters =filters1, kernel_size=(ks, ks), activation='relu', padding="same", name = "c1c")(x)
    x = MaxPooling2D((3,3), strides=(2,2), name="mp1")(x)

    x = Convolution2D(filters =filters2, kernel_size=(ks, ks), activation='relu', padding="same", name = "c2a")(x)
    x = Convolution2D(filters =filters2, kernel_size=(ks, ks), activation='relu', padding="same", name = "c2b")(x)
    x = Convolution2D(filters =filters2, kernel_size=(ks, ks), activation='relu', padding="same", name = "c2c")(x)
    x = MaxPooling2D((2,2), strides=(2,2), name="mp2")(x)

    x = Convolution2D(filters =filters3, kernel_size=(ks, ks), activation='relu', padding="same", name = "c3a")(x)
    x = MaxPooling2D((2,2), strides=(2,2), name="mp3")(x)

    x = Convolution2D(filters =filters4, kernel_size=(ks, ks), activation='relu', padding="same", name = "c4a")(x)
    x = MaxPooling2D((2,2), strides=(2,2), name="mp4")(x)

    ####################################################################################
    # Here is where we merge the sensor fusion data (the additional metadata columns) in
    # Note that later layers names are labelled with the number of additional columns, see
    # comment at start of function.
    ####################################################################################
    fl = Flatten(name="flatten_"+str(fusionLen))(x)
    fusion_model_input = Input(shape=(fusionLen,), name="fmi_"+ str(fusionLen))
    merged = layers.concatenate([fusion_model_input, fl])

    # , input_dim = 5626
    x = Dense(op1, activation='relu', name = "dense_1_"+str(fusionLen))(merged)
    x = Dropout(0.4)(x)

    x = Dense(op2, activation='relu', name = "dense_2_"+str(fusionLen))(x)
    x = Dropout(0.2)(x)

    final = Dense(1, activation='sigmoid', name = "final_"+str(fusionLen))(x)

    model = Model(inputs=[image_model_input, fusion_model_input], outputs = final)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    print (model.summary())
    return model




####################################################################################
#
# This callback reduces the lr when the model stops learning. It also reloads the 
# best weights so far when the LR is reduced. This helps avoid overfitting 
# with the small, noisy data set
# 
####################################################################################
class ReduceLROnPlateauCustom(callbacks.ReduceLROnPlateau):
    def __init__(self, model_filepath, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateauCustom, self).__init__(monitor=monitor, factor=factor, patience=patience,
                 verbose=verbose, mode=mode, epsilon=epsilon, cooldown=cooldown, min_lr=min_lr)
        self.model_filepath = model_filepath

    def on_epoch_end(self, epoch, logs=None):
        lrBefore = backend.get_value(self.model.optimizer.lr)
        super(ReduceLROnPlateauCustom, self).on_epoch_end(epoch, logs)
        lrAfter = backend.get_value(self.model.optimizer.lr)

        if (lrBefore != lrAfter):
            # lr has been reduced, load in best previous model.

            print ("LR reduced, loading in best previous weights")
            self.model.load_weights(self.model_filepath)



#
# Return a list of callbacks for the model.
#
def getCallbacks(model_filepath, min_delta=0.00005, patience = 50):

    cb= []

    checkpoint = callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=2, save_best_only=True,
                                                     save_weights_only=False, mode='min', period=1)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience,
                                                       verbose=True, mode='min')

    reduce_LR = ReduceLROnPlateauCustom(model_filepath=model_filepath, monitor='val_loss', factor=0.3, patience=10,
                                        verbose=1,
                                        mode='auto',
                                        epsilon=0.00001, cooldown=20, min_lr=0.00000000001)
    cb.append(early_stopping)
    cb.append(checkpoint)
    cb.append(reduce_LR)

    return cb



# Return the model we will use
# weightsFile = initial file to load weights from
# addCols = list of additional metadata columns (ie not the raw image data) to feed into the convnet.
#           This is where the sensor fusion (ie adding the metadata to the convnet) is done
#           For example ['inc_angle'] if just adding inc_angle.
#
def getModel(lr = 0.0001, addCols = None, weights_path = ""):

    model = Convnet_exp_2(fusionLen=len(addCols),
                                   weights_path=weights_path
                                   )
    optimizer = Adam(lr, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


#
# Take a dataframe such as train, which has 'is_iceberg' labels, and split into X and Y, where
# X is the data without the labels, and y is the labels only.
#
# Returns X, y as dataframes, with the same indexing as X.
#   X has all the columns of df except 'is_iceberg'
#   y has only the "is_iceberg" columns#
def splitXy( df ):
    y = pd.DataFrame(index = df.index)
    y.loc[:,"is_iceberg"] = df.loc[:,"is_iceberg"]
    X = df.drop("is_iceberg", axis=1, inplace = False)
    return X, y


# Create a Keras ImageDataGenerator. We will use this in our custom image generator.
#
# I have hand-coded the width and height shifts as I wanted to be 100% clear exactly what these
# were doing, but you should be able to just use the built in width and height shift
#
def getImageDataGenerator():
    idg = ImageDataGenerator(
        vertical_flip= True,
        horizontal_flip=True,
    )

    return idg

# Get the list of sensor fusion (additional metadata) columns we want to add to our CNN image processing model.
# This is just the columns from the input DF with the image data removed.
def getAddCols(df):
    retVal = df.columns.tolist()

    retVal.remove('band_1')
    retVal.remove('band_2')

    print ("Additiona columns: ", retVal)
    return retVal

#
# Custom data generator to generate a batch of training images with associated sensor fusion metadata
#
class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, X, y, idg, img_size=75, add_cols = ['inc_angle'],  width_shift_range=0, height_shift_range = 0,
               batch_size = 32, debug = False):
      self.batch_size = batch_size
      self.idg = idg
      self.Xy = X
      self.Xy.loc[:,'is_iceberg'] = y.loc[:,'is_iceberg']
      self.debug = debug
      self.width_shift_pixels = int(width_shift_range * img_size)
      self.height_shift_pixels = int(height_shift_range * img_size)
      self.add_cols = add_cols
      self.img_size = img_size

  def __get_batch_df(self, random_state=None):
    return self.Xy.sample(self.batch_size, replace=True, random_state=random_state, axis=0)

  # Should be able to replace this with Keras generator width and height shift, but I chose to
  # roll my own to be sure of exactly what was happening
  def __shiftWidthHeight(self, img):

      if (self.height_shift_pixels > 0) :
          height_shift = np.random.randint(low=-self.height_shift_pixels, high=self.height_shift_pixels)
      else:
          height_shift = 0

      if (self.width_shift_pixels > 0):
          width_shift = np.random.randint(low=-self.width_shift_pixels, high=self.width_shift_pixels)
      else:
          width_shift = 0

      # print("ws/hs", width_shift, height_shift)
      # print("shape", img.shape)
      return np.roll(img, shift = (height_shift, width_shift), axis=(0,1) )

  def generate(self, channels=2):
      size = self.img_size
      while 1:
          batchDf = self.__get_batch_df()
          raw_batch_imgs = np.array(get_imgs(batchDf, img_size=size)['imgs'])

          add_cols = np.array((batchDf)[self.add_cols])

          y = np.array( batchDf['is_iceberg'])

          if (self.debug):
              print ("Prepping batch data")
              print ("raw_batch_imgs.shape ", raw_batch_imgs.shape)
              print ("add_cols.shape ", add_cols.shape)

          output_imgs = np.zeros((self.batch_size, size, size, 3))

          for i in range(raw_batch_imgs.shape[0]) :
              x = self.idg.random_transform(raw_batch_imgs[i])
              output_imgs[i] = self.__shiftWidthHeight(x)

          X = [output_imgs[:,:,:,0:channels], add_cols]

          if (self.debug):
              print ("Yielding a batch")
              print ("X[0].shape is ", X[0].shape)
              print ("X[1].shape is ", X[1].shape)
              print ("y.shape is ", y.shape)

          yield X, y

# Transforms for initial model, normalise the image data and strip everything else out
transformers_images_only = [
    tr_drop_angle_col,
    tr_minmax_scale_band_global,
    ]

# Transforms for sensor fusion model, includes image, angle and histogram metadata, all normalised.
transformers_with_metadata = [
    tr_dropna_angle,
    tr_norm_angle,
    tr_minmax_scale_band_global,
    tr_image_metadata
    ]

# Get model filename
def getModelFilename(prefix, fold):
    return prefix + str(fold) + ".hdf5"

def getOutputFilename(prefix, fold):
    return prefix+ "_preds_" + str(fold) +".csv"

# Main function for running a model. 
def loadTrainAndPredict(initial_weights_file_prefix="", model_file_prefix = "",
        transforms = transformers_images_only,
        batch_size=32,
        epochs=400):

    channels = 2
    img_size = 75

    test = pd.read_json('../input/test.json').set_index("id", drop=True)
    train = pd.read_json('../input/train.json').set_index("id", drop=True)

    # Transform data
    [train_trans, test_trans] = applyTransforms(transforms, [train, test])
    X_train_full, y_train_full = splitXy(train_trans)
    X_test = test_trans

    
    foldCount = 5
    for fold in range (1, foldCount+1):

                model_path = getModelFilename(model_file_prefix, fold)

                # Split the fold data out
                X_train, y_train, X_cv, y_cv = getNFoldTrainCv(X_train_full, y_train_full, fold, foldCount)

                # Get list of sensor fusion columns to feed correctly into model inputs
                addCols = getAddCols(X_train_full)

                idg = getImageDataGenerator()
                trainDataGenerator = DataGenerator(X_train, y_train, idg, img_size = img_size,
                                                   add_cols = addCols,
                                                   width_shift_range=0.12,
                                                   height_shift_range= 0.21,
                                                   batch_size= batch_size,
                                                   debug=False,
                                                   )
                # Create model
                weightsPath = None if initial_weights_file_prefix == "" else getModelFilename(initial_weights_file_prefix, fold)
                model= getModel(addCols=addCols, weights_path=weightsPath)

                cv_imgs=np.array( get_imgs_as_list(X_cv, size=img_size))[:,:,:,0:channels]

                history = model.fit_generator(trainDataGenerator.generate(channels=channels),
                                    steps_per_epoch=X_train.shape[0]/batch_size,
                                    validation_data=([cv_imgs, np.array(X_cv[addCols])], np.array(y_cv)),
                                    epochs=epochs,
                                    use_multiprocessing=False,
                                    callbacks=getCallbacks(model_path),
                                    verbose=2,
                                    workers=1)

                outputFile = getOutputFilename(model_file_prefix,fold)

                writeResults(model_path, X_test, addCols, outputFile)

    saveFoldedPreds(model_file_prefix, model_file_prefix + "_folded_preds.csv")


def writeResults(modelFile, X_test, addCols, outputFile):

    print ("Generating predictions using model file ", modelFile)
    model = load_model(modelFile)
    preds = model.predict(getModelInputs(X_test, addCols=addCols))

    dfOut = pd.DataFrame(index=X_test.index)
    dfOut['is_iceberg'] = preds
    dfOut.to_csv(outputFile, index_label="id")

    print ("Results " +  outputFile + " written")



def saveFoldedPreds(contrib_file_prefix, outputFile, numFolds =5, upper = 0.9999, lower = 0.0001):
    outputDf = None

    for i in range(1, numFolds+1):
        filename = getOutputFilename(contrib_file_prefix,i)
        print ("reading filename ", filename)
        predsDf = pd.read_csv(filename, index_col="id")
        if outputDf is None:
            outputDf = pd.DataFrame(index=predsDf.index)
            outputDf['is_iceberg'] = 0
        outputDf['is_iceberg'] = outputDf['is_iceberg'] + predsDf['is_iceberg']

    outputDf['is_iceberg'] = outputDf['is_iceberg'] / numFolds

    outputDf['is_iceberg'] = outputDf['is_iceberg'].apply(lambda x: upper if x > upper else (lower if x < lower else x))

    outputDf.to_csv(outputFile, index_label="id")

# Train first model with just images
#**************************************************************
# CHANGE THE NUMBER OF EPOCHS TO 200 OR MORE FOR A GOOD RESULT
# EPOCHS SET LOW TO ALLOW KERNEL TO RUN TO COMPLETION
#*************************************************************
epochs = 20
loadTrainAndPredict(initial_weights_file_prefix="", transforms= transformers_images_only, model_file_prefix = "model_base_", epochs=epochs)

# Train second model with extra metadata using weights loaded from the previous model
loadTrainAndPredict(initial_weights_file_prefix="model_base_", transforms= transformers_with_metadata, model_file_prefix = "model_with_fusion_data_", epochs=epochs)