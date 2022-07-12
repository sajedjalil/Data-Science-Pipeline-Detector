# all the imports
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from os.path import join
from keras import backend as K
import matplotlib.image as mpimg
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
K.clear_session()

def cam(model, img, inShape):
    def overlayHeatmap(img, heatmap):
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = 255 - np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, .8, heatmap, .4, 0)
        return img

    preImg = cv2.resize(img, inShape, interpolation=cv2.INTER_CUBIC)
    preImg = np.expand_dims(preImg, axis=0)
    argmax = model.predict(preImg).argmax(axis=1)[0]
    output = model.output[:, argmax]
    convLayers = [layer for layer in model.layers if 'conv' in layer.name][::-1]
    for convLayer in convLayers:
        grads = K.gradients(output, convLayer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, convLayer.output[0]])
        poolGrads, layerOutput = iterate([preImg])
        for i in range(len(poolGrads)):
            layerOutput[:, :, i] *= poolGrads[i]
        heatmap = np.mean(layerOutput, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = overlayHeatmap(img, heatmap)
        plt.imshow(heatmap)
        plt.show()
        
np.random.seed(42)
imDir = '../input/plant-pathology-2020-fgvc7/images'
id2impath = lambda x: join(imDir, '%s.jpg' % x)

query = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
data = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')#[:20]
data = data.sample(frac=1).reset_index(drop=True) # shuffle data
data.image_id = data.image_id.apply(id2impath)
query.image_id = query.image_id.apply(id2impath)
classes = data.columns[1:].tolist() # remove image_id columns

nImgs = len(data)
splitBy = .7,.2,.1 # changed from .6,.2,.2
splitBy = [int(sum(splitBy[:i]) * nImgs) for i, _ in enumerate(splitBy, 1)]
train, valid, test, _ = np.split(data, splitBy)
img = mpimg.imread(train.image_id[3])

print("img.shape    : ", img.shape)
print("train.shape  : ", train.shape)
print("valid.shape  : ", valid.shape)
print("test.shape   : ", test.shape)
print("query.shape  : ", query.shape)
print("classes      : ", classes)

display(train.iloc[[0,1,-2,-1]], valid.iloc[[0,1,-2,-1]], test.iloc[[0,1,-2,-1]], query.iloc[[0,1,-2,-1]])
del data

plt.imshow(img)
plt.show()
for data in [train, valid, test]:
    v = data[classes].idxmax(axis=1).value_counts().plot.bar()
    plt.show()
    
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import inception_v3
from keras import regularizers
from keras.metrics import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor


def getModel(nClasses, weights='default'):
    if weights == 'default':
        weights = glob('../input/keras-pretrained-models/inception_v3*_notop.h5')[0]
    base_model = InceptionV3(weights=weights, include_top=False)
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(nClasses, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers[:-22]:
        layer.trainable = False
    # Compile with Adam
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, (299,299), '/kaggle/working/inception_v3_plant_fgvc7.h5'


model, inShape, outWeightPath = getModel(len(classes),weights='imagenet')
print(model.summary())

def fetch_imgs_np(names, img_size=100, num_channels=3, normalize=True):
    imgs_np = np.ndarray(shape=(len(names), img_size, img_size, num_channels), dtype=np.float32)
    for i, name in enumerate(names):
        img_path = name
        img=cv2.imread(img_path)
        img=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
        if normalize is True:
            img = img / 255
        imgs_np[i] = img
    return imgs_np
trainImages = fetch_imgs_np(train['image_id'], img_size=inShape[0])
validImages = fetch_imgs_np(valid['image_id'], img_size=inShape[0])
print("Training Data Shape : ",trainImages.shape)
print("Validation Data Shape : ",validImages.shape)

trainLabels = train.drop('image_id', axis=1).to_numpy()
validLabels = valid.drop('image_id', axis=1).to_numpy()
print("Training Labels shape : ",trainLabels.shape)
print("Validation Lables shape : ",validLabels.shape)

# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        horizontal_flip=True, # Randomly flip inputs horizon7777tally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(trainImages)

# Fits the model on batches with real-time data augmentation
hist = model.fit_generator(datagen.flow(trainImages, trainLabels, batch_size=32),
               steps_per_epoch=trainImages.shape[0] // 32,
               epochs=12,
               verbose=1,
               validation_data=(validImages, validLabels))

# Create Testing Test and Calculate Accuracy
testImages = fetch_imgs_np(test['image_id'], img_size=inShape[0])
testLabels = test.drop('image_id', axis=1).to_numpy()
print("Testing Data Shape : ",testImages.shape)
print("Testing Labels Shape : ",testLabels.shape)

print('\n# Evaluate on test data')
results = model.evaluate(testImages, testLabels)
print('test loss, test acc:', results)

# Create the predictions for sample_submissions.csv
submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
submission.image_id = submission.image_id.apply(id2impath)
print(submission.head())

subImages = fetch_imgs_np(submission['image_id'], img_size=inShape[0])
print("Submission Data Shape : ",subImages.shape)

# Get the Predictions for Sample_Submission Images
subPredictions = model.predict(subImages)
print("Prediction Shape : ",subPredictions.shape)

all_predict = np.ndarray(shape = (subImages.shape[0],4),dtype = np.float32)
for i in range(0,subImages.shape[0]):
    for j in range(0,4):
        if subPredictions[i][j]==max(subPredictions[i]):
            all_predict[i][j] = 1
        else:
            all_predict[i][j] = 0 
healthy = [pred[0] for pred in all_predict]
multiple_diseases = [pred[1] for pred in all_predict]
rust = [pred[2] for pred in all_predict]
scab = [pred[3] for pred in all_predict]

submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
df_dict = {'image_id':submission['image_id'],'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
pred = pd.DataFrame(df_dict)

pred.head()

pred.to_csv('submission.csv', index=False)