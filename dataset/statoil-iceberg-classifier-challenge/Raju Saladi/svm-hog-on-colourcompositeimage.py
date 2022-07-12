# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #open cv for image processing, imafe feature extraction
from sklearn import svm # for fitting model to features
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pdb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Loading the training and testing data

trainDataFilePath = "../input/train.json"
testDataFilePath = "../input/test.json"

trainDataFrame = pd.read_json('../input/train.json')
#trainDataFrame.head()

testDataFrame = pd.read_json('../input/test.json')
#testDataFrame.head()



# Any results you write to the current directory are saved as output.


#Calculate the Hog features out of the Image
def getHogDescriptor(image,binNumber = 16):
   gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
   gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
   mag, ang = cv2.cartToPolar(gx, gy)
   bins = np.int32(binNumber*ang/(2*np.pi))    # quantizing binvalues in (0...16)
   bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
   mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
   hists = [np.bincount(b.ravel(), m.ravel(), binNumber) for b, m in zip(bin_cells, mag_cells)]
   hist = np.hstack(hists)     # hist is a 64 bit vector
   hist = np.array(hist,dtype=np.float32)
   return hist

#Get the Colour Composite RGB Image from HH and HV bands
def makeRGBImageFromHnV(bandHH,bandHV):
    b = np.divide(bandHH, bandHV, out=np.zeros_like(bandHH), where=(bandHV!=0))
    rgb = np.dstack((bandHH.astype(np.float32), bandHV.astype(np.float32),b.astype(np.uint16)))
    return rgb

# get mean and std of image dataframe


def getMeanImageFromImageDataFrame(trainDataFrame):
    meanImage = np.zeros(shape =(75,75,3),dtype = np.float32)
    for currentImage in trainDataFrame["fullImage"]:
        meanImage = meanImage + currentImage
    meanImage = meanImage/len(trainDataFrame)
    return meanImage.astype(np.float32)

def getStandardDeviationFromImageDataFrame(trainDataFrame,meanImage):
    stdImage = np.zeros(shape =(75,75,3),dtype = np.float32)
    for currentImage in trainDataFrame["fullImage"]:
        stdImage = stdImage + (currentImage - meanImage)
    stdImage = stdImage/len(trainDataFrame)
    return stdImage.astype(np.float32)

def normalizedImageParamFromDataFrame(trainDataFrame):
    
    meanImageData = getMeanImageFromImageDataFrame(trainDataFrame)
    stdImageData = getStandardDeviationFromImageDataFrame(trainDataFrame,meanImageData)
    #for i in range(0,len(trainDataFrame)):
        #currentImage = trainDataFrame["fullImage"][i]
        #trainDataFrame["fullImageNormalized"][i] = (currentImage - meanImageData)/stdImageData
    return meanImageData,stdImageData

def normalizeSingleImage(currentImage,meanImageData,stdImageData):
    normalizedImage = (currentImage - meanImageData)/stdImageData
    return normalizedImage

def transformImageDataFrame(inputDataFrame,meanImageData,stdImageData):
    inputDataFrame["fullImageNormalized"] = [normalizeSingleImage(inputDataFrame["fullImage"][i],meanImageData,stdImageData) for i in range(0,len(inputDataFrame["fullImage"]))]
    return inputDataFrame

def normalizeImageUsingOpenCV(currentImage):
    norm_image = cv2.normalize(currentImage,currentImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

# Convert the dB Band values into normalValues
def getValuesfromDB(bandDB):
    currentband = np.array(bandDB).reshape(75,75)
    actualValue = 10**(currentband/10)
    return actualValue

# Adding required columns for the DataFrame
def getImageFromBandDataFrame(dataFrame):
    dataFrame["valueBand1"] = [getValuesfromDB(dataFrame["band_1"][i]) for i in range(0,len(dataFrame["band_1"]))]
    dataFrame["valueBand2"] = [getValuesfromDB(dataFrame["band_2"][i]) for i in range(0,len(dataFrame["band_2"]))]
    dataFrame["fullImage"] = [makeRGBImageFromHnV(dataFrame["valueBand1"][i],dataFrame["valueBand2"][i]) for i in range(0,len(dataFrame["band_1"]))]
    dataFrame["fullImageNormalized"] = [normalizeImageUsingOpenCV(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    #dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    return dataFrame


def bootstrapAndEqualizeTheData(inputDataFrame):
	noOfIceBergData = len(inputDataFrame[inputDataFrame.is_iceberg == 1])
	totalData = len(inputDataFrame)
	print("Randomly bootstrap and make the iceberg and ship data to be equal")
	randomSamplesForIceBerg = inputDataFrame[inputDataFrame.is_iceberg == 1].sample((totalData - (2*noOfIceBergData)))
	inputDataFrame = pd.concat([inputDataFrame, randomSamplesForIceBerg], ignore_index=True)
	return inputDataFrame


def addFeatureDataFrame(dataFrame):
    #dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImageNormalized"][i]) for i in range(0,len(dataFrame["fullImageNormalized"]))]
    return dataFrame

# Extracting Features from the Data Frame
def getFeatureFromDataFrame(dataFrame,isTestData=0):
    featureDataVector = []
    responseVector = []
    featureDataVector =  np.array(featureDataVector).reshape(-1,64)
    for i in range(0,len(dataFrame)):
        currentFeature = dataFrame["hogFeature"][i].tolist()
        if(isTestData is 0):
            currentResponse = dataFrame["is_iceberg"][i].tolist()
        else:
            currentResponse = 2 # dummy which will be ignored later
        currentFeature = np.array(currentFeature[0:64]).reshape(-1,64)
        currentResponse = int(currentResponse)
        if(i == 0):
            featureDataVector = currentFeature
            responseVector.append(currentResponse)
        else:
            featureDataVector = np.vstack((featureDataVector,currentFeature))
            responseVector.append(currentResponse)
    return featureDataVector,responseVector





print("Adding New Columns to DataFrame")


print("Normalizing Data Frame")
#scaler = preprocessing.MinMaxScaler()
#pdb.set_trace()
#trainDataFrame["fullImage"] = trainDataFrame["fullImage"].astype(np.float32)
#trainDataFrame["fullImage"] = scaler.fit_transform(trainDataFrame["fullImage"])
#meanImage = int(np.mean(trainDataFrame["fullImage"]))


#meanImageData,stdImageData = normalizedImageParamFromDataFrame(trainDataFrame)
#trainDataFrame = transformImageDataFrame(trainDataFrame,meanImageData,stdImageData)
trainDataFrame = getImageFromBandDataFrame(trainDataFrame)
trainDataFrame = bootstrapAndEqualizeTheData(trainDataFrame)
trainDataFrame = addFeatureDataFrame(trainDataFrame)

print("Calculating Feature Vectors")
trainFeatureData , trainResponseData = getFeatureFromDataFrame(trainDataFrame)
print("done..")


#print("Feature Normalization")
#scaler = preprocessing.StandardScaler().fit(trainFeatureData)
#trainFeatureData = scaler.transform(trainFeatureData)

#scaler = preprocessing.QuantileTransformer(random_state=0)
#trainFeatureData = scaler.fit_transform(trainFeatureData)

#scaler = preprocessing.MinMaxScaler()
#trainFeatureData = scaler.fit_transform(trainFeatureData)

print(str(trainFeatureData.shape))
#pdb.set_trace()
print("Fitting Data to Model")
#clf = svm.SVC(gamma=0.01,C=10,kernel='poly',probability=True)
clf = svm.SVC(gamma=0.001,C=100,kernel='rbf',probability=True)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),algorithm="SAMME",n_estimators=500)
#clf1 = DecisionTreeClassifier(max_depth=4)
#clf2 = KNeighborsClassifier(n_neighbors=7)
#clf3 = SVC(kernel='rbf', probability=True)
#param_grid = {'kernel':('poly','rbf'),'C':[1,10,100],'gamma':[0.0001,0.01,0.1]}
#clf = GridSearchCV(svm.SVC(probability=True), param_grid)
clf.fit(trainFeatureData,trainResponseData)

print("Done")

print("getting TestData features...")
print("Normalizing Data Frame")
testDataFrame = getImageFromBandDataFrame(testDataFrame)
#testDataFrame = transformImageDataFrame(testDataFrame,meanImageData,stdImageData)
#testDataFrame["fullImage"] = scaler.transform(testDataFrame["fullImage"])
testDataFrame = addFeatureDataFrame(testDataFrame)
#testDataFrame = getValuesFromDBDataFrame(testDataFrame)
testFeatureData , _ = getFeatureFromDataFrame(testDataFrame,1)
#testFeatureData , testResponseData = trainFeatureData , trainResponseData
print("done")

#pdb.set_trace()

#trainFeatureData = scaler.transform(trainFeatureData)
print(str(testFeatureData.shape))
test_predictions = clf.predict_proba(testFeatureData)
trainPredictions = clf.predict_proba(trainFeatureData)
print("Prediction done")

print("Log Loss for training Data is "+str(log_loss(trainResponseData,trainPredictions[:,1])))


#pdb.set_trace()

pred_df = testDataFrame[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
pred_df.to_csv('predictions.csv', index = False)
#pred_df.sample(3)