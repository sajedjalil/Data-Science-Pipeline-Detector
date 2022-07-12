import pandas as pd
import os

#os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())

#sampleSubmission = pd.read_csv("../input/sampleSubmission.csv")
#print("sampleSubmission set has {0[0]} rows and {0[1]} columns".format(sampleSubmission.shape))

#print(sampleSubmission.head())

#test = pd.read_csv("../input/test.csv")
#print("test set has {0[0]} rows and {0[1]} columns".format(test.shape))

#print(test.head())

#weather = pd.read_csv("../input/weather.csv")
#print("weather set has {0[0]} rows and {0[1]} columns".format(weather.shape))

#print(weather.head())
