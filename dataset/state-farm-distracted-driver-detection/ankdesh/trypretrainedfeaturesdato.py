# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# coding: utf-8

# In[1]:

import graphlab
import collections


# In[2]:

topFolder = "../input/"
image_sarray = graphlab.image_analysis.load_images( topFolder + 'downsample_train/', 
                                                   "auto", with_path=True,recursive=True)
image_testdata =  graphlab.image_analysis.load_images( topFolder + 'downsample_test/', 
                                                   "auto", with_path=True,recursive=True)


# In[3]:

#print image_sarray[0]


# In[4]:

image_sarray['label'] = image_sarray['path'].apply(lambda x: x.split('/')[-2])
image_sarray['fileName'] = image_sarray['path'].apply(lambda x: x.split('/')[-1])

image_testdata['label'] = image_testdata['path'].apply(lambda x: x.split('/')[-2])
image_testdata['fileName'] = image_testdata['path'].apply(lambda x: x.split('/')[-1])


# In[5]:

#print set(image_sarray['label'])


# In[6]:

training_data, validation_data = image_sarray.random_split(0.8)


# In[7]:

#allHeights96px=image_testdata['image'].apply(lambda x:x.height == 96)
#allWidth128px=image_testdata['image'].apply(lambda x:x.width == 128)


# In[8]:

#allHeights96px.all()
#allWidth128px.all()


# In[9]:

extractor = graphlab.feature_engineering.DeepFeatureExtractor(features = 'image',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            model='auto')


# In[10]:

extractor = extractor.fit(training_data)
extracted_model = extractor['model']


# In[11]:

features_train = extractor.transform(training_data)
#print 'Train feature Done'
features_validation = extractor.transform(validation_data)
#print 'Validation feature Done'


# In[ ]:




# In[13]:

features_train.save('trainFeaturesDownSample.csv', format='csv')
features_validation.save('validationFeaturesDownSample.csv', format='csv')


# In[20]:

custom_net_layers = list()
#custom_net_layers.append(graphlab.deeplearning.layers.ConvolutionLayer(kernel_size=3,
#                                                                    stride=2,
#                                                                    num_channels=10))
custom_net_layers.append(graphlab.deeplearning.layers.FullConnectionLayer(100))
custom_net_layers.append(graphlab.deeplearning.layers.SigmoidLayer())
custom_net_layers.append(graphlab.deeplearning.layers.FullConnectionLayer(100))
custom_net_layers.append(graphlab.deeplearning.layers.SigmoidLayer())
custom_net_layers.append(graphlab.deeplearning.layers.FullConnectionLayer(10))
custom_net_layers.append(graphlab.deeplearning.layers.SoftmaxLayer())

custom_net = graphlab.deeplearning.NeuralNet()
custom_net.layers = custom_net_layers
#print custom_net.verify()


# In[21]:

model = graphlab.neuralnet_classifier.create(features_train, validation_set = features_validation,
                                             network=custom_net,
                                             features=['deep_features.image'], target='label')


# In[22]:

features_test = extractor.transform(image_testdata)
features_test.save('testFeaturesDownSample.csv', format='csv')
#print 'Test feature Done'


# In[24]:

classProb = model.predict_topk(features_test,k=10)


# In[25]:

classProb.save('tempSaveClassProb.csv')


# In[26]:

formattedData = collections.defaultdict(dict)
for eachRow in classProb:
        formattedData[eachRow['row_id']][(int(eachRow['class'][1]))] = eachRow['probability']


# In[27]:

resultCsv = []
for i in range(image_testdata.shape[0]):
    csvStr = ''
    csvStr = image_testdata['fileName'][i] + ','
    for j in range(10):
        csvStr += str(formattedData[i][j]) + ','
    resultCsv.append(csvStr[:-1])
#print resultCsv[0]


# In[ ]:




# Any results you write to the current directory are saved as output.