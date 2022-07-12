#Load standard libraries
#%pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Load data
df = pd.read_csv('../input/train.csv')
print(df.columns.values)


fig = plt.figure()
N = 5
for k in range(N):
    margin0 = df.filter(regex = "margin*").loc[k].reshape((8,8))
    imgplot1 = fig.add_subplot(N,4,4*k+1)
    imgplot1.imshow(margin0)
    imgplot1.axis('off')
    shape0 = df.filter(regex = "shape*").loc[k].reshape((8,8))
    imgplot1 = fig.add_subplot(N,4,4*k+2)
    imgplot1.imshow(shape0)
    imgplot1.axis('off')
    texture0 = df.filter(regex = "margin*").loc[k].reshape((8,8))
    imgplot1 = fig.add_subplot(N,4,4*k+3)
    imgplot1.imshow(texture0)
    imgplot1.axis('off')
    imgplot1 = fig.add_subplot(N,4,4*k+4)
    imgplot1.text(0, 0.5, df['species'].loc[k],
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=20)
    imgplot1.axis('off')




#How many samples are there of each species?
train_labels = df['species'].values
class_count = {}
for sample1 in train_labels:
    if sample1 not in class_count:
        class_count[sample1] = 1
    else:
        class_count[sample1] += 1


print(str(len(class_count)) + " classes " + str(len(train_labels)) + " samples.")
#print("-----------------------------")
        
#for label,count in class_count.items():
#    print(label + ": " + str(count))


#Load Keras
from keras.layers import Input, Dense, merge
from keras.models import Model


#Define network
M1 = 32
margin_input = Input(shape=(64,), name='margin_input')
margin_layer = Dense(M1, activation="tanh")(margin_input)

shape_input = Input(shape=(64,), name='shape_input')
shape_layer = Dense(M1, activation="tanh")(shape_input)

texture_input = Input(shape=(64,), name='texture_input')
texture_layer = Dense(M1,activation="tanh")(texture_input)


merge_layer = merge([margin_layer,shape_layer,texture_layer],mode="concat",name='merge_layer')
output_layer = Dense(99,activation="softmax",name='output_layer')(merge_layer)

model = Model(input=[margin_input,shape_input,texture_input],output=output_layer)


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

#Load train data
margin_train = df.filter(regex="margin*").values
shape_train = df.filter(regex="shape*").values
texture_train = df.filter(regex="texture*").values
labels_train = df['species'].values
labels_train = LabelEncoder().fit(labels_train).transform(labels_train)
labels_train = to_categorical(labels_train)

#Train network
model.compile(optimizer='rmsprop', loss='binary_crossentropy')#, loss_weights=[1., 0.2])
model.fit([margin_train, shape_train, texture_train], labels_train, nb_epoch=50, batch_size=32)


#Load test data
df_test = pd.read_csv("../input/test.csv")
margin_test = df_test.filter(regex="margin*").values
shape_test = df_test.filter(regex="shape*").values
texture_test = df_test.filter(regex="texture*").values


#Predict labels
predicted_labels = model.predict([margin_test,shape_test,texture_test])


#Save as csv
df_pred = pd.DataFrame(predicted_labels,index=df_test.pop('id'),columns=set(class_count.keys()))
csv_test = open('submission1.csv','w')
csv_test.write(df_pred.to_csv())


