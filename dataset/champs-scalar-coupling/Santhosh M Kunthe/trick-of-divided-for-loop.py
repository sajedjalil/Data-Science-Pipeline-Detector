import pandas as pd

trainingCSVdata = pd.read_csv(
    '../input/train.csv')

# print(trainingCSVdata.head())

# trainingCSVdata.to_html('featureMolecular.html')

testingCSVdata = pd.read_csv(
    '../input/test.csv')
# print(testingCSVdata.head())
# testingCSVdata.to_html('testfatureMolecular.html')

trainingCSVdata = trainingCSVdata.drop('id', axis=1)
print(trainingCSVdata.head())

featureDF = trainingCSVdata.drop('scalar_coupling_constant', axis=1)
labelDF = trainingCSVdata['scalar_coupling_constant']

moleculeNameToReplace = trainingCSVdata['molecule_name']

uniquesInMoleculeNameToReplace = moleculeNameToReplace.unique().tolist()

uniqueWordsDictWordToIndex = dict()
index = 36
for i in range(len(uniquesInMoleculeNameToReplace)):
    index = index + 54
    uniqueWordsDictWordToIndex[uniquesInMoleculeNameToReplace[i]] = index

moleculeNameReplaced = []
for i in range(len(moleculeNameToReplace)):
    moleculeNameReplaced.append(uniqueWordsDictWordToIndex[moleculeNameToReplace[i]])

featureDF = featureDF.drop('molecule_name', axis=1)
featureDF['molecule_name_indexed'] = moleculeNameReplaced
##---------------------


typeNameToReplace = trainingCSVdata['type']

uniquesIntypeNameToReplace = typeNameToReplace.unique().tolist()

uniqueWordsDictWordToIndex = dict()
index = 36
for i in range(len(uniquesIntypeNameToReplace)):
    index = index + 54
    uniqueWordsDictWordToIndex[uniquesIntypeNameToReplace[i]] = index

typeNameReplaced = []
for i in range(len(typeNameToReplace)):
    typeNameReplaced.append(uniqueWordsDictWordToIndex[typeNameToReplace[i]])

featureDF = featureDF.drop('type', axis=1)
featureDF['type_indexed'] = typeNameReplaced
# -------------------------
# featureDF.to_html('featureDF.html')
# pd.DataFrame(labelDF).to_html('labelDF.html')
import numpy as np

featureNP = np.array(featureDF)
labelNP = np.array(labelDF)
labelNP = np.reshape(labelNP, [-1, 1])

from sklearn.utils import shuffle

featureNP, labelNP = shuffle(featureNP, labelNP)

print(featureNP.shape, labelNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float64, shape=[None, 1])

m0a = tf.Variable(tf.zeros(shape=[4, 7], dtype=tf.float64))
m0b = tf.Variable(tf.zeros(shape=[7, 4], dtype=tf.float64))

m = tf.Variable(tf.zeros(shape=[4, 3], dtype=tf.float64))
m1 = tf.Variable(tf.zeros(shape=[3, 2], dtype=tf.float64))
m2 = tf.Variable(tf.zeros(shape=[2, 1], dtype=tf.float64))

x = tf.placeholder(tf.float64, shape=[None, 4])

lr = tf.placeholder(tf.float64)

b0a = tf.Variable(tf.zeros(shape=[7], dtype=tf.float64))
b0b = tf.Variable(tf.zeros(shape=[4], dtype=tf.float64))

b = tf.Variable(tf.zeros(shape=[3], dtype=tf.float64))
b1 = tf.Variable(tf.zeros(shape=[2], dtype=tf.float64))
b2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))

mx_b0a = tf.add(tf.matmul(x, m0a), b0a)
mx_b0b = tf.add(tf.matmul(mx_b0a, m0b), b0b)

mx_b = tf.add(tf.matmul(mx_b0b, m), b)
mx_b1 = tf.add(tf.matmul(mx_b, m1), b1)
mx_b1 = tf.nn.relu(mx_b1)
mx_b2 = tf.add(tf.matmul(mx_b1, m2), b2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print(sess.run(mx_b2, feed_dict={x: featureNP}))

loss = tf.reduce_mean(tf.sqrt(tf.square(mx_b2 - y)))
trainingStep = tf.train.GradientDescentOptimizer(lr).minimize(loss)

for i in range(100):
    print(sess.run([trainingStep, loss], feed_dict={x: featureNP, y: labelNP, lr: 1}))
    print(i, 'a')

for i in range(300):
    print(sess.run([trainingStep, loss], feed_dict={x: featureNP, y: labelNP, lr: 0.1}))
    print(i, 'b')

for i in range(450):
    print(sess.run([trainingStep, loss], feed_dict={x: featureNP, y: labelNP, lr: 0.0000000000000000001}))
    print(i, 'c')

# *-*--*-*-*-*-*-*-*----*-*-*-*-*-*-*-*-*-*---*-*--*-*-*
# testingCSVdata = testingCSVdata.drop('id', axis=1)
# print(testingCSVdata.head())
idList = testingCSVdata['id']
featureDF = testingCSVdata.drop('id', axis=1)

moleculeNameToReplace = testingCSVdata['molecule_name']

uniquesInMoleculeNameToReplace = moleculeNameToReplace.unique().tolist()

uniqueWordsDictWordToIndex = dict()
index = 36
for i in range(len(uniquesInMoleculeNameToReplace)):
    index = index + 54
    uniqueWordsDictWordToIndex[uniquesInMoleculeNameToReplace[i]] = index

moleculeNameReplaced = []
for i in range(len(moleculeNameToReplace)):
    moleculeNameReplaced.append(uniqueWordsDictWordToIndex[moleculeNameToReplace[i]])

featureDF = featureDF.drop('molecule_name', axis=1)
featureDF['molecule_name_indexed'] = moleculeNameReplaced
##---------------------


typeNameToReplace = testingCSVdata['type']

uniquesIntypeNameToReplace = typeNameToReplace.unique().tolist()

uniqueWordsDictWordToIndex = dict()
index = 36
for i in range(len(uniquesIntypeNameToReplace)):
    index = index + 54
    uniqueWordsDictWordToIndex[uniquesIntypeNameToReplace[i]] = index

typeNameReplaced = []
for i in range(len(typeNameToReplace)):
    typeNameReplaced.append(uniqueWordsDictWordToIndex[typeNameToReplace[i]])

featureDF = featureDF.drop('type', axis=1)
featureDF['type_indexed'] = typeNameReplaced
# -------------------------
# featureDF.to_html('featureDF.html')
# pd.DataFrame(labelDF).to_html('labelDF.html')
import numpy as np

featureNP = np.array(featureDF)
print(featureNP.shape)

outputDF = pd.DataFrame()

outputDF['id'] = idList
output = sess.run(mx_b2, feed_dict={x: featureDF})
output = np.array(output).flatten().tolist()
outputDF['scalar_coupling_constant'] = output


outputDF.to_csv('outputDF.csv',index=False)