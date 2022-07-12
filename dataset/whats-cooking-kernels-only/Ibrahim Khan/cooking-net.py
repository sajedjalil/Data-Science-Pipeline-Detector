# Neural Network for 20-way classification

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import tensorflow as tf
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
tf.logging.set_verbosity(tf.logging.INFO)
# Read datasets
f=open('../input/train.json')
train = json.load(f)
f=open('../input/test.json')
test = json.load(f)

train_data=[x['ingredients'] for x in train]
train_label=[x['cuisine'] for x in train]

test_data=[x['ingredients'] for x in test]
test_id=[x['id'] for x in test]

ing,cui=[],[]
for item in train:
    ing+=[x for x in item['ingredients'] if x not in ing]
    if item['cuisine'] not in cui:
        cui.append(item['cuisine'])

for item in test:
    ing+=[x for x in item['ingredients'] if x not in ing]

map_ing, map_cui={},{}
i=0
for x in ing:
    map_ing[x]=i
    i+=1
i=0
for x in cui:
    map_cui[x]=i
    i+=1

train_val, test_val=[], []

def to_bin(array, new, map_arr):
    for data in array:
        data=[map_arr[x] for x in data]
        temp=np.zeros(len(ing))
        for i in data:
            temp[i]=1
        new.append(temp)
        
to_bin(train_data, train_val, map_ing)
to_bin(test_data, test_val, map_ing)

train_label=[map_cui[x] for x in train_label]

train_val=np.array(train_val)
train_label=np.array(train_label)

test_val=np.array(test_val)
test_id=np.array(test_id)

def cnn_model_fn(features, labels, mode):
	dense1b = tf.layers.dense(inputs=features["x"], units=1024, activation=tf.nn.relu)
	dense2b = tf.layers.dense(inputs=dense1b, units=32, activation=tf.nn.relu)
	dense3b = tf.layers.dense(inputs=dense2b, units=24, activation=tf.nn.relu)
	dense4b = tf.layers.dense(inputs=dense3b, units=24, activation=tf.nn.relu)
	dropout2b = tf.layers.dropout(inputs=dense4b, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout2b, units=20)
	
	predictions = {"classes":tf.argmax(input=logits, axis=1), "probabilities":tf.nn.softmax(logits, name="softmax_tensor")}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=20)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op=optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


train_data = np.asarray(train_val, dtype=np.float32)
train_labels = np.asarray(train_label, dtype=np.int32)
eval_data = np.asarray(test_val, dtype=np.float32)
classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./store")
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
classifier.train(input_fn=train_input_fn, steps=100000)

f=open('sol.csv','w')
writer=csv.writer(f)
writer.writerow(['id','cuisine'])

pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data}, shuffle=False)
pred_results = classifier.predict(input_fn=pred_input_fn)
j=0
a=''
for i in pred_results:
	cat=i['classes']
	for k,v in map_cui.items():
		if v==cat:
			a=k
	writer.writerow([test_id[j],a])
	j+=1
for i in pred_results:
	cat=i['classes']
	for k,v in map_cui.items():
		if v==cat:
			a=k
	writer.writerow([test_id[j],a])
	j+=1
print('complete')