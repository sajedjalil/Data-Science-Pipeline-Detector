#! -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd

# Read data.
cities = pd.read_csv("../input/cities.csv").values
CityID = np.array(cities[:, 0], np.int32)
XY = np.array(cities[:, 1:], np.float32)
path = np.array([0], np.int32)

# Build graph.
tf_path_ph = tf.placeholder(tf.int32, shape=(None))

# XY pairs.
tf_XY = tf.constant(XY)

# Current city ID.
tf_city_id = tf.Variable(0)

# Get current city's XY and apply tile on it.
tf_xy_tile = tf.tile(
	tf.expand_dims(
		tf_XY[tf_city_id], 0
	),
	[CityID.shape[0], 1]
)

# Take distance of all cities relative to current city.
tf_distances = tf.norm(tf_XY-tf_xy_tile, ord="euclidean", axis=1)

# Sort distances.
tf_top_k = tf.nn.top_k(tf_distances, k=CityID.shape[0], sorted=True)
tf_distances_top_k = tf_top_k.values[::-1]
tf_indices_top_k = tf_top_k.indices[::-1]

# Find which one isn't itself and not processed before.
tf_minIndex = tf.Variable(0)
while_cond = lambda tf_minIndex: tf.cast(
	tf.bitwise.bitwise_or(
		tf.cast(
			tf.less_equal(
				tf_distances_top_k[tf_minIndex],
				0
			),
			tf.int32
		),
		tf.cast(
			tf.greater(
				tf.reduce_sum(
					tf.cast(
						tf.equal(
							tf_path_ph,
							tf_indices_top_k[tf_minIndex]
						),
						tf.int32
					)
				),
				0
			),
			tf.int32
		)
	),
	tf.bool
)

while_body = lambda tf_minIndex: tf.add(tf_minIndex, 1)
tf_minIndex_while = tf.while_loop(while_cond, while_body, [tf_minIndex])

# Final outputs for current process.
tf_finalDistance = tf_distances_top_k[tf_minIndex_while]
tf_finalIndex = tf_indices_top_k[tf_minIndex_while]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

currentCity = path[0]
while True:
	distance, nearestCity = sess.run(
		[tf_finalDistance, tf_finalIndex],
		feed_dict={
			tf_city_id:currentCity,
			tf_minIndex:0,
			tf_path_ph:path
		}
	)

	if (path.shape[0]%1000) == 0:
		print("[%s/%s] City %s -> City %s | Distance %s" % (
				path.shape[0],
				CityID.shape[0],
				currentCity,
				nearestCity,
				distance
			)
		)

	# Add processed city to path.
	path = np.concatenate([path, [nearestCity]], axis=0)
	currentCity = nearestCity
	if path.shape[0] == CityID.shape[0]:
		break

# Submission.
path = np.concatenate([path, [0]])
submission_df = pd.DataFrame({"Path": path})
submission_df.to_csv("path.csv", index=False)