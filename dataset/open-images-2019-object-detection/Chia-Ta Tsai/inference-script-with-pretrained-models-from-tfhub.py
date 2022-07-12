"""
reference from 
https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb
https://www.kaggle.com/xhlulu/intro-to-tf-hub-for-object-detection
https://www.kaggle.com/vikramtiwari/baseline-predictions-using-inception-resnet-v2
"""

import os
import gc
gc.enable()
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from six import BytesIO
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps


def form_one_prediction_strings(result, i):
    class_name = result['detection_class_names'][i].decode("utf-8")
    boxes = result['detection_boxes'][i]
    score = result['detection_scores'][i]
    return f"{class_name} {score} " + " ".join(map(str, boxes))


def format_prediction_string(detected):
    image_id, result = detected
    prediction_strings = [form_one_prediction_strings(result, i) for i in range(len(result['detection_scores']))]
    return {
        "ImageID": image_id,
        "PredictionString": " ".join(prediction_strings)
    }


def inference_one_chunk(data_path, list_image_ids, session, result, image_string_placeholder, predictions):
    img_files = {
        i: tf.gfile.Open(
            os.sep.join([data_path, 'test', f'{i}.jpg']), "rb").read() for i in list_image_ids}
    
    for image_id in tqdm(list_image_ids):
        result_out = session.run(
            result, feed_dict={image_string_placeholder: img_files[image_id]})

        predictions.append((image_id, result_out))
        
    del img_files
    gc.collect()
    return


def inference():
    
    # load model
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            detector = hub.Module(module_handle)
            image_string_placeholder = tf.placeholder(tf.string)
            decoded_image = tf.image.decode_jpeg(image_string_placeholder)
            # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
            # of size 1 and type tf.float32.
            decoded_image_float = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
            module_input = tf.expand_dims(decoded_image_float, 0)
            result = detector(module_input, as_dict=True)
            init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

            session = tf.Session()
            session.run(init_ops)

    # load test tset index
    data_path = "../input"
    sample_submission_df = pd.read_csv(f'{data_path}/sample_submission.csv')
    image_ids = sample_submission_df['ImageId']

    predictions = []
    with tf.device('/device:GPU:0'):
        step = 10000
        for ii in range(0, len(image_ids), step):
            list_image_ids = image_ids.tolist()[ii: ii+step]
            inference_one_chunk(data_path, list_image_ids, session, result, image_string_placeholder, predictions)
    
    # post processing and save
    predictions_df = pd.DataFrame(list(map(format_prediction_string, predictions)))
    predictions_df.to_csv('submission.csv', index=False)
    session.close()


if "__main__" == __name__:
    inference()