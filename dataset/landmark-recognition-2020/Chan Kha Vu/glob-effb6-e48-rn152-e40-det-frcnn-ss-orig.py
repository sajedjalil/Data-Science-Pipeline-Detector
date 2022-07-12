"""Baseline kernel for "Google Landmarks Recognition Challenge 2020".

Generates `submission.csv` in Kaggle format. When the number of training images
indicates that the kernel is being run against the public dataset,
simply copies `sample_submission.csv` to allow for quickly starting reruns
on the private dataset. When in a rerun against the private dataset,
makes predictions via retrieval, using DELG TensorFlow SavedModels for global
and local feature extraction.

First, ranks all training images by embedding similarity to each test image.
Then, performs geometric-verification and re-ranking on the `NUM_TO_RERANK`
most similar training images. For a given test image, each class' score is
the sum of the scores of re-ranked training images, and the predicted
class is the one with the highest aggregate score.

NOTE: For speed, this uses `pydegensac` as its RANSAC implementation.
Since the module has no interface for setting random seeds, RANSAC results
and submission scores will vary slightly between reruns.
"""

import copy
import csv
import gc
import operator
import os
import cv2
import pathlib
import shutil

import numpy as np
import PIL
import pydegensac
from scipy import spatial
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub

# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')

# DEBUGGING PARAMS:
NUM_PUBLIC_TRAIN_IMAGES = 1580470 # Used to detect if in session or re-run.
MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.

# Retrieval & re-ranking parameters:
NUM_TO_RERANK = 3
TOP_K = 5 #Number of retrieved images used to make prediction for a test image.

# RANSAC parameters:
MAX_INLIER_SCORE = 30
MAX_REPROJECTION_ERROR = 6.0
MAX_RANSAC_ITERATIONS = 100_000
HOMOGRAPHY_CONFIDENCE = 0.95

# Landmark / Non-Landmark recognition parameters
CHECK_IF_LANDMARK = True
DET_AGGREGATION_METHOD = 'sum' # either 'sum' or 'leaveout'
DET_MAX_BOXES = 100
DET_MIN_SCORE = 0.0
DET_MIN_NEGATIVE_CLASS_SCORE = 0.1
DET_MIN_NEGATIVE_CLASS_RATIO = 0.6

# Rerank with classification
RERANK_WITH_CLASSIFICATION = False

# How to combine re-ranking scores
C_EMB = 1.
C_RANSAC = 1.
C_CLS = 1.
C_DET = 1.

# DELG model:
SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = [
    'input_image:0', 'input_scales:0', 'input_abs_thres:0'
]

# Global feature extraction:
# NUM_EMBEDDING_DIMENSIONS = 2048
# GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,
#                                                 ['global_descriptors:0'])
NUM_EMBEDDING_DIMENSIONS_1 = 512
C_EMBEDDING_1 = 1.0
GLOBAL_MODEL_1_DIR = '../input/save-effb6-gap-arcface-globonly-e48/submission'
GLOBAL_MODEL_1 = tf.saved_model.load(GLOBAL_MODEL_1_DIR)
GLOBAL_FEATURE_EXTRACTION_1_FN = GLOBAL_MODEL_1.signatures['single_scale_original']
GLOBAL_CLASSIFIER_1_FN = GLOBAL_MODEL_1.signatures['classifier']

NUM_EMBEDDING_DIMENSIONS_2 = 512
C_EMBEDDING_2 = 0.7
GLOBAL_MODEL_2_DIR = '../input/save-rn152v2-gem-arcface-globonly-faster-ep40/submission'
GLOBAL_MODEL_2 = tf.saved_model.load(GLOBAL_MODEL_2_DIR)
GLOBAL_FEATURE_EXTRACTION_2_FN = GLOBAL_MODEL_2.signatures['single_scale_original']
GLOBAL_CLASSIFIER_2_FN = GLOBAL_MODEL_2.signatures['classifier']

NUM_EMBEDDING_DIMENSIONS = NUM_EMBEDDING_DIMENSIONS_1 + NUM_EMBEDDING_DIMENSIONS_2

@tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
    ])
def GLOBAL_FEATURE_EXTRACT(image_tensor):
    emb1 = C_EMBEDDING_1 * GLOBAL_FEATURE_EXTRACTION_1_FN(image_tensor)['global_descriptor']
    emb2 = C_EMBEDDING_2 * GLOBAL_FEATURE_EXTRACTION_2_FN(image_tensor)['global_descriptor']
    return tf.nn.l2_normalize(tf.concat([emb1, emb2], axis=0), axis=0)

# Local feature extraction:
LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)
LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(
    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],
    ['boxes:0', 'features:0'])

# Id mapping for classification model
if RERANK_WITH_CLASSIFICATION:
  LABEL_MAPPING_CSV_PATH = '../input/gldv2cleanmeta/gld-v2-clean-id-mapping.csv'
  LABEL_MAPPING_DF = pd.read_csv(LABEL_MAPPING_CSV_PATH)
  SQUEEZED_ID_FROM_LANDMARK_ID = dict([
      (landmark_id, squeezed_id)
      for (landmark_id, squeezed_id) in  zip(
          LABEL_MAPPING_DF['landmark_id'], LABEL_MAPPING_DF['squeezed_id'])
  ])

# load detector
if CHECK_IF_LANDMARK:
  MODULE_HANDLE = "../input/third-party-models/openimages_v4_faster_rcnn_inception_resnet_v2/openimages_v4_faster_rcnn_inception_resnet_v2" 
  OPENIMAGES_DETECTOR = hub.load(MODULE_HANDLE).signatures['default']

  # just an additional safety measure to catch potential bugs
  if DET_AGGREGATION_METHOD == 'leaveout':
    C_DET = None
    
    
def is_landmark(image_tensor,
                max_boxes=10,
                min_score=0.1,
                negative_min_score=0.15,
                negative_min_ratio=0.6):
    """Decides if the image is landmark or not
    """
    positive_classes = ['Building', 'Tower', 'Castle',
                        'Sculpture', 'Skyscraper']
    neutral_classes = ['House', 'Tree', 'Palm tree',
                       'Watercraft', 'Aircraft',
                       'Swimming pool', 'Fountain']

    image_tensor = image_tensor / 255.    
    result = OPENIMAGES_DETECTOR(image_tensor)
    result = {key:value.numpy() for key,value in result.items()}
    
    boxes, class_names, scores = \
        result["detection_boxes"], \
        result["detection_class_entities"], \
        result["detection_scores"]
    
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            class_name = class_names[i].decode("ascii")
            
            # If any object from the positive classes
            # is found, this is a landmark
            if class_name in positive_classes:
                return 1.0
            
            # If object belongs to negative classes,
            # we should consider some more conditions
            if class_name not in positive_classes \
                    and class_name not in neutral_classes:
                if scores[i] >= negative_min_score:
                    bba = (xmax - xmin) * (ymax - ymin)
                    if bba > negative_min_ratio:
                        # Two conditions were met:
                        # - score >= negative_min_score
                        # - area >= negative_min_ratio
                        return 0.0

    # By default, we dont't want to throw out samples.
    # Let's hope that re-ranking will resolve it.
    return 0.5


def to_hex(image_id) -> str:
  return '{0:0{1}x}'.format(image_id, 16)


def get_image_path(subset, image_id):
  name = to_hex(image_id)
  return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],
                      '{}.jpg'.format(name))


def load_image_tensor(image_path, scale_factor=1.0):
  img = np.array(PIL.Image.open(image_path).convert('RGB'))
  if abs(scale_factor - 1.0) > 1e-5:
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
  return tf.convert_to_tensor(img)


def extract_global_features(image_root_dir, is_test=False):
  """Extracts embeddings for all the images in given `image_root_dir`."""
  print(f'Start extract_global_features for {image_root_dir}')
  image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]
  print('- Parsed the directory for images')

  num_embeddings = len(image_paths)
  if MAX_NUM_EMBEDDINGS > 0:
    num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)

  ids = num_embeddings * [None]
  embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))
  is_landmark_scores = np.zeros(shape=(num_embeddings, ))

  for i, image_path in enumerate(image_paths):
    if i >= num_embeddings:
      break
    
    ids[i] = int(image_path.name.split('.')[0], 16)
    image_tensor = load_image_tensor(image_path)
    
    # We check landmarks ONLY for test images
    if CHECK_IF_LANDMARK and is_test:
      # image pyramid  
      is_landmark_scores[i] = is_landmark(tf.expand_dims(
                                              tf.cast(image_tensor, tf.float32),
                                              axis=0),
                                          max_boxes=DET_MAX_BOXES,
                                          min_score=DET_MIN_SCORE,
                                          negative_min_score=DET_MIN_NEGATIVE_CLASS_SCORE,
                                          negative_min_ratio=DET_MIN_NEGATIVE_CLASS_RATIO)

      if DET_AGGREGATION_METHOD == 'leaveout' and is_landmark_scores[i] < 1e-4:
        embeddings[i, :] = 1000. * np.ones(
            shape=(NUM_EMBEDDING_DIMENSIONS, ), dtype=np.float64)
        continue
    
    # features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,
    #                                         DELG_IMAGE_SCALES_TENSOR,
    #                                         DELG_SCORE_THRESHOLD_TENSOR)
    # embeddings[i, :] = tf.nn.l2_normalize(
    #     tf.reduce_sum(features[0], axis=0, name='sum_pooling'),
    #     axis=0,
    #     name='final_l2_normalization').numpy()
    embeddings[i, :] = GLOBAL_FEATURE_EXTRACT(image_tensor).numpy()

  print(f'- Finished embeddings calculation for {image_root_dir}')
  if not is_test:
    return ids, embeddings
  else:
    return ids, embeddings, is_landmark_scores


def extract_local_features(image_path):
  """Extracts local features for the given `image_path`."""

  image_tensor = load_image_tensor(image_path)

  features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,
                                         DELG_SCORE_THRESHOLD_TENSOR,
                                         LOCAL_FEATURE_NUM_TENSOR)

  # Shape: (N, 2)
  keypoints = tf.divide(
      tf.add(
          tf.gather(features[0], [0, 1], axis=1),
          tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()

  # Shape: (N, 128)
  descriptors = tf.nn.l2_normalize(
      features[1], axis=1, name='l2_normalization').numpy()

  return keypoints, descriptors


def get_putative_matching_keypoints(test_keypoints,
                                    test_descriptors,
                                    train_keypoints,
                                    train_descriptors,
                                    max_distance=0.9):
  """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""

  train_descriptor_tree = spatial.cKDTree(train_descriptors)
  _, matches = train_descriptor_tree.query(
      test_descriptors, distance_upper_bound=max_distance)

  test_kp_count = test_keypoints.shape[0]
  train_kp_count = train_keypoints.shape[0]

  test_matching_keypoints = np.array([
      test_keypoints[i,]
      for i in range(test_kp_count)
      if matches[i] != train_kp_count
  ])
  train_matching_keypoints = np.array([
      train_keypoints[matches[i],]
      for i in range(test_kp_count)
      if matches[i] != train_kp_count
  ])

  return test_matching_keypoints, train_matching_keypoints


def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,
                    train_descriptors):
  """Returns the number of RANSAC inliers."""

  test_match_kp, train_match_kp = get_putative_matching_keypoints(
      test_keypoints, test_descriptors, train_keypoints, train_descriptors)

  if test_match_kp.shape[
      0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
    return 0

  try:
    _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                        MAX_REPROJECTION_ERROR,
                                        HOMOGRAPHY_CONFIDENCE,
                                        MAX_RANSAC_ITERATIONS)
  except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
    return 0

  return int(copy.deepcopy(mask).astype(np.float32).sum())


def get_total_score(num_inliers, global_score):
  local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
  return C_RANSAC * local_score + global_score


def rescore_and_rerank_by_num_inliers(test_image_id,
                                      train_ids_labels_and_scores):
  """Returns rescored and sorted training images by local feature extraction."""

  test_image_path = get_image_path('test', test_image_id)
  test_keypoints, test_descriptors = extract_local_features(test_image_path)

  for i in range(len(train_ids_labels_and_scores)):
    train_image_id, label, global_score = train_ids_labels_and_scores[i]

    train_image_path = get_image_path('train', train_image_id)
    train_keypoints, train_descriptors = extract_local_features(
        train_image_path)

    num_inliers = get_num_inliers(test_keypoints, test_descriptors,
                                  train_keypoints, train_descriptors)
    total_score = get_total_score(num_inliers, global_score)
    train_ids_labels_and_scores[i] = (train_image_id, label, total_score)

  train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)

  return train_ids_labels_and_scores


def load_labelmap():
  with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

  return labelmap


def get_prediction_map(test_ids, train_ids_labels_and_scores, is_landmark_scores):
  """Makes dict from test ids and ranked training ids, labels, scores."""

  prediction_map = dict()

  for test_index, test_id in enumerate(test_ids):
    hex_test_id = to_hex(test_id)
    
    # if we got nothing to re-rank, then we will just return empty dict. This can be
    # only possible if CHECK_IF_LANDMARK is True
    if CHECK_IF_LANDMARK:
      if len(train_ids_labels_and_scores[test_index]) == 0:
        prediction_map[hex_test_id] = None
        continue

    aggregate_scores = {}
    for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
      if label not in aggregate_scores:
        aggregate_scores[label] = 0
      aggregate_scores[label] += score

    label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))
    
    # if we're checking for non-landmarks and DET_AGGREGATION_METHOD is sum,
    # adjust the score.
    if CHECK_IF_LANDMARK and DET_AGGREGATION_METHOD == 'sum':
      score += C_DET * is_landmark_scores[test_index]

    prediction_map[hex_test_id] = {'score': score, 'class': label}

  return prediction_map


def get_predictions(labelmap):
  """Gets predictions using embedding similarity and local feature reranking."""
  print('get_predictions started')

  test_ids, test_embeddings, is_landmark_scores = \
      extract_global_features(TEST_IMAGE_DIR, is_test=True)

  train_ids, train_embeddings = \
      extract_global_features(TRAIN_IMAGE_DIR, is_test=False)

  train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

  # Using (slow) for-loop, as distance matrix doesn't fit in memory.
  for test_index in range(test_embeddings.shape[0]):
    
    if CHECK_IF_LANDMARK and DET_AGGREGATION_METHOD == 'leaveout':
      # if the embedding is crazy, which indicates a non-landmark image
      if is_landmark_scores[test_index] < 1e-4:
        train_ids_labels_and_scores[test_index] = []
        continue
        
    distances = spatial.distance.cdist(
        test_embeddings[np.newaxis, test_index, :], train_embeddings,
        'cosine')[0]
    partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]

    nearest = sorted([(train_ids[p], distances[p]) for p in partition],
                     key=lambda x: x[1])

    train_ids_labels_and_scores[test_index] = [
        (train_id, labelmap[to_hex(train_id)], 1. - cosine_distance)
        for train_id, cosine_distance in nearest
    ]
    
    # Rerank with classifier (global score = cosine dist + classification)
    if RERANK_WITH_CLASSIFICATION:
      current_test_classification = GLOBAL_CLASSIFIER_FN(
          tf.convert_to_tensor(test_embeddings[test_index], dtype=tf.float32)
      )['global_classifier'].numpy()
      
      train_ids_labels_and_scores[test_index] = [
          (
            train_id, label, 
            C_EMB * score + C_CLS * current_test_classification[
                SQUEEZED_ID_FROM_LANDMARK_ID[int(label)]]
          )
          for train_id, label, score in train_ids_labels_and_scores[test_index]
      ]
    else:
      train_ids_labels_and_scores[test_index] = [
          (train_id, label, C_EMB * score)
          for train_id, label, score in train_ids_labels_and_scores[test_index]
      ]

  print('finished embedding calculation and rescore-by-classification')
  del test_embeddings
  del train_embeddings
  del labelmap
  gc.collect()

  # pre_verification_predictions = get_prediction_map(
  #     test_ids, train_ids_labels_and_scores)

#  return None, pre_verification_predictions

  for test_index, test_id in enumerate(test_ids):
        
    if CHECK_IF_LANDMARK:
      # if this is a non-landmark, do nothing and leave stuffs empty
      if len(train_ids_labels_and_scores[test_index]) == 0:
        continue
        
    train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(
        test_id, train_ids_labels_and_scores[test_index])
    
  print('finished re-ranking by RANSAC')

  post_verification_predictions = get_prediction_map(
      test_ids, train_ids_labels_and_scores, is_landmark_scores)

  # return pre_verification_predictions, post_verification_predictions
  return None, post_verification_predictions


def save_submission_csv(predictions=None):
  """Saves optional `predictions` as submission.csv.

  The csv has columns {id, landmarks}. The landmarks column is a string
  containing the label and score for the id, separated by a ws delimeter.

  If `predictions` is `None` (default), submission.csv is copied from
  sample_submission.csv in `IMAGE_DIR`.

  Args:
    predictions: Optional dict of image ids to dicts with keys {class, score}.
  """

  if predictions is None:
    # Dummy submission!
    shutil.copyfile(
        os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')
    return

  with open('submission.csv', 'w') as submission_csv:
    csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])
    csv_writer.writeheader()
    for image_id, prediction in predictions.items():
        
      # if we have non-landmark detection logic
      if CHECK_IF_LANDMARK and prediction is None:
        csv_writer.writerow({'id': image_id, 'landmarks': ''})
        continue
        
      label = prediction['class']
      score = prediction['score']
      csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})


def main():
  labelmap = load_labelmap()
  print('Labelmap loaded')
  num_training_images = len(labelmap.keys())
  print(f'Found {num_training_images} training images.')

  if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:
    print('Copying sample submission.')
    save_submission_csv()
    return

  _, post_verification_predictions = get_predictions(labelmap)
  print('Finished')
  save_submission_csv(post_verification_predictions)


if __name__ == '__main__':
  main()