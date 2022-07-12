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
import pathlib
import shutil

import numpy as np
import PIL
import pydegensac
from scipy import spatial
import tensorflow as tf

# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')

# DEBUGGING PARAMS:
NUM_PUBLIC_TRAIN_IMAGES = 1580470  # Used to detect if in session or re-run.
MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.

# Retrieval & re-ranking parameters:
NUM_TO_RERANK = 10
TOP_K = 5  # Number of retrieved images used to make prediction per test image.

# RANSAC parameters:
MAX_INLIER_SCORE = 70
MAX_REPROJECTION_ERROR = 4.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 0.99

# DELG model:
SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = ['input_image:0', 'input_scales:0']

# Global feature extraction:
NUM_EMBEDDING_DIMENSIONS = 2048
GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,
                                                ['global_descriptors:0'])

# Local feature extraction:
LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)
LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(
    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0', 'input_abs_thres:0'],
    ['boxes:0', 'features:0'])


def to_hex(image_id) -> str:
  return '{0:0{1}x}'.format(image_id, 16)


def get_image_path(subset, image_id):
  name = to_hex(image_id)
  return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],
                      '{}.jpg'.format(name))


def load_image_tensor(image_path):
  return tf.convert_to_tensor(
      np.array(PIL.Image.open(image_path).convert('RGB')))


def extract_global_features(image_root_dir):
  """Extracts embeddings for all the images in given `image_root_dir`."""

  image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]

  num_embeddings = len(image_paths)
  if MAX_NUM_EMBEDDINGS > 0:
    num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)

  ids = num_embeddings * [None]
  embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))

  for i, image_path in enumerate(image_paths):
    if i >= num_embeddings:
      break

    ids[i] = int(image_path.name.split('.')[0], 16)

    image_tensor = load_image_tensor(image_path)

    embedding_tensor = tf.nn.l2_normalize(
        GLOBAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR)[0],
        axis=1,
        name='l2_normalization')
    embedding_tensor = tf.reduce_sum(
        embedding_tensor, axis=0, name='sum_pooling')
    embeddings[i, :] = tf.nn.l2_normalize(
        embedding_tensor, axis=0, name='final_l2_normalization').numpy()

  return ids, embeddings


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


def compute_putative_matching_keypoints(test_keypoints,
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


def compute_num_inliers(test_keypoints, test_descriptors, train_keypoints,
                        train_descriptors):
  """Returns the number of RANSAC inliers."""

  test_match_kp, train_match_kp = compute_putative_matching_keypoints(
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
  return local_score + global_score


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

    num_inliers = compute_num_inliers(test_keypoints, test_descriptors,
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


def get_prediction_map(test_ids, train_ids_labels_and_scores):
  """Makes dict from test ids and ranked training ids, labels, scores."""

  prediction_map = dict()

  for test_index, test_id in enumerate(test_ids):
    hex_test_id = to_hex(test_id)

    aggregate_scores = {}
    for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
      if label not in aggregate_scores:
        aggregate_scores[label] = 0
      aggregate_scores[label] += score

    label, score = max(aggregate_scores.iteritems(), key=operator.itemgetter(1))

    prediction_map[hex_test_id] = {'score': score, 'class': label}

  return prediction_map


def get_predictions(labelmap):
  """Gets predictions using embedding similarity and local feature reranking."""

  test_ids, test_embeddings = extract_global_features(TEST_IMAGE_DIR)

  train_ids, train_embeddings = extract_global_features(TRAIN_IMAGE_DIR)

  train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

  # Using (slow) for-loop, as distance matrix doesn't fit in memory.
  for test_index in range(test_embeddings.shape[0]):
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

  del test_embeddings
  del train_embeddings
  del labelmap
  gc.collect()

  pre_verification_predictions = get_prediction_map(
      test_ids, train_ids_labels_and_scores)

  for test_index, test_id in enumerate(test_ids):
    train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(
        test_id, train_ids_labels_and_scores[test_index])

  post_verification_predictions = get_prediction_map(
      test_ids, train_ids_labels_and_scores)

  return pre_verification_predictions, post_verification_predictions


def save_submission_csv(predictions=None):
  """Saves optional `predictions` as submission.csv.

  The csv has columns {id, landmarks}. The landmarks column is a string
  containing the label and score for the id, separated by a ws delimeter.

  If `predictions` is "None" (default), submission.csv is copied from
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
      label = prediction['class']
      score = prediction['score']
      csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})


def main():
  labelmap = load_labelmap()
  num_training_images = len(labelmap.keys())
  print(f'Found {num_training_images} training images.')

  if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:
    print('Found NUM_PUBLIC_TRAIN_IMAGES. Copying sample submission.')
    save_submission_csv()
    return

  _, post_verification_predictions = get_predictions(labelmap)
  save_submission_csv(post_verification_predictions)


if __name__ == '__main__':
  main()