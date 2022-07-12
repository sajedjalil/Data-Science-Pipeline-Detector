# %% [code]
"""Baseline kernel for "Google Landmarks Retrieval Challenge 2021".

Generates `submission.csv` in Kaggle format. When the number of index images
indicates that the kernel is being run against the public dataset,
simply copies `sample_submission.csv` to allow for quickly starting reruns
on the private dataset. When in a rerun against the private dataset,
makes predictions via retrieval, using DELG TensorFlow SavedModels for global
and local feature extraction.

First, ranks all index images by embedding similarity to each test image.
Then, performs geometric-verification and re-ranking on the `NUM_TO_RERANK`
most similar index images. For a given test image, each class' score is
the sum of the scores of re-ranked index images, and the predicted
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
from time import time
import datetime as dt

import numpy as np
import PIL
import pydegensac
from scipy import spatial
import tensorflow as tf
import humanize

from metric_util import Metrics
import solution

# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-retrieval-2021')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
INDEX_IMAGE_DIR = os.path.join(DATASET_DIR, 'index')
INDEX_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')

# DEBUGGING PARAMS:
DRY_RUN_ENABLED = False # Set to true to quickly submit in public environment.
NUM_PUBLIC_INDEX_IMAGES = 1580470  # Used to detect if in session or re-run.
MAX_NUM_EMBEDDINGS = -1  # Set to > 0 to subsample dataset while debugging.

# Retrieval & re-ranking parameters:
NUM_TO_RERANK = 50
TOP_K = 5 # Number of retrieved images used to make prediction for a test image.

# RANSAC parameters:
MAX_INLIER_SCORE = 10
MAX_REPROJECTION_ERROR = 4.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 0.99

class SavedModel:
    def __init__(self, name, path, num_dimensions):
        self.name = name
        self.path = path
        self.num_dimensions = num_dimensions
print(os.listdir('../input'))
DELG_RESNET101 = SavedModel('DELG (ResNet-101)', '../input/delg101/delg_101', 2048)
# DELG_RESNET50 = SavedModel('DELG (ResNet-50)', '../input/saved-models/saved_models/delg_resnet50', 2048)
# GLOF = SavedModel('GLOF', '../input/saved-models/saved_models/glof', 128)

SAVED_MODEL = DELG_RESNET101

print(f'Loading model "{SAVED_MODEL.name}"...')
start = time()
DELG_MODEL = tf.saved_model.load(SAVED_MODEL.path)
print(f'Loaded in {humanize.precisedelta(dt.timedelta(seconds=time() - start))}.')

DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([1.0])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = [
    'input_image:0', 'input_scales:0', 'input_abs_thres:0'
]

# Global feature extraction:
GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.signatures['serving_default']
# Output `submission.csv`:
SUBMISSION_CSV = os.path.join('/kaggle', 'working', 'submission.csv')


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
  embeddings = np.empty((num_embeddings, SAVED_MODEL.num_dimensions))

  for i, image_path in enumerate(image_paths):
    if i >= num_embeddings:
      break

    ids[i] = int(image_path.name.split('.')[0], 16)
    image_tensor = load_image_tensor(image_path)
    
    features = GLOBAL_FEATURE_EXTRACTION_FN(input_max_feature_num=tf.convert_to_tensor(100), input_abs_thres=tf.convert_to_tensor(175.0), input_scales=DELG_IMAGE_SCALES_TENSOR, input_global_scales_ind=tf.range(1), input_image=image_tensor)['global_descriptors']

    embeddings[i, :] = tf.nn.l2_normalize(
        tf.reduce_sum(features, axis=0, name='sum_pooling'),
        axis=0,
        name='final_l2_normalization').numpy()

  return ids, embeddings


def extract_local_features(image_path):
  """Extracts local features for the given `image_path`."""

  image_tensor = load_image_tensor(image_path)

  features = LOCAL_FEATURE_EXTRACTION_FN(input_image=image_tensor)

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
                                    index_keypoints,
                                    index_descriptors,
                                    max_distance=0.9):
  """Finds matches from `test_descriptors` to KD-tree of `index_descriptors`."""

  index_descriptor_tree = spatial.cKDTree(index_descriptors)
  _, matches = index_descriptor_tree.query(
      test_descriptors, distance_upper_bound=max_distance)

  test_kp_count = test_keypoints.shape[0]
  index_kp_count = index_keypoints.shape[0]

  test_matching_keypoints = np.array([
      test_keypoints[i,]
      for i in range(test_kp_count)
      if matches[i] != index_kp_count
  ])
  index_matching_keypoints = np.array([
      index_keypoints[matches[i],]
      for i in range(test_kp_count)
      if matches[i] != index_kp_count
  ])

  return test_matching_keypoints, index_matching_keypoints


def get_num_inliers(test_keypoints, test_descriptors, index_keypoints,
                    index_descriptors):
  """Returns the number of RANSAC inliers."""

  test_match_kp, index_match_kp = get_putative_matching_keypoints(
      test_keypoints, test_descriptors, index_keypoints, index_descriptors)

  if test_match_kp.shape[
      0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
    return 0

  try:
    _, mask = pydegensac.findHomography(test_match_kp, index_match_kp,
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
                                      index_ids_labels_and_scores):
  """Returns rescored and sorted index images by local feature extraction."""

  test_image_path = get_image_path('test', test_image_id)
  test_keypoints, test_descriptors = extract_local_features(test_image_path)

  for i in range(len(index_ids_labels_and_scores)):
    index_image_id, label, global_score = index_ids_labels_and_scores[i]

    index_image_path = get_image_path('index', index_image_id)
    index_keypoints, index_descriptors = extract_local_features(
        index_image_path)

    num_inliers = get_num_inliers(test_keypoints, test_descriptors,
                                  index_keypoints, index_descriptors)
    total_score = get_total_score(num_inliers, global_score)
    index_ids_labels_and_scores[i] = (index_image_id, label, total_score)

  index_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)

  return index_ids_labels_and_scores


def load_labelmap():
  with open(INDEX_LABELMAP_PATH, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

  return labelmap


def get_prediction_map(test_ids, index_ids_labels_and_scores):
  """Makes dict from test ids and ranked index ids, labels, scores."""

  prediction_map = dict()

  for test_index, test_id in enumerate(test_ids):
    hex_test_id = to_hex(test_id)

    aggregate_scores = {}
    for index_id, label, score in index_ids_labels_and_scores[test_index][:TOP_K]:
      if label not in aggregate_scores:
        aggregate_scores[label] = 0
      aggregate_scores[label] += score

    label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))

    prediction_map[hex_test_id] = {'score': score, 'class': label}

  return prediction_map


def get_predictions(labelmap):
  """Gets predictions using embedding similarity and local feature reranking."""
  print('Embedding test...')
  s = time()
  test_ids, test_embeddings = extract_global_features(TEST_IMAGE_DIR)
  test_time = time() - s
  print(f'Extracted global features for {len(test_ids)} test images in {humanize.precisedelta(dt.timedelta(seconds=test_time))}.')

  print('Embedding index...')
  s = time()
  index_ids, index_embeddings = extract_global_features(INDEX_IMAGE_DIR)
  index_time = time() - s
  print(f'Extracted global features for {len(index_ids)} index images in {humanize.precisedelta(dt.timedelta(seconds=index_time))}.')

  print('Computing distances...')
  s = time()
  distances = spatial.distance.cdist(np.array(test_embeddings), np.array(index_embeddings), 'cosine')
  distance_time = time() - s
  print(f'Distances computed in {humanize.precisedelta(dt.timedelta(seconds=distance_time))}.')

  print(f'Finding NN indices...')
  s = time()
  predicted_positions = np.argpartition(distances, TOP_K, axis=1)[:, :TOP_K]
  nn_time = time() - s
  print(f'Found NN indices in {humanize.precisedelta(dt.timedelta(seconds=nn_time))}.')

  print('Getting predictions...')
  s = time()
  predictions = []
  for i, test_id in enumerate(test_ids):
    nearest = [(index_ids[j], distances[i, j]) for j in predicted_positions[i]]
    nearest.sort(key=lambda x: x[1])
    prediction = {'id': to_hex(test_id), 'images': ' '.join([to_hex(index_id) for index_id, d in nearest])} 
    predictions.append(prediction)

  predictions.sort(key=lambda p: p['id'])
  prediction_time = time() - s
  print(f'Got predictions in {humanize.precisedelta(dt.timedelta(seconds=prediction_time))}.')

  total_time = test_time + index_time + distance_time + nn_time + prediction_time
  print(f'Total time: {humanize.precisedelta(dt.timedelta(seconds=total_time))}.')

  return predictions


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
        os.path.join(DATASET_DIR, 'sample_submission.csv'), SUBMISSION_CSV)
    return

  with open(SUBMISSION_CSV, 'w') as submission_csv:
    csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'images'])
    csv_writer.writeheader()
    csv_writer.writerows(predictions)


def main():
  labelmap = load_labelmap()
  num_index_images = len(labelmap.keys())
  print(f'Found {num_index_images} index images.')

  if DRY_RUN_ENABLED and num_index_images == NUM_PUBLIC_INDEX_IMAGES:
    print(
        f'Num index images matches public dataset. Copying sample submission.'
    )
    save_submission_csv()
    return

  predictions = get_predictions(labelmap)
  save_submission_csv(predictions)
  public_solution, private_solution, ignored_ids = solution.load('../input/retrieval-solution/solution.csv', solution.RETRIEVAL_TASK_ID)
  metrics = Metrics(predictions, public_solution)
  print(metrics)

if __name__ == '__main__':
  main()
