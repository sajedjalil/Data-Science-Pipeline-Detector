### Baseline DELG global feature for retrieval ###

### Code Adaptived from the Original Kernal ###
### https://www.kaggle.com/camaskew/host-baseline-example?scriptVersionId=40287695


import os
import pandas as pd
import numpy as np
import PIL
from scipy import spatial
import tensorflow as tf

INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-retrieval-2021')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'index')

TOP_K = 100

# DELG model:
SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_INPUT_TENSOR_NAMES = ['input_image:0', 'input_scales:0']

# Global feature extraction:
NUM_EMBEDDING_DIMENSIONS = 2048
GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES, ['global_descriptors:0'])

def to_hex(image_id) -> str:
    return '{0:0{1}x}'.format(image_id, 16)

def get_image_path(subset, image_id):
    name = to_hex(image_id)
    return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2], '{}.jpg'.format(name))

def load_image_tensor(image_path):
    return tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB')))

def extract_global_features(image_root_dir):
    image_paths = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith('.jpg'):
                 image_paths.append(os.path.join(root, file))
                    
    num_embeddings = len(image_paths)

    ids = num_embeddings * [None]
    ids = []
    for path in image_paths:
        ids.append(path.split('/')[-1][:-4])
    
    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))
    for i, image_path in enumerate(image_paths):
        image_tensor = load_image_tensor(image_path)
        embedding_tensor = tf.nn.l2_normalize(GLOBAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR)[0], axis=1, name='l2_normalization')
        embedding_tensor = tf.reduce_sum(embedding_tensor, axis=0, name='sum_pooling')
        embeddings[i, :] = tf.nn.l2_normalize(embedding_tensor, axis=0, name='final_l2_normalization').numpy()

    return ids, embeddings

def get_predictions():
    test_ids, test_embeddings = extract_global_features(TEST_IMAGE_DIR)
    train_ids, train_embeddings = extract_global_features(TRAIN_IMAGE_DIR)

    PredictionString_list = []
    for test_index in range(test_embeddings.shape[0]):
        distances = spatial.distance.cdist(test_embeddings[np.newaxis, test_index, :], train_embeddings, 'cosine')[0]
        partition = np.argpartition(distances, TOP_K)[:TOP_K]
        nearest = sorted([(train_ids[p], distances[p]) for p in partition], key=lambda x: x[1])
        pred_str = ""
        for train_id, cosine_distance in nearest:
            pred_str += train_id
            pred_str += " "
        PredictionString_list.append(pred_str)

    return test_ids, PredictionString_list

def main():
    test_image_list = []
    for root, dirs, files in os.walk(TEST_IMAGE_DIR):
        for file in files:
            if file.endswith('.jpg'):
                 test_image_list.append(os.path.join(root, file))
                    
    if len(test_image_list)==1129:
        sub_df = pd.read_csv('../input/landmark-retrieval-2021/sample_submission.csv')
        sub_df.to_csv('submission.csv', index=False)
        return
    
    test_ids, PredictionString_list = get_predictions()
    sub_df = pd.DataFrame(data={'id': test_ids, 'images': PredictionString_list})
    sub_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()