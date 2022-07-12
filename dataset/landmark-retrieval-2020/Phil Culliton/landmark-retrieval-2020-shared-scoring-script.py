## We have removed the sensitive portions of this script, and included those
## that show you how we:
## 1. Load your model
## 2. Create embeddings
## 3. Compare and score those embeddings.
##
## Note that this means this code will NOT run as-is.

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
import time
from scipy.spatial import distance

import solution
import metrics

REQUIRED_SIGNATURE = 'serving_default'
REQUIRED_OUTPUT = 'global_descriptor'

DATASET_DIR = '' # path to internal dataset

SAVED_MODELS_DIR = os.path.join('kaggle', 'input')
QUERY_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
INDEX_IMAGE_DIR = os.path.join(DATASET_DIR, 'index')
SOLUTION_PATH = ''

def to_hex(image_id: int) -> str:
    return '{0:0{1}x}'.format(image_id, 16)


def show_elapsed_time(start):
    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []

    if hours > 0:
        parts.append('{:>02}h'.format(hours))

    if minutes > 0:
        parts.append('{:>02}m'.format(minutes))

    parts.append('{:>05.2f}s'.format(seconds))

    print('Elapsed Time: {}'.format(' '.join(parts)))


def get_distance(scored_prediction):
    return scored_prediction[1]

embedding_fn = None

def get_embedding(image_path: Path) -> np.ndarray:
    image_data = np.array(Image.open(str(image_path)).convert('RGB'))
    image_tensor = tf.convert_to_tensor(image_data)
    return embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()


class Submission:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        public_solution, private_solution, ignored_ids = solution.load(SOLUTION_PATH, 
                                                         solution.RETRIEVAL_TASK_ID)
        predictions = self.get_predictions()
        
        self.private_score = self.get_metrics(predictions, private_solution)
        self.public_score = self.get_metrics(predictions, public_solution)

    def load(self, saved_model_proto_filename):
        saved_model_path = Path(saved_model_proto_filename).parent
        
        print (saved_model_path, saved_model_proto_filename)
        
        name = saved_model_path.relative_to(SAVED_MODELS_DIR)
        
        model = tf.saved_model.load(str(saved_model_path))
        
        found_signatures = list(model.signatures.keys())
        
        if REQUIRED_SIGNATURE not in found_signatures:
            return None
        
        outputs = model.signatures[REQUIRED_SIGNATURE].structured_outputs
        if REQUIRED_OUTPUT not in outputs:
            return None
        
        global embedding_fn
        embedding_fn = model.signatures[REQUIRED_SIGNATURE]

        return Submission(name, model)
    

    def get_id(self, image_path: Path):
        return int(image_path.name.split('.')[0], 16)


    def get_embeddings(self, image_root_dir: str):
        image_paths = [p for p in Path(image_root_dir).rglob('*.jpg')]
        
        embeddings = [get_embedding(image_path) 
                      for i, image_path in enumerate(image_paths)]
        ids = [self.get_id(image_path) for image_path in image_paths]

        return ids, embeddings
    
    def get_predictions(self):
        print('Embedding queries...')
        start = time.time()
        query_ids, query_embeddings = self.get_embeddings(QUERY_IMAGE_DIR)
        show_elapsed_time(start)

        print('Embedding index...')
        start = time.time()
        index_ids, index_embeddings = self.get_embeddings(INDEX_IMAGE_DIR)
        show_elapsed_time(start)

        print('Computing distances...', end='\t')
        start = time.time()
        distances = distance.cdist(np.array(query_embeddings), 
                                   np.array(index_embeddings), 'euclidean')
        show_elapsed_time(start)

        print('Finding NN indices...', end='\t')
        start = time.time()
        predicted_positions = np.argpartition(distances, K, axis=1)[:, :K]
        show_elapsed_time(start)

        print('Converting to dict...', end='\t')
        predictions = {}
        for i, query_id in enumerate(query_ids):
            nearest = [(index_ids[j], distances[i, j]) 
                       for j in predicted_positions[i]]
            nearest.sort(key=lambda x: x[1])
            prediction = [to_hex(index_id) for index_id, d in nearest]
            predictions[to_hex(query_id)] = prediction
        show_elapsed_time(start)

        return predictions
    
    def get_metrics(self, predictions, solution):
        relevant_predictions = {}

        for key in solution.keys():
            if key in predictions:
                relevant_predictions[key] = predictions[key]

        # Mean average precision.
        mean_average_precision = metrics.MeanAveragePrecision(
            relevant_predictions, solution, max_predictions=K)
        print('Mean Average Precision (mAP): {:.4f}'.format(mean_average_precision))

        return mean_average_precision
    
## after unpacking your zipped submission to /kaggle/working, the saved_model.pb
## file and attendant directory structure are passed to the the Submission object
## for loading.

# submission_object = Submission.load("/kaggle/working/saved_model.pb")