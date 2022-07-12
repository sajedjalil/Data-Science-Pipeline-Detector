# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from datetime import datetime

from sklearn.model_selection import train_test_split

from google.cloud import storage
from google.cloud import automl_v1beta1 as automl

# Any results you write to the current directory are saved as output.

class AutoMLWrapper():
    def __init__(self, client, project_id, bucket_name, region='us-central1', dataset_display_name=None, model_display_name=None):
        self.client = client
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        self._location_path = self.client.location_path(self.project_id, self.region)
        
        self.dataset = None
        self.dataset_id = None
        self.dataset_display_name = dataset_display_name
        
        self.model = None
        self.model_full_path = None
        self.model_display_name = model_display_name
        
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.dataset_id = dataset.name.split('/')[-1]
        self.dataset_display_name = dataset.display_name
        
    def create_dataset(self, classification_type='MULTICLASS', dataset_display_name=None):
        if not classification_type:
            classification_type = 'MULTICLASS'
        dataset_metadata = {'classification_type': classification_type}
        
        if dataset_display_name:
            self.dataset_display_name = dataset_display_name
            
        if not (dataset_display_name or self.dataset_display_name):
            self.dataset_display_name = f'automldataset_{datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")}'
            
        dataset_info = {
            'display_name': self.dataset_display_name,
            'text_classification_dataset_metadata': dataset_metadata,
        }
        print(f'making new dataset, display name: {self.dataset_display_name}')
        dataset = self.client.create_dataset(self._location_path, dataset_info)
        self.set_dataset(dataset)
        
        print(dataset)
        return dataset
        
    def import_gcs_data(self, gcs_path):
        # Slice off leading slash
        if gcs_path[0] == '/':
            gcs_path = gcs_path[1:]
        data_config = {'gcs_source': { 'input_uris': [f'gs://{self.bucket_name}/{gcs_path}']}}

        print('importing csv data. This may take a moment')
        operation = self.client.import_data(name=self.dataset.name, input_config=data_config)
        print(operation)

        result = operation.result()
        print(result)

        return
    
    def get_dataset_by_display_name(self, dataset_display_name=None):
        if not dataset_display_name:
            dataset_display_name = self.dataset_display_name
            
        if not dataset_display_name:
            print('no dataset_display_name provided. Please set as attribute, or provide in call parameters.')
            return None
        
        filter_ = 'display_name=' + dataset_display_name
        print(f'searching for dataset named: {dataset_display_name}')
        list_datasets_response = self.client.list_datasets(self._location_path, filter_=filter_)

        matching_datasets = []
        for d in list_datasets_response:
            matching_datasets.append(d)
            self.set_dataset(d) # set the most recent dataset as our match
        
        if not matching_datasets:
            print('no matching datasets found')
        else:
            print(f'found {len(matching_datasets)} matching datasets')
            
        return matching_datasets
    
    def set_model(self, model):
        self.model = model
        self.model_full_path = model.name
        self.model_display_name = model.display_name
        
    def get_model_by_display_name(self, model_display_name=None):
        if not model_display_name:
            model_display_name = self.model_display_name
            
        if not model_display_name:
            print('no model_display_name provided. Please set as attribute, or provide in call parameters.')
            return None
        
        filter_ = 'display_name=' + model_display_name
        print(f'searching for model named: {model_display_name}')
        models_list = self.client.list_models(self._location_path, filter_)
        
        matching_models = []
        for model in models_list:
            matching_models.append(model)
            self.set_model(model)
            
        if not matching_models:
            print('no matching models found')
        else:
            print(f'found {len(matching_models)} matching models')
            
        return matching_models
    
    def deploy_model(self):
        if self.model.deployment_state == 2: # because undeployed == 2 for some reason
            print(f'Deploying model: {self.model_display_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
            response = self.client.deploy_model(name=self.model_full_path)
            while self.model.deployment_state == 2:
                time.sleep(120) # check every other minute if deployment is done
                self.get_model_by_display_name() # refresh the model info
            print(f'Finished deploying model: {self.model_display_name} around {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
        else:
            print(f'model {self.model_display_name} is already deployed')

    def undeploy_model(self):
        if self.model.deployment_state == 1:
            print(f'Undeploying model: {self.model_display_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
            response = self.client.undeploy_model(name=self.model_full_path)
            while self.model.deployment_state == 1:
                time.sleep(120) # check every other minute if undeployment is done
                self.get_model_by_display_name() # refresh the model info
            print(f'Finished undeploying model: {self.model_display_name} around {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
        else:
            print(f'model {self.model_display_name} is already undeployed')
        
    # my_model = amw.train_model(dataset.name.split('/')[-1], model_display_name, project_location)
    def train_model(self, model_display_name=None): # dataset_id, model_display_name, project_location):
        
        if model_display_name:
            self.model_display_name = model_display_name
        if not (model_display_name or self.model_display_name):
            self.model_display_name = f'automlmodel_{datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")}'
            
        print(f'making new model with dataset {self.dataset_id}, named {self.model_display_name}')

        # Set model name and model metadata for the dataset.
        model_info = {
            "display_name": self.model_display_name,
            "dataset_id": self.dataset_id,
            "text_classification_model_metadata": {},
        }

        print('creating and training model')
        # Create a model with the model metadata in the region.
        create_model_response = self.client.create_model(self._location_path, model_info)
        print(create_model_response)

        result = create_model_response.result()

        return self.get_model_by_display_name()[0]
    
    def set_prediction_client(self, prediction_client):
        self.prediction_client = prediction_client
    
    # awm.get_predictions(self, test_df, input_col_name='text', ground_truth_col_name='choose_one', threshold=0.5, limit=20)
    def get_predictions(self, test_df, input_col_name, ground_truth_col_name=None, threshold=0.5, limit=None, verbose=False):
        # score | class | text
        predictions_list = []
        correct = 0
        total_test_size = limit if limit else len(test_df)
        
        for i in range(total_test_size):
            row = test_df.iloc[i]
            snippet = row[input_col_name]
            if ground_truth_col_name:
                ground_truth = row[ground_truth_col_name]

            # Set the payload by giving the content and type of the file.
            payload = {"text_snippet": {"content": snippet, "mime_type": "text/plain"}}

            # params is additional domain-specific parameters.
            # currently there is no additional parameters supported.
            params = {}
            response = self.prediction_client.predict(self.model_full_path, payload, params)

            for result in response.payload:
        #         print("Predicted class name: {}".format(result.display_name))
        #         print("Predicted class score: {}".format(result.classification.score))
                if result.classification.score >= threshold:
                    prediction = {'score': result.classification.score, 
                                  'class': result.display_name,
                                  'text' : snippet}
                    if verbose:
                        print(f'[{i}/{total_test_size}] Predicted class: {result.display_name}, score: {result.classification.score}')

                    if ground_truth_col_name:
                        prediction['ground_truth'] = ground_truth
                        if result.display_name == ground_truth:
                            correct += 1 
                    predictions_list.append(prediction)
            time.sleep(0.3)
                    
        if ground_truth_col_name:
            acc = correct / len(predictions_list)
            print(f'accuracy is {acc}')

        predictions_df = pd.DataFrame(predictions_list)
        return predictions_df