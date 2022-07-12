import google.auth
from google.auth.transport.requests import AuthorizedSession
import json


#The url is : https://siim.org/resource/resmgr/community/train-rle.csv <<<<<<<<<<<<<<<<<<<<<<<<<<


"""
#IMPORTANT README#
# If you get "permission denied" PLEASE Just join the group : https://groups.google.com/forum/#!forum/siim-kaggle
#And do not forget to allow the Google Healthcare API
#If you get any difficulties refer to the official tutorial : https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/siim-cloud-healthcare-api-tutorial
#Be sure to run the command 'gcloud auth application-default login' in your terminal as well
#####################
"""

def get_FHIR_annotations():
    PROJECT_ID ='kaggle-siim-healthcare'
    REGION = 'us-central1'
    DATASET_ID = 'siim-pneumothorax'
    FHIR_STORE_ID = 'fhir-masks-train'
    DOCUMENT_REFERENCE_ID = 'd70d8f3e-990a-4bc0-b11f-c87349f5d4eb'
    resource_type ='DocumentReference'
    base_url = 'https://healthcare.googleapis.com/v1beta1'
    url = '{}/projects/{}/locations/{}'.format(base_url,
                                               PROJECT_ID, REGION)

    resource_path = '{}/datasets/{}/fhirStores/{}/fhir/{}/{}'.format(
        url, DATASET_ID, FHIR_STORE_ID, resource_type, DOCUMENT_REFERENCE_ID )

    credentials, _ = google.auth.default()
    authed_session = AuthorizedSession(credentials)
    response = authed_session.get(resource_path)
    response.raise_for_status()

    resource = response.json()
    print(json.dumps(resource, indent=2))
    print(resource)
    print('url to download Label files :{}'.format(str(resource['content'][0]['attachment']['url'])))
    return resource

# get_FHIR_annotations() <<<<<<<<<<<<<<<<<<<<<<<<<<<<- UNCOMMENT ME TO RETRIEVE THE URL USING GCP API CALL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<