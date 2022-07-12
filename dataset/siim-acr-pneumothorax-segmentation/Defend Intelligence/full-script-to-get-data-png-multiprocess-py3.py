import posixpath
from concurrent import futures
from retrying import retry
import google.auth
from google.auth.transport.requests import AuthorizedSession
import shutil
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pydicom
from multiprocessing import Pool
from functools import partial

# URL of CHC API
CHC_API_URL = 'https://healthcare.googleapis.com/v1beta1'
PROJECT_ID = 'kaggle-siim-healthcare'
REGION = 'us-central1'
DATASET_ID = 'siim-pneumothorax'
TRAIN_DICOM_STORE_ID = 'dicom-images-train'
TEST_DICOM_STORE_ID = 'dicom-images-test'


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def download_instance(dicom_web_url, dicom_store_id, study_uid, series_uid,
                      instance_uid, credentials):
    """Downloads a DICOM instance and saves it under the current folder."""
    instance_url = posixpath.join(dicom_web_url, 'studies', study_uid, 'series',
                                  series_uid, 'instances', instance_uid)
    authed_session = AuthorizedSession(credentials)
    response = authed_session.get(
        instance_url, headers={'Accept': 'application/dicom; transfer-syntax=*'})
    file_path = posixpath.join(dicom_store_id, study_uid, series_uid,
                               instance_uid)
    filename = '%s.dcm' % file_path
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(response.content)

"""Script to download all instances in a DICOM Store."""
def download_all_instances(dicom_store_id, credentials):
    """Downloads all DICOM instances in the specified DICOM store."""
    # Get a list of all instances.
    dicom_web_url = posixpath.join(CHC_API_URL, 'projects', PROJECT_ID,
                                   'locations', REGION, 'datasets', DATASET_ID,
                                   'dicomStores', dicom_store_id, 'dicomWeb')
    qido_url = posixpath.join(dicom_web_url, 'instances')
    authed_session = AuthorizedSession(credentials)
    response = authed_session.get(qido_url, params={'limit': '15000'})
    if response.status_code != 200:
        print(response.text)
        return
    content = response.json()
    # DICOM Tag numbers
    study_instance_uid_tag = '0020000D'
    series_instance_uid_tag = '0020000E'
    sop_instance_uid_tag = '00080018'
    value_key = 'Value'
    with futures.ThreadPoolExecutor() as executor:
        future_to_study_uid = {}
        for instance in content:
            study_uid = instance[study_instance_uid_tag][value_key][0]
            series_uid = instance[series_instance_uid_tag][value_key][0]
            instance_uid = instance[sop_instance_uid_tag][value_key][0]
            future = executor.submit(download_instance, dicom_web_url, dicom_store_id,
                                     study_uid, series_uid, instance_uid, credentials)
            future_to_study_uid[future] = study_uid
        processed_count = 0
        for future in futures.as_completed(future_to_study_uid):
            try:
                future.result()
                processed_count += 1
                if not processed_count % 100 or processed_count == len(content):
                    print('Processed instance %d out of %d' %
                          (processed_count, len(content)))
            except Exception as e:
                print('Failed to download a study. UID: %s \n exception: %s' %
                      (future_to_study_uid[future], e))

def get_data_from_gcp(argv=None):
    credentials, _ = google.auth.default()
    print('Downloading all instances in %s DICOM store' % TRAIN_DICOM_STORE_ID)
    download_all_instances(TRAIN_DICOM_STORE_ID, credentials)
    print('Downloading all instances in %s DICOM store' % TEST_DICOM_STORE_ID)
    download_all_instances(TEST_DICOM_STORE_ID, credentials)

def move_data():
    if not os.path.exists('../input'):
        os.makedirs('../input')
    for type in ['train','test']:
        inputdir = f'dicom-images-{type}'
        outdir = f'../input/{type}/'
        files = [f for f in glob.glob(inputdir+f'/**/*.dcm',recursive=True)]
        for filename in tqdm(files):
            fname = os.path.basename(filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            shutil.copy(str(filename), os.path.join(outdir, fname))


def convert(inputdir,outputdir,file):
    fname = os.path.basename(file)
    ds = pydicom.read_file(inputdir + fname)  # read dicom image
    img = ds.pixel_array  # get image array
    img_mem = Image.fromarray(img)  # Creates an image memory from an object exporting the array interface
    img_mem.save(outputdir + fname.replace('.dcm', '.png'))

if __name__=='__main__':
    """
    DOWNLOAD ME AND RUN THE SCRIPT ON YOUR COMPUTER
    Do not forget to up vote if it was useful for you :-)
    """
    
    """
    #IMPORTANT README#
    # If you get "permission denied" PLEASE Just join the group : https://groups.google.com/forum/#!forum/siim-kaggle
    #And do not forget to allow the Google Healthcare API
    #If you get any difficulties refer to the official tutorial : https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/siim-cloud-healthcare-api-tutorial
    #Be sure to run the command 'gcloud auth application-default login' in your terminal as well
    #####################
    """
    
    print('##########################')
    print('- GETTING DATA FROM GCP... -')
    print('##########################')
    #get_data_from_gcp() <<<<<<<<<<<<<<<<<<<<<<<<<<<<- UNCOMMENT ME TO GET DATA FROM GCP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print('##########################')
    print('- MOVING DATA... -')
    print('##########################')
    #move_data() <<<<<<<<<<<<<<<<<<<<<<<<<<<<- UNCOMMENT ME TO MOVE DATA TO CLEAN FOLDER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print('##########################')
    print('- CONVERT DICOM TO JPEG... -')
    print('##########################')
    pool = Pool(5)
    for type in ['train', 'test']:
        inputdir = f'..\\input\\{type}\\'
        outputdir = f'..\\input\\{type}\\images\\'
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        files = [f for f in glob.glob(inputdir + f'*.dcm', recursive=True)]
        func =partial(convert,inputdir,outputdir)
        for _ in tqdm(pool.imap_unordered(func,files),total = len(files)):
            pass
    pool.close()
    pool.join()
    
    print('##########################')
    print('- SUCESS -')
    print('##########################')

