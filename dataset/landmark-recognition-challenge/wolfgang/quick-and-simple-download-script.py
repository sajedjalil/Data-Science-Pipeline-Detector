# This Python script can't be run within Kaggle but it allows you to download the training set into 
# a dedicated folder structure that allows simple training of a Keras model.

import requests
import time
import os
import os.path
import random
import hashlib
import pandas as pd
from requests.auth import HTTPBasicAuth


# download url and put it into the specified class folder
def download(row):
	print(row['url'])
	url = row['url']
	class_id = row['landmark_id']
	
	os.makedirs("./train/%s/" % class_id, exist_ok=True)
	fname = "./train/%s/%s.jpg" % (class_id, row['id'])
	if os.path.isfile(fname):
		return
	# otherwise download it 
	r = requests.get(url, timeout=5.0)
	if r.status_code == 200:
	    with open(fname, 'wb') as f:
	        f.write(r.content)
	
def main():
	start_time = time.time()
	df = pd.read_csv('../input/train.csv')
	df.apply (lambda row: download (row),axis=1)

if __name__ == '__main__':
    main()