# Generate submission-format file which perfectly classifies training cases
#    as to whether opacities are present or not but
#    without giving any information amount opacity locations

import os
import csv
import numpy as np
import pandas as pd

predstring = '0.5 0 0 1024 1024'  # Opacity somewhere, won't say where
results = {}
with open(os.path.join('../input/stage_1_train_labels.csv'), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        pid = rows[0]
        target = rows[5]
        # if row contains pneumonia add file to dictionary
        # which contains a list of positive cases without segmentation
        if pid not in results:
            if target == '1':
                results[pid] = predstring
            else:
                results[pid] = ''

out = pd.DataFrame.from_dict(data=results, orient='index', columns=['PredictionString'])
out.index.name = 'patientId'
out.to_csv('perfect_classification_baseline.csv')