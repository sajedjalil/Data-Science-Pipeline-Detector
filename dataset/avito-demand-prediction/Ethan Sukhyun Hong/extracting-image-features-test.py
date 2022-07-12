#Thanks to Nooh, who gave an inspiration of im KP extraction : https://www.kaggle.com/c/avito-demand-prediction/discussion/59414#348151


import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt



image_path = "data/competition_files/test_jpg/"
def keyp(img):
    try:        
        img = image_path + str(img) + ".jpg"
        exfile = zipped.read(img)
        arr = np.frombuffer(exfile, np.uint8)

        imz = cv2.imdecode(arr, 1)
        fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
        kp = fast.detect(imz,None)
        kp =len(kp)
        return kp
    except:
        return 0

test = pd.read_csv("../input/test.csv")


images = test[["image"]].drop_duplicates().dropna()
zipped = ZipFile('../input/test_jpg.zip')

images["Image_kp_score"] = images["image"].apply(lambda x: keyp(x))

images.to_csv("Image_KP_SCORES_test.csv", index = False)