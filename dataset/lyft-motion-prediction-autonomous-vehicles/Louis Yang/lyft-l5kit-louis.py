import os

## =====================================================================================
## This is a temporarly fix for the freezing and the cuda issues. You can add this
## utility script instead of kaggle_l5kit until Kaggle resolve these issues.
## 
## You will be able to train and submit your results, but not all the functionality of
## l5kit will work properly.

## More details here:
## https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/177125

## this script transports l5kit and dependencies
os.system('pip install --target=/kaggle/working pymap3d==2.1.0')
os.system('pip install --target=/kaggle/working protobuf==3.12.2')
os.system('pip install --target=/kaggle/working transforms3d')
os.system('pip install --target=/kaggle/working zarr')
os.system('pip install --target=/kaggle/working ptable')
os.system('pip install --target=/kaggle/working torch==1.7.0')
# os.system('pip install --no-dependencies --target=/kaggle/working l5kit')

