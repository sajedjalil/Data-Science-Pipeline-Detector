##############################################################################################
# Modify original Cassava Leaf Disease Classification dataset 
# to generate the specified class directories (ex. mv example.jpg 0-CBB)
# This is done to input it into the tf.keras.preprocessing.image_dataset_from_directory() func
# Note: I ran this on my local machine NOT in a kaggle notebook
# You can find the outputted dataset at
# https://www.kaggle.com/raunakingcoder/cavleadclassificationmodifieddataset
##############################################################################################

# import libraries
import os 
import pandas as pd

train = pd.read_csv("train.csv") # read train.csv

# mkdirs for each class
os.system("mkdir train_images/0") # Cassava Bacterial Blight
os.system("mkdir train_images/1") # Cassava Brown Streak Disease 
os.system("mkdir train_images/2") # Cassava Green Mottle
os.system("mkdir train_images/3") # Cassava Mosaic Disease 
os.system("mkdir train_images/4") # Healthy

# move images into specified class folder
for i in range(len(train.index)):
    if train["label"][i]==0:
        os.system("mv train_images/"+train['image_id'][i]+" train_images/0/")
    elif train["label"][i]==1:
        os.system("mv train_images/"+train['image_id'][i]+" train_images/1/")
    elif train["label"][i]==2:
        os.system("mv train_images/"+train['image_id'][i]+" train_images/2/")
    elif train["label"][i]==3:
        os.system("mv train_images/"+train['image_id'][i]+" train_images/3/")
    elif train["label"][i]==4:
        os.system("mv train_images/"+train['image_id'][i]+" train_images/4/")