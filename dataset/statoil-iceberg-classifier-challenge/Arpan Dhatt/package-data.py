import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
print("Dependencies Loaded")

df_train = pd.read_json('../input/statoil-iceberg-classifier-challenge/train.json') # this is a dataframe
print("Training Data Read")

def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)
    
Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)


Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]
print("Preprocessed Training Set")

def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
    
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
    
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
    
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

X_validation = Xtrain[-100:,...]
Y_validation = Ytrain[-100:]
Xtrain = Xtrain[0:-100,...]
Ytrain = Ytrain[0:-100]
print("Validation Set of Size "+str(Y_validation.shape[0])+" Separated")
    
Xtrain = get_more_images(Xtrain)
Ytrain = np.concatenate((Ytrain,Ytrain,Ytrain))
print("Data Augmentation Complete")

print(Xtrain.shape,Ytrain.shape)
print ("Completed Data set concatenated and shuffled\nNow slices have size of "+str((Xtrain.nbytes+Ytrain.nbytes+X_validation.nbytes+Y_validation.nbytes)/1000000000)+" gigabytes")
print("Shuffled Complete Dataset\nSaving data to .npz file")
np.savez_compressed('input_data',X_train=Xtrain[:,...],
                                 Y_train=Ytrain[:],
                                 X_validation=X_validation[:,...],
                                 Y_validation=Y_validation[:])
print("Success! Data Saved!")