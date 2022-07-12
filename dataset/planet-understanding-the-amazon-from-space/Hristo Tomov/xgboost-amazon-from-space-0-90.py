
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import cv2
import scipy
from osgeo import gdal
from numpy import mean, sqrt, square, arange


def extract_features(df, data_path):
    im_features = df.copy()

    NDVI_mean = []
    NDWI_mean = []
    NDHI_mean = []

    NDVI_std = []
    NDWI_std = []
    NDHI_std = []

    NDVI_max = []
    NDWI_max = []
    NDHI_max = []
        
    NDVI_sum = []
    NDWI_sum = []
    NDHI_sum = []

    NDVI_min = []
    NDWI_min = []
    NDHI_min = []    
    
    NDVI_kurtosis = []
    NDWI_kurtosis = []
    NDHI_kurtosis = []
    
    NDVI_skewness = []
    NDWI_skewness = []
    NDHI_skewness = []
    
    NDVI_rms = []
    NDWI_rms = []
    NDHI_rms = []
    
    NDVI_var = []
    NDWI_var = []
    NDHI_var = []
    
    NDVI_Sobel = []
    NDWI_Sobel = []
    NDHI_Sobel = []
    
    NDVI_Laplacian = []
    NDWI_Laplacian = []
    NDHI_Laplacian = []
    
    R_mean = []
    G_mean = []
    B_mean = []
    NIR_mean = []

    R_std = []
    G_std = []
    B_std = []
    NIR_std = []

    R_max = []
    G_max = []
    B_max = []
    NIR_max = []
    
    R_sum = []
    G_sum = []
    B_sum = []
    NIR_sum = []
    
    R_min = []
    G_min = []
    B_min = []    
    NIR_min = []
    
    R_kurtosis = []
    G_kurtosis = []
    B_kurtosis = []
    NIR_kurtosis = []
    
    R_skewness = []
    G_skewness = []
    B_skewness = []
    NIR_skewness = []
    
    R_rms = []
    G_rms = []
    B_rms = []
    NIR_rms = []
    
    R_var = []
    G_var = []
    B_var = []
    NIR_var = []
    
    R_Sobel = []
    G_Sobel = []
    B_Sobel = []
    NIR_Sobel = []
    
    R_Laplacian = []
    G_Laplacian = []
    B_Laplacian = []
    NIR_Laplacian = []
    
    r_hist_tiff = None
    g_hist_tiff = None
    b_hist_tiff = None
    nir_hist_tiff = None
    
    NDVI_hist = None
    NDWI_hist = None
    NDHI_hist = None
    
    for image_name in tqdm(im_features.image_name.values, miniters=100): 
                
        img = gdal.Open(data_path + image_name + '.tif')
        imgData = img.ReadAsArray()
            
        r = imgData[2,:,:]
        g = imgData[1,:,:]
        b = imgData[0,:,:]
        nir = imgData[3,:,:]

        R = (r / 65535.)**(1/2.2) 
        G = (g / 65535.)**(1/2.2) 
        B = (b / 65535.)**(1/2.2) 
        NIR = (nir / 65535.)**(1/2.2) 

        NDVI = (NIR-R)/(NIR+R)
        NDWI = (G-NIR)/(G+NIR)
        NDHI = (B-NIR)/(B+NIR)  
        
        NDVI_mean.append(np.mean(NDVI.ravel()))
        NDWI_mean.append(np.mean(NDWI.ravel()))
        NDHI_mean.append(np.mean(NDHI.ravel()))

        NDVI_std.append(np.std(NDVI.ravel()))
        NDWI_std.append(np.std(NDWI.ravel()))
        NDHI_std.append(np.std(NDHI.ravel()))

        NDVI_max.append(np.max(NDVI.ravel()))
        NDWI_max.append(np.max(NDWI.ravel()))
        NDHI_max.append(np.max(NDHI.ravel()))

        NDVI_sum.append(np.sum(NDVI.ravel()))
        NDWI_sum.append(np.sum(NDWI.ravel()))
        NDHI_sum.append(np.sum(NDHI.ravel()))

        NDVI_min.append(np.min(NDVI.ravel()))
        NDWI_min.append(np.min(NDWI.ravel()))
        NDHI_min.append(np.min(NDHI.ravel()))

        NDVI_kurtosis.append(scipy.stats.kurtosis(NDVI.ravel()))
        NDWI_kurtosis.append(scipy.stats.kurtosis(NDWI.ravel()))
        NDHI_kurtosis.append(scipy.stats.kurtosis(NDHI.ravel()))

        NDVI_skewness.append(scipy.stats.skew(NDVI.ravel()))
        NDWI_skewness.append(scipy.stats.skew(NDWI.ravel()))
        NDHI_skewness.append(scipy.stats.skew(NDHI.ravel()))
        
        NDVI_rms.append(sqrt(mean(square(NDVI.ravel()))))
        NDWI_rms.append(sqrt(mean(square(NDWI.ravel()))))
        NDHI_rms.append(sqrt(mean(square(NDHI.ravel()))))
        
        NDVI_var.append(np.var(NDVI.ravel()))
        NDWI_var.append(np.var(NDWI.ravel()))
        NDHI_var.append(np.var(NDHI.ravel()))
        
        NDVI_Sobel.append(cv2.Sobel(NDVI,cv2.CV_64F,1,0,ksize=5).var())
        NDWI_Sobel.append(cv2.Sobel(NDWI,cv2.CV_64F,1,0,ksize=5).var())
        NDHI_Sobel.append(cv2.Sobel(NDHI,cv2.CV_64F,1,0,ksize=5).var())
        
        NDVI_Laplacian.append(cv2.Laplacian(NDVI, cv2.CV_64F).var())
        NDWI_Laplacian.append(cv2.Laplacian(NDWI, cv2.CV_64F).var())
        NDHI_Laplacian.append(cv2.Laplacian(NDHI, cv2.CV_64F).var())
        
        R_mean.append(np.mean(R.ravel()))
        G_mean.append(np.mean(G.ravel()))
        B_mean.append(np.mean(B.ravel()))
        NIR_mean.append(np.mean(NIR.ravel()))
        
        R_std.append(np.std(R.ravel()))
        G_std.append(np.std(G.ravel()))
        B_std.append(np.std(B.ravel()))
        NIR_std.append(np.std(NIR.ravel()))
        
        R_max.append(np.max(R.ravel()))
        G_max.append(np.max(G.ravel()))
        B_max.append(np.max(B.ravel()))
        NIR_max.append(np.max(B.ravel()))

        
        R_sum.append(np.sum(R.ravel()))
        G_sum.append(np.sum(G.ravel()))
        B_sum.append(np.sum(B.ravel()))
        NIR_sum.append(np.sum(NIR.ravel()))
        
        R_min.append(np.min(R.ravel()))
        G_min.append(np.min(G.ravel()))
        B_min.append(np.min(B.ravel()))
        NIR_min.append(np.min(NIR.ravel()))
        
        R_kurtosis.append(scipy.stats.kurtosis(R.ravel()))
        G_kurtosis.append(scipy.stats.kurtosis(G.ravel()))
        B_kurtosis.append(scipy.stats.kurtosis(B.ravel()))
        NIR_kurtosis.append(scipy.stats.kurtosis(NIR.ravel()))
        
        R_skewness.append(scipy.stats.skew(R.ravel()))
        G_skewness.append(scipy.stats.skew(G.ravel()))
        B_skewness.append(scipy.stats.skew(B.ravel()))
        NIR_skewness.append(scipy.stats.skew(NIR.ravel()))
        
        R_rms.append(sqrt(mean(square(R.ravel()))))
        G_rms.append(sqrt(mean(square(G.ravel()))))
        B_rms.append(sqrt(mean(square(B.ravel()))))
        NIR_rms.append(sqrt(mean(square(NIR.ravel()))))
        
        R_var.append(np.var(R.ravel()))
        G_var.append(np.var(G.ravel()))
        B_var.append(np.var(B.ravel()))
        NIR_var.append(np.var(NIR.ravel()))

        R_Sobel.append(cv2.Sobel(R,cv2.CV_64F,1,0,ksize=5).var())
        G_Sobel.append(cv2.Sobel(G,cv2.CV_64F,1,0,ksize=5).var())
        B_Sobel.append(cv2.Sobel(B,cv2.CV_64F,1,0,ksize=5).var())
        NIR_Sobel.append(cv2.Sobel(NIR,cv2.CV_64F,1,0,ksize=5).var())
        
        R_Laplacian.append(cv2.Laplacian(R, cv2.CV_64F).var())
        G_Laplacian.append(cv2.Laplacian(G, cv2.CV_64F).var())
        B_Laplacian.append(cv2.Laplacian(B, cv2.CV_64F).var())
        NIR_Laplacian.append(cv2.Laplacian(NIR, cv2.CV_64F).var())
               
        ## Histograms TIFF
        rh_tiff, _ = np.histogram(R, bins = 20, range=(0,1))
        gh_tiff, _ = np.histogram(G, bins = 20, range=(0,1))   
        bh_tiff, _ = np.histogram(B, bins = 20, range=(0,1)) 
        nirh_tiff, _ = np.histogram(NIR, bins = 20, range=(0,1)) 
        
        if r_hist_tiff is None:
            r_hist_tiff = rh_tiff
            g_hist_tiff = gh_tiff
            b_hist_tiff = bh_tiff
            nir_hist_tiff = nirh_tiff
        else:
            r_hist_tiff = np.vstack([r_hist_tiff, rh_tiff])
            g_hist_tiff = np.vstack([g_hist_tiff, gh_tiff])
            b_hist_tiff = np.vstack([b_hist_tiff, bh_tiff])
            nir_hist_tiff = np.vstack([nir_hist_tiff, nirh_tiff])
                                 
                                 
        ## Histograms NDVI, NDHI, NDWI
        h_NDVI, _ = np.histogram(NDVI, bins = 5, range=(0,1))
        h_NDHI, _ = np.histogram(NDHI, bins = 5, range=(0,1))   
        h_NDWI, _ = np.histogram(NDWI, bins = 5, range=(0,1)) 

        if NDVI_hist is None:
            NDVI_hist = h_NDVI
            NDHI_hist = h_NDHI
            NDWI_hist = h_NDWI
        else:
            NDVI_hist = np.vstack([NDVI_hist, h_NDVI])
            NDHI_hist = np.vstack([NDHI_hist, h_NDHI])
            NDWI_hist = np.vstack([NDWI_hist, h_NDWI])
                                 
                                  
                                  
    im_features['NDVI_mean'] = NDVI_mean
    im_features['NDWI_mean'] = NDWI_mean
    im_features['NDHI_mean'] = NDHI_mean

    im_features['NDVI_std'] = NDVI_std
    im_features['NDWI_std'] = NDWI_std
    im_features['NDHI_std'] = NDHI_std

    im_features['NDVI_max'] = NDVI_max
    im_features['NDWI_max'] = NDWI_max
    im_features['NDHI_max'] = NDHI_max

    im_features['NDVI_sum'] = NDVI_sum
    im_features['NDWI_sum'] = NDWI_sum
    im_features['NDHI_sum'] = NDHI_sum
    
    im_features['NDVI_min'] = NDVI_min
    im_features['NDWI_min'] = NDWI_min
    im_features['NDHI_min'] = NDHI_min

    im_features['NDVI_kurtosis'] = NDVI_kurtosis
    im_features['NDWI_kurtosis'] = NDWI_kurtosis
    im_features['NDHI_kurtosis'] = NDHI_kurtosis
    
    im_features['NDVI_skewness'] = NDVI_skewness
    im_features['NDWI_skewness'] = NDWI_skewness
    im_features['NDHI_skewness'] = NDHI_skewness
    
    im_features['NDVI_rms'] = NDVI_rms
    im_features['NDWI_rms'] = NDWI_rms
    im_features['NDHI_rms'] = NDHI_rms
    
    im_features['NDVI_var'] = NDVI_var
    im_features['NDWI_var'] = NDWI_var
    im_features['NDHI_var'] = NDHI_var
    
    
    im_features['NDVI_rms'] = NDVI_Sobel
    im_features['NDWI_rms'] = NDWI_Sobel
    im_features['NDHI_rms'] = NDHI_Sobel
    
    im_features['NDVI_Laplacian'] = NDVI_Laplacian
    im_features['NDWI_Laplacian'] = NDWI_Laplacian
    im_features['NDHI_Laplacian'] = NDHI_Laplacian

    im_features['R_mean'] = R_mean
    im_features['G_mean'] = G_mean
    im_features['B_mean'] = B_mean
    im_features['NIR_mean'] = NIR_mean
    
    im_features['R_std'] = R_std
    im_features['G_std'] = G_std
    im_features['B_std'] = B_std
    im_features['NIR_std'] = NIR_std
    
    im_features['R_max'] = R_max
    im_features['G_max'] = G_max
    im_features['B_max'] = B_max
    im_features['NIR_max'] = NIR_max
    
    im_features['R_sum'] = R_sum
    im_features['G_sum'] = G_sum
    im_features['B_sum'] = B_sum
    im_features['NIR_sum'] = NIR_sum
    
    im_features['R_min'] = R_min
    im_features['G_min'] = G_min
    im_features['B_min'] = B_min
    im_features['NIR_min'] = NIR_min
    
    im_features['R_kurtosis'] = R_kurtosis
    im_features['G_kurtosis'] = G_kurtosis
    im_features['B_kurtosis'] = B_kurtosis
    im_features['NIR_kurtosis'] = NIR_kurtosis
        
    im_features['R_skewness'] = R_skewness
    im_features['G_skewness'] = G_skewness
    im_features['B_skewness'] = B_skewness
    im_features['NIR_skewness'] = NIR_skewness
    
    im_features['R_rms'] = R_rms
    im_features['G_rms'] = G_rms
    im_features['B_rms'] = B_rms
    im_features['NIR_rms'] = NIR_rms
        
    im_features['R_var'] = R_var
    im_features['G_var'] = G_var
    im_features['B_var'] = B_var
    im_features['NIR_var'] = NIR_var
    
    
    im_features['R_Sobel'] = R_Sobel
    im_features['G_Sobel'] = G_Sobel
    im_features['B_Sobel'] = B_Sobel
    im_features['NIR_Sobel'] = NIR_Sobel
        
    im_features['R_var'] = R_Laplacian
    im_features['G_var'] = G_Laplacian
    im_features['B_var'] = B_Laplacian
    im_features['NIR_var'] = NIR_Laplacian
      
    # histograms
    for i in range(0, r_hist_tiff.shape[1]):
        im_features['r_hist_tiff_%d'%(i)] = r_hist_tiff[:,i]
    for i in range(0, g_hist_tiff.shape[1]):
        im_features['g_hist_tiff_%d'%(i)] = g_hist_tiff[:,i]
    for i in range(0, b_hist_tiff.shape[1]):
        im_features['b_hist_tiff_%d'%(i)] = b_hist_tiff[:,i]
    for i in range(0, nir_hist_tiff.shape[1]):
        im_features['nir_hist_tiff_%d'%(i)] = nir_hist_tiff[:,i]
        
    for i in range(0, NDVI_hist.shape[1]):
        im_features['NDVI_hist_%d'%(i)] = NDVI_hist[:,i]
    for i in range(0, NDHI_hist.shape[1]):
        im_features['NDHI_hist_%d'%(i)] = NDHI_hist[:,i]
    for i in range(0, NDWI_hist.shape[1]):
        im_features['NDWI_hist_%d'%(i)] = NDWI_hist[:,i]                  

    return im_features
    
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

train_path = '..\\train-tif-v2\\'
test_path = '..\\test-tif-v2\\'
train = pd.read_csv('..\\train_v2.csv')
test = pd.read_csv('..\\XGBoost_submission.csv')

# Extract features
print('Extracting train features')
train_features = extract_features(train, train_path)
print('Extracting test features')
test_features = extract_features(test, test_path)

# Prepare data
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.uint8)

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))

n_classes = y.shape[1]
X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))
# Train and predict with one-vs-all strategy
y_pred = np.zeros((X_test.shape[0], n_classes))

print('Training and making predictions')
for class_i in tqdm(range(n_classes), miniters=1): 
#     print('Analysing class ' + str(class_i))
    model = xgb.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=200, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = [' '.join(labels[y_pred_row > 0.2]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = test_features.image_name.values
subm['tags'] = preds
subm.to_csv('..\\XGB_submission_v11.csv', index=False)