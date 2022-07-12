'''
Here's a simple script that reads the images located at the URLs contained in the "photos" attribute of the training and testing datasets.
It simply reads te image and computes simple figures, like the size, the min/max/mean values of the pixels, etc.
The results are stored in two csv files (one for the training set, one for the test set).

In case some error occur (e.g. glitches in the network connexion), the get_photo_features function will keep track of it in its output
err_list. Another run of the function applied only to the errored listings can be done to make sure every listing is processed.
'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import requests
from io import StringIO


def get_photo_features(df):
    nrows = df.shape[0]
    features = {}
    features['size'] = np.zeros((nrows,))
    features['mini'] = np.zeros((nrows,))
    features['maxi'] = np.zeros((nrows,))
    features['mean'] = np.zeros((nrows,))
    features['std'] = np.zeros((nrows,))
    features['median'] = np.zeros((nrows,))
    features['percent25'] = np.zeros((nrows,))
    features['percent75'] = np.zeros((nrows,))
    
    err_list = np.zeros((nrows,))
    err_list[:] = -1

    #for n in range(df.shape[0]):
    for n in range(10):
        if n % 1000 == 0:
            # saving every 1000 listings
            df_features = pd.DataFrame(features)
            df_features.to_csv("df_photo_features" + str(n) + ".csv", index=False)
        if n % 1 == 0:
            print ('processing photos ... ' + str(n) + ' / ' + str(df.shape[0]))
        # list of photo URLs
        photos = df['photos'].values[n]
        nphotos = len(photos)
        # if there are no photos, all values are set to 0
        if nphotos == 0:
            for k in features.keys():
                features[k][n] = 0
            continue
        
        size = np.zeros((nphotos,))
        mini = np.zeros((nphotos,))
        maxi = np.zeros((nphotos,))
        mean = np.zeros((nphotos,))
        std = np.zeros((nphotos,))
        median = np.zeros((nphotos,))
        percent25 = np.zeros((nphotos,))
        percent75 = np.zeros((nphotos,))        
        for p in range(nphotos):
            url = photos[p]
            # get image from URL
            #try:
            print(url)
            response = requests.get(url)
            #except ConnectionError:
            #    image = None
            #    print ("error image " + str(p) + ", n = " + str(n))
            #    err_list[n] = p
            try:
                image = Image.open(StringIO(response.content))    
            except IOError:
                image = None
                print ("error image " + str(p) + ", n = " + str(n))
                err_list[n] = p
            
            if image == None:
                # no image --> we use the features coming from the last valid image (so that we don't mess too much with the statistics...)
                # size
                size[p] = len(img)
                # min value
                mini[p] = min(img)
                # max value
                maxi[p] = max(img)
                # mean
                mean[p] = np.mean(img)
                # standard deviation
                std[p] = np.std(img)
                # median 
                median[p] = np.median(img)
                # 1st quartile
                percent25[p] = np.percentile(img, q=25)
                # 3rd quartile
                percent75[p] = np.percentile(img, q=75)
                
            else:
                # convert to grayscale
                image = image.convert(mode = "L")
                # convert to numpy array
                img = np.array(image.getdata())
                
                # size
                size[p] = len(img)
                # min value
                mini[p] = min(img)
                # max value
                maxi[p] = max(img)
                # mean
                mean[p] = np.mean(img)
                # standard deviation
                std[p] = np.std(img)
                # median 
                median[p] = np.median(img)
                # 1st quartile
                percent25[p] = np.percentile(img, q=25)
                # 3rd quartile
                percent75[p] = np.percentile(img, q=75)
                
                #plt.imshow(img.reshape(np.flipud(image.size)))
        
        features['size'][n] = np.mean(size)
        features['mini'][n] = np.mean(mini)
        features['maxi'][n] = np.mean(maxi)
        features['mean'][n] = np.mean(mean)
        features['std'][n] = np.mean(std)
        features['median'][n] = np.mean(median)
        features['percent25'][n] = np.mean(percent25)
        features['percent75'][n] = np.mean(percent75)
    
    return features, err_list



if __name__ == '__main__':
    data_path = '../input/'
    train_df = pd.read_json(open(data_path + "train.json", "r"))
    train_ft = get_photo_features(train_df)
    train_features = pd.DataFrame(train_ft)
    train_features.to_csv("train_photo_features.csv", index=False)

    test_df = pd.read_json(open(data_path + "test.json", "r"))
    test_ft = get_photo_features(train_df)
    test_features = pd.DataFrame(test_ft)
    test_features.to_csv("test_photo_feature.csv", index=False)

