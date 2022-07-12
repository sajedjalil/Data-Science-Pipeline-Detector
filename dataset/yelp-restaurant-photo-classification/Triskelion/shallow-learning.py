"""
  Experimenting with ideas from
  "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" 
  By Coates, Lee, and Ng.
  
  If it works, it's all them. Bugs, errors and misinterpretation is all mine.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.misc import imread, imresize
from collections import defaultdict

################################################################################
# Collecting some data and calculating a "popularity" feature for restaurants
# Popularity = Number of photos per business in dataset / length dataset

print("""Collecting some essentials""")

photo_to_biz_train = {}
photo_to_biz_test = {}
popularity_train = defaultdict(int)
popularity_test = defaultdict(int)
popularity = {}

for e, line in enumerate(open("../input/train_photo_to_biz_ids.csv")):
  if e > 0:
    r = line.strip().split(",")
    photo_to_biz_train[r[0]] = r[1]
    popularity_train[r[1]] += 1
  
for e, line in enumerate(open("../input/test_photo_to_biz.csv")):
  if e > 0:
    r = line.strip().split(",")
    photo_to_biz_test[r[0]] = r[1]
    popularity_test[r[1]] += 1

for biz, count in popularity_train.items():
  popularity[biz] = count / float(len(photo_to_biz_train)) 

for biz, count in popularity_test.items():
  popularity[biz] = count / float(len(photo_to_biz_test))   

y = {}

for e, line in enumerate(open("../input/train.csv")):
  if e > 0:
    r = line.strip().split(",")
    y[r[0]] = r[1]
    
################################################################################
# Creating our codebook (image patch clusters) using batch processing

print("""Creating our codebook""")

kmeans = MiniBatchKMeans(n_clusters=50, random_state=1)

buffer = []
index = 1

for i, photo_id in enumerate(photo_to_biz_train):

  # Open, Resize (150x150), and greyscale the images
  img = imresize(imread("../input/train_photos/%s.jpg"%(photo_id), "L"), (150, 150))
  
  # Extract 30 random patches (20x20)
  data = extract_patches_2d(img, (20,20), max_patches=30, random_state=1)
  data = np.reshape(data, (len(data), -1))
  
  buffer.append(data)
  index += 1
  
  # Process batch every 100 images (3000 patches)
  if index % 100 == 0:
    # Normalizing (local brightness and contrast normalization)
    data = np.concatenate(buffer, axis=0).astype(np.float64)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    
    kmeans.partial_fit(data)
    buffer = [] # reset buffer
  
  if i > 3000:
    break
  if i % 100 == 0:
    print("%s/%s"%(i,3000))

################################################################################
# Creating train set


"""
print("Vectorizing Train Images")

errors = 0
with open("train.csv", "w") as outfile:
  outfile.write("photo_id,biz_id,t0,t1,t2,t3,t4,t5,t6,t7,t8,popularity,img_x,img_y,%s\n"
                %(",".join(["centroid_%s"%i for i in range(30)])))

  for i, photo_id in enumerate(photo_to_biz_train):
    if i > 10000:
      break
    if i % 1000 == 0:
      print("%s/%s (%s errors)"%(i,10000,errors))     
    # Open, Resize (150x150), and greyscale the images
    img_o = imread("../input/train_photos/%s.jpg"%(photo_id), "L")
    img = imresize(img_o, (150, 150))
  
    # Extract 30 random patches (20x20)
    data = extract_patches_2d(img, (20,20), max_patches=30, random_state=1)
    data = np.reshape(data, (len(data), -1)).astype(np.float64)
  
    # Normalization
    try:
      data -= np.mean(data, axis=0)
      data /= np.std(data, axis=0)
  
      business_id = photo_to_biz_train[photo_id]
  
      csv_row = [photo_id, business_id]
  
      # Labels
      for i in range(9):
        if str(i) in y[business_id]:
          csv_row.append(1)
        else:
          csv_row.append(0)
      
      # Add some hand-crafted features
      csv_row.append(popularity[business_id])
      csv_row.append(img_o.shape[0])
      csv_row.append(img_o.shape[1])
  
      # Add cluster predictions per patch
      csv_row += list(kmeans.predict(data))
  
      outfile.write("%s\n"%(",".join([str(f) for f in csv_row])))
    except:
      errors += 1
  
"""  
################################################################################
# Creating test set

print("""Vectorizing Test Images""")

errors = 0
with open("test.csv", "w") as outfile:
  outfile.write("photo_id,biz_id,t0,t1,t2,t3,t4,t5,t6,t7,t8,popularity,img_x,img_y,%s\n"
                %(",".join(["centroid_%s"%i for i in range(30)])))

  for i, photo_id in enumerate(photo_to_biz_test):
    if i > 30000:
      break
    if i % 1000 == 0:
      print("%s/%s (%s errors)"%(i,10000,errors))     
    # Open, Resize (150x150), and greyscale the images
    img_o = imread("../input/test_photos/%s.jpg"%(photo_id), "L")
    img = imresize(img_o, (150, 150))
  
    # Extract 30 random patches (20x20)
    data = extract_patches_2d(img, (20,20), max_patches=30, random_state=1)
    data = np.reshape(data, (len(data), -1)).astype(np.float64)
  
    # Normalization
    try:
      data -= np.mean(data, axis=0)
      data /= np.std(data, axis=0)
  
      business_id = photo_to_biz_test[photo_id]
  
      csv_row = [photo_id, business_id]
  
      # Labels
      for i in range(9):
        csv_row.append(1)
      
      # Add some hand-crafted features
      csv_row.append(popularity[business_id])
      csv_row.append(img_o.shape[0])
      csv_row.append(img_o.shape[1])
  
      # Add cluster predictions per patch
      csv_row += list(kmeans.predict(data))
  
      outfile.write("%s\n"%(",".join([str(f) for f in csv_row])))
    except:
      errors += 1