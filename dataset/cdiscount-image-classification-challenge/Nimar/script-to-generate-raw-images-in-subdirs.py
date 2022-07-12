# Creates subdirectories with raw images to use for training with keras.preprocessing.image.ImageDataGenerator, for example:
# 1. mkdir input
# 2. download train.bson to input/
# 3. mkdir transform
# 4. create a file called generate_raw.py with the contents of this script
# 5. python3 generate_raw.py 10 input/train.bson transform/train transform/valid
# 6. Wait an hour !
# 7. ls transfrom/train/raw transform/valid/raw
import io
import os
import sys
import random
import bson
import skimage.data
import random
import scipy
import collections

def main():
  if len(sys.argv) != 5:
    print("Usage: generate_raw.py <valid-pct> <bson-file> <traindir> "
          "<validdir>", file=sys.stderr)
    sys.exit(1)
  
  validpct = float(sys.argv[1]) / 100
  trainfile = sys.argv[2]
  traindir = sys.argv[3]
  validdir = sys.argv[4]
  
  for dir_ in [traindir, validdir]:
    mkdir_if_needed(dir_)
    mkdir_if_needed(os.path.join(dir_, "raw"))
    
  data = bson.decode_file_iter(open(trainfile, 'rb'))

  train_cats, valid_cats = collections.Counter(), collections.Counter()
  
  for prod in data:
    product_id = prod['_id']
    category_id = prod['category_id']

    # decide if this product will go into the validation or test data
    if random.random() < validpct:
      outdir = validdir
      valid_cats[category_id] += 1
    else:
      outdir = traindir
      train_cats[category_id] += 1
    
    cat_dir = os.path.join(outdir, "raw", str(category_id))
    mkdir_if_needed(cat_dir)
    
    for picidx, pic in enumerate(prod['imgs']):
      filename = os.path.join(cat_dir, "{}.{}.jpg".format(product_id, picidx))
      with open(filename, 'wb') as outf:
        outf.write(pic['picture'])

  for name, cnt in [("training", train_cats), ("validation", valid_cats)]:
    print("{}: {} categories with {} products".format(name, len(cnt),
                                                     sum(cnt.values())))
        
def mkdir_if_needed(dir_):
  if not os.path.isdir(dir_):
    os.mkdir(dir_)

if __name__ == "__main__":
  main()
  
