# Purpose: download images of iMaterial-Fashion dataset

# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

######################################################################################################################
## Imports
######################################################################################################################



import sys, os, multiprocessing, urllib3, csv
from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

######################################################################################################################
## Functions
######################################################################################################################

client = urllib3.PoolManager(500)

def ParseData(data_file):

  j = json.load(open(data_file))

  annotations = {}

  if 'train' in data_file or 'validation' in data_file:
      _annotations = j['annotations']
      for annotation in _annotations:
        annotations[annotation['imageId']] = [int(i) for i in annotation['labelId']]

  key_url_list = []
  images = j['images']

  for item in images:
    url = item['url']
    id_ = item['imageId']

    if id_ in annotations:
        id_ = "id_{}_labels_{}".format(id_, annotations[id_])
    key_url_list.append((id_, url))

  return key_url_list




def DownloadImage(key_url):

  out_dir = sys.argv[2]
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    global client
    response = client.request('GET', url)#, timeout=30)
    image_data = response.data
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s %s' % (key,url))
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return


def Run():

  if len(sys.argv) != 3:
    print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=12)

  with tqdm(total=len(key_url_list)) as bar:
    for _ in pool.imap_unordered(DownloadImage, key_url_list):
      bar.update(1)


if __name__ == '__main__':
  Run()