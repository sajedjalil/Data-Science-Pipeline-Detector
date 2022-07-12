########################################################################################################################
##
## Copyright (C) 2018 Amro Tork <amtc2018@gmail.com>
##
########################################################################################################################

## NOTE: This script doesn't run on Kaggle.


########################################################################################################################
## Imports
########################################################################################################################
import os
import requests
import concurrent.futures
import tqdm
import logging
import sys
import coloredlogs
import json
import pandas as pd

from keras.preprocessing import image

try:
    from urlparse import urlsplit
    from urllib import unquote
except ImportError: # Python 3
    from urllib.parse import urlsplit, unquote


########################################################################################################################
## procedures
########################################################################################################################
def delete_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
        
def download_image(img_id, tag, url, download_path, chunk_size=1024):
    filename = os.path.join(download_path,str(img_id) + "_" + tag + ".jpg")
    
    if not os.path.isfile(filename):
        with open(filename, 'wb') as handle:
            try:
                response = requests.get(url, stream=True, timeout=20.0)
            except:
                logging.warning("## Was not able to connect to: {} for image {:d}".format(url,img_id))
                delete_file(filename)
                return img_id, filename, False, 0, 0
                
            if not response.ok:
                logging.warning("## Was not able to get the image: {} for image {:d}".format(response, img_id))
                delete_file(filename)
                return img_id, filename, False, 0, 0

            for block in response.iter_content(chunk_size):
                if not block:
                    break

                handle.write(block)
    
    try:
        image_arr = image.load_img(filename)
    except:
        logging.warning('Failed to read image %d from %s' % (img_id,url))
        delete_file(filename)
        return img_id, filename, False, 0, 0
    
    return img_id, filename, True, image_arr.width, image_arr.height

def download_all(images_data, download_path, tag, num_concurrent):
    pbar = tqdm.tqdm(total=len(images_data))
    results = list()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_concurrent) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = [executor.submit(download_image, int(d["imageId"]), tag, d["url"], download_path) for d in images_data]
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                result = future.result()
                results.append(result)
                
            except Exception as exc:
                logging.error('Exception: {}'.format(exc))
            
            pbar.update()
    
    pbar.close()
    return results

    
def run():
    if len(sys.argv) != 4:
        print('Syntax: %s <train|validation|test.json> <output_dir> <tag>' % sys.argv[0])
        sys.exit(0)
  
    (data_file, out_dir, tag) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(data_file) as datafile:
        data = json.load(datafile)

    all_images_info = download_all(data["images"], out_dir, tag, num_concurrent=20)
    
    image_df = pd.DataFrame(columns=["img_id","filename","Good","width","height"]).from_records(all_images_info)
    image_df.columns = ["img_id","filename","Good","width","height"]
    image_df.sort_values(by="img_id")
    image_df.to_csv(os.path.join(out_dir,"meta_data.csv"), index=False)
    
if __name__ == '__main__':
    run()
    
    
    