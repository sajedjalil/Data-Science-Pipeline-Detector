import pickle
from io import BytesIO
import logging

import ujson as json
import pandas as pd
import numpy as np
from scipy.spatial import distance

from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from PIL import Image, ImageStat, ImageOps

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg19 import VGG19, decode_predictions

import pytesseract

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

ses = Session()
ses.mount('https://', HTTPAdapter(max_retries=Retry(total=5)))

resnet = ResNet50(include_top=False, input_shape=(210, 266, 3))
vgg = VGG19()


def get_data(train=True):
    if train:
        with open('train.json', 'r') as raw_data:
            data = json.load(raw_data)
    else:
        with open('test.json', 'r') as raw_data:
            data = json.load(raw_data)

    df = pd.DataFrame(data)
    return df['photos']


def _parse(pic):
    pic = ses.get(pic)
    img = Image.open(BytesIO(pic.content))
    t = pytesseract.image_to_string(img)
    has_text = 1 if len(t) else 0

    img = ImageOps.fit(img, (266, 210), Image.ANTIALIAS)
    img_for_clf = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    img_for_clf = preprocess_input(np.array(img_for_clf.convert("RGB")).reshape(1, 224, 224, 3).astype(np.float64))

    preds = decode_predictions(vgg.predict(img_for_clf), top=3)
    objects = [x[1] for x in preds[0]]
    logger.info('There are {} on the picture'.format(objects))

    stats = ImageStat.Stat(img, mask=None)
    if len(stats.mean) == 3:
        r_mean, g_mean, b_mean = stats.mean
        r_var, g_var, b_var = stats.var
    else:
        # grayscale image happened
        r_mean, g_mean, b_mean = stats.mean[0], stats.mean[0], stats.mean[0]
        r_var, g_var, b_var = stats.var[0], stats.var[0], stats.var[0]

    img = preprocess_input(np.array(img.convert("RGB")).reshape(1, 210, 266, 3).astype(np.float64))
    features = resnet.predict(img)
    return (r_mean, g_mean, b_mean, r_var, g_var, b_var, has_text), features.reshape(2048), objects


default_values = (255, 255, 255, 0, 0, 0, 0), np.zeros(2048), []


def parse(pic):
    try:
        return _parse(pic)
    except KeyboardInterrupt:
        raise SystemExit()
    except:
        logger.exception('Parsing failed: {}')
        return default_values


def process_photo(photo):
    # two pics are used because preview at website contains two first photos
    if not len(photo):
        m1, e1, objects1 = default_values
        m2, e2, objects2 = default_values
        dist = 0
        objects_set = {}
    elif len(photo) == 1:
        m1, e1, objects1 = parse(photo[0])
        m2, e2, objects2 = default_values
        dist = distance.euclidean(e1, e2)
        objects_set = set(objects1)
    else:
        m1, e1, objects1 = parse(photo[0])
        m2, e2, objects2 = parse(photo[1])
        dist = distance.euclidean(e1, e2)
        objects_set = set(objects1 + objects2)

    return [m1, m2], [e1, e2], dist, objects_set


def main(train=True):
    photos = get_data(train=train)

    processed = map(process_photo, photos)
    manual, extracted, distances, objects = zip(*processed)

    fname = 'pics_train.bin' if train else 'pics_test.bin'
    with open(fname, 'wb') as out:
        pickle.dump((manual, extracted, distances, objects), out)


if __name__ == '__main__':
    main()
    main(False)
