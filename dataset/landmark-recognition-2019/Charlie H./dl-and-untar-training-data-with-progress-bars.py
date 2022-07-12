#THIS WILL DL AND UNTAR ALL FILES
#SET THE SAVE_DIR TO WHERE YOU WANT THE DL TO HAPPEN, 'train' BY DEFAULT
#SET NUM_TAR_TO_DL (DEFAULTED TO 1 FOR COMMITTING) TO 500 IF YOU WANT ALL TAR FILES
#SET UNTAR (DEFAULTED TO FALSE FOR COMMITTING) TO TRUE TO UNTAR ALL FILES

from tqdm.auto import tqdm
import requests
import errno
import tarfile
import os

def download(url, save_dir):
    filename = url.rsplit('/', 1)[1]

    if not os.path.exists(os.path.dirname(save_dir)):
        try:
            os.makedirs(save_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(f'{save_dir}/{filename}', 'wb+') as f:
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length'))

        if total is None:
            f.write(response.content)
        else:
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in tqdm(response.iter_content(chunk_size=1024)):
                    f.write(data)
                    pbar.update(1024)


TRAIN_CSV = 'https://s3.amazonaws.com/google-landmark/metadata/train.csv'
TRAIN_ATTRIBUTION_CSV = 'https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv'
TAR_URLS = [f'https://s3.amazonaws.com/google-landmark/train/images_{ ("00" + str(n))[-3:] }.tar' for n in range(0,500)]

SAVE_DIR = 'train'
NUM_TARS_TO_DL = 1
UNTAR = False

print('DOWNLOADING train.csv')
download(TRAIN_CSV, SAVE_DIR)

print('\nDOWNLOADING train_attribution.csv')
download(TRAIN_ATTRIBUTION_CSV, SAVE_DIR)

print('\nDOWNLOADING image tar files')
for url in tqdm(TAR_URLS[:NUM_TARS_TO_DL]):
    download(url, SAVE_DIR)

if UNTAR:
    print('\n UNTARRING image tar files')
    for filename in tqdm(os.listdir(SAVE_DIR)):
        if filename.endswith('.tar'):
            img_tar = tarfile.open(f'{SAVE_DIR}/{filename}')
            img_tar.extractall(path=SAVE_DIR)
