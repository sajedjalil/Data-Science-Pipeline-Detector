"""
The script globs dataset folders and joines channel images into RGB images.

The more CPU cores you have, the faster this script will be executed. Also, the
script doesn't depend on accessor functions provided by the competition host. 
It uses only the meta-information on your local disk.
"""
import glob
from multiprocessing import cpu_count
import os
from pprint import pprint as pp
import sys
import shutil

from imageio import imread
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm_notebook as tqdm


def donwload_rxrx_tools():
    if not os.path.exists('rxrx1-utils'):
        print('Cloning RxRx repository...')
        os.system('git clone https://github.com/recursionpharma/rxrx1-utils')
    print('Adding to the search path.')
    sys.path.append('rxrx1-utils')
    print('RxRx tools are ready!')
donwload_rxrx_tools()
import rxrx.io as rio


def collect_records(basedir):
    """Globs the folder with images and constructs data frame with image paths
    and additional meta-information.
    """
    records = []
    columns = ['experiment', 'plate', 'well', 'site', 'channel', 'filename']
    for path in glob.glob(f'{basedir}/**/*.png', recursive=True):
        exp, plate, filename = os.path.relpath(path, start=basedir).split('/')
        basename, _ = os.path.splitext(filename)
        well, site, channel = basename.split('_')
        records.append([exp, int(plate[-1]), well, int(site[1:]), int(channel[1:]), path])
    records = pd.DataFrame(records, columns=columns)
    return records


def parallel_join_channels(image_groups, output_dir):
    """Reads separate channels from disk and merges them into single RGB sample.
    
    Each generated file name includes information about experiment and label so
    you can easily restore it from file path string.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def worker(channel_group, output_dir):
        x = np.zeros((512, 512, 6), dtype=np.uint8)
        for info in channel_group:
            xc = np.asarray(imread(info['filename']))
            x[:, :, info['channel']-1] = xc
        
        sirna = info['sirna']
        y = 0 if pd.isna(sirna) else int(sirna)
        output_file = f"{info['id_code']}_s{info['site']}_{y}.png"
        output_path = os.path.join(output_dir, output_file)
        rgb = rio.convert_tensor_to_rgb(x)
        img = PIL.Image.fromarray(rgb.astype(np.uint8))
        img.save(output_path)
        return output_path
    
    with Parallel(n_jobs=cpu_count()) as p:
        paths = p(delayed(worker)(g, output_dir) for g in tqdm(image_groups))
        
    return paths
    
    
def merge_channels(meta, subset):
    """Go through the train/test subset of data and join standalone channels into RGB images."""
    
    print('Taking meta-information for subset:', subset)
    meta = meta[meta.dataset == subset]
    dirpath = f'../input/recursion-cellular-image-classification/{subset}'
    print('Glob folders with images...')
    records = collect_records(dirpath)
    print('Joining meta information with image paths...')
    info = pd.merge(records, meta, on=['experiment', 'plate', 'well', 'site'])
    print('Joining channels...')
    groups = [group.to_dict('records') for _, group in info.groupby(['id_code', 'site'])]
    output_path = f'{subset}_rgb'
    paths = parallel_join_channels(groups, output_path)
    print(f'Done! The data saved into {output_path}')


# ----------------------
# Converting the dataset
# ----------------------


root = '../input/recursion-cellular-image-classification/'
meta_trn = pd.read_csv(f'{root}/train.csv')
meta_trn_ctrl = pd.read_csv(f'{root}/train_controls.csv')
meta_tst = pd.read_csv(f'{root}/test.csv')
meta_tst_ctrl = pd.read_csv(f'{root}/test_controls.csv')

meta_trn['well_type'] = 'treatment'
meta_trn['dataset'] = 'train'
meta_trn_ctrl['dataset'] = 'train'

meta_tst['well_type'] = 'treatment'
meta_tst['dataset'] = 'test'
meta_tst_ctrl['dataset'] = 'test'

meta = pd.concat([meta_trn, meta_trn_ctrl, meta_tst, meta_tst_ctrl], sort=False, axis='rows')
meta_site1 = meta.copy()
meta_site1['site'] = 1
meta_site2 = meta.copy()
meta_site2['site'] = 2
meta = pd.concat([meta_site1, meta_site2], axis='rows').reset_index(drop=True)

# commented-out because kernel cannot fit that many images; uncomment on your machine
# merge_channels(meta, 'train')
# merge_channels(meta, 'test')

shutil.rmtree('rxrx1-utils')