### This is a simple script to optimize prototyping with this dataset.
### There actually not many training data for this competition but since most of the images are huge it will take long to resize it on the fly
### We dont need to save this as separate dataset (to save our kaggle dataset limit), just keep it as kernel output file.
### However since kernel output is limited (500 files) at most if I am not wrong hence we will zip it as tar.gz file

import PIL
from PIL import Image
import glob
import tarfile
from multiprocessing import Pool
import tqdm
import os

IMG_SIZE = 224

os.mkdir('train_resized_{}'.format(IMG_SIZE))

src_list = glob.glob('../input/train_images/*.png')

def resize_mp(src_file) :
    f_name = src_file.split('/')[-1]
    img_chk = Image.open(src_file)
    img_chk = img_chk.resize((IMG_SIZE,IMG_SIZE),resample = PIL.Image.LANCZOS)
    img_chk.save(os.path.join('train_resized_{}'.format(IMG_SIZE), f_name))
    
with Pool(4) as p :
    p.map(resize_mp , src_list)
    
tar_f = tarfile.open('train_resized_{}.tar.gz'.format(IMG_SIZE) , 'w:gz')

for f in tqdm.tqdm(glob.glob('train_resized_{}/*.png'.format(IMG_SIZE))) :
    tar_f.add(f)
    os.remove(f)
    
tar_f.close()

### To unpack the output file simply load it to your kernel and do this 
###
### import tarfile
### with tarfile.open('../input/resizing-script-224/train_resized_224.tar.gz') as tar_f :
###     tar_f.extract_all()
