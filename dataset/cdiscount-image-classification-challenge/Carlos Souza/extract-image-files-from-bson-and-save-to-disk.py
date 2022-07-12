import os
import io
import bson
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp
from glob import iglob


bson_file = 'train_example.bson'
NCORE = 16
max_images = 7069896

input_dir = os.path.abspath(os.path.join(os.getcwd(), '../input'))
base_dir = os.path.join(os.getcwd())
images_dir = os.path.join(base_dir, 'images')
bson_file = os.path.join(input_dir, bson_file)

product_count = 0
category_count = 0
picture_count = 0


def process(q, iolock):
    global product_count
    global category_count
    global picture_count
    while True:
        d = q.get()
        if d is None:
            break

        product_count += 1
        product_id = str(d['_id'])
        category_id = str(d['category_id'])

        category_dir = os.path.join(images_dir, category_id)
        if not os.path.exists(category_dir):
            category_count += 1
            try:
                os.makedirs(category_dir)
            except:
                pass

        for e, pic in enumerate(d['imgs']):
            picture_count += 1
            picture = imread(io.BytesIO(pic['picture']))
            picture_file = os.path.join(category_dir, product_id + '_' + str(e) + '.jpg')
            if not os.path.isfile(picture_file):
                plt.imsave(picture_file, picture)


q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))


data = bson.decode_file_iter(open(bson_file, 'rb'))

for c, d in tqdm(enumerate(data)):
    if (c + 1) > max_images:
        break
    q.put(d)  # blocks until q below its max size

# tell workers we're done
for _ in range(NCORE):
    q.put(None)
pool.close()
pool.join()

print('Images saved at %s' % images_dir)
print('Products: \t%d\nCategories: \t%d\nPictures: \t%d' % (product_count, category_count, picture_count))

file = open(os.path.join(base_dir, 'retrained_labels.txt'), 'w')

rootdir_glob = images_dir + '/**/*'
folder_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isdir(f)]
for folder in folder_list:
    category = folder.split('/')[-1]
    file.write(category + '\n')

file.close()

print('"retrained_labels.txt" saved at %s' % base_dir)
