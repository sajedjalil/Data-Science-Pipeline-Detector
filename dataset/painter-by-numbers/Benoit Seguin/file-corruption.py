from PIL import Image
import glob
from concurrent.futures import ThreadPoolExecutor

filenames = glob.glob('/scratch/benoit/test/*.jpg')

def scan_img(f):
    try:
        img = Image.open(f)
        img.load()
        return img.size
    except Exception as e:
        return e

with ThreadPoolExecutor(max_workers=20) as e:
    file_status = e.map(scan_img, filenames)

file_status = list(file_status)
err_indexes = [i for (i, s) in enumerate(file_status) if not isinstance(s, tuple)]
for i in err_indexes:
    print(filenames[i].split('/')[-1])
    print(file_status[i])

# OUTPUT TEST FOLDER: 
# 20153.jpg
# image file is truncated (0 bytes not processed)
# 100532.jpg
# image file is truncated (57 bytes not processed)
# 18649.jpg
# image file is truncated (79 bytes not processed)

# OUTPUT TRAIN FOLDER:
# 3917.jpg
# image file is truncated (39 bytes not processed)
# 101947.jpg
# image file is truncated (82 bytes not processed)
# 79499.jpg
# image file is truncated (4054 bytes not processed)
# 95347.jpg
# image file is truncated (0 bytes not processed)
# 91033.jpg
# image file is truncated (86 bytes not processed)
# 92899.jpg
# image file is truncated (24 bytes not processed)
# 41945.jpg
# image file is truncated (53 bytes not processed)
