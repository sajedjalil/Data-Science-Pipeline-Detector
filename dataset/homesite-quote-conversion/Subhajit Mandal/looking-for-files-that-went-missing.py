import os
import fnmatch

print(os.listdir('/'))

path = '/'

for dirpath, dirnames, files in os.walk(path):
    for f in fnmatch.filter(files, 'train.csv'):
        print(os.path.join(dirpath, f))