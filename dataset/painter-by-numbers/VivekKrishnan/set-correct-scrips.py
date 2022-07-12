# Run this bash script first in the image directories to set the correct
# extensions for all the image files

# for f in *jpg; do 
#     type=$(file -0 -F" " "$f" | grep -aPo '\0\s*\K\S+') 
#     mv "$f" "${f%%.*}.${type,,}"  
# done

# Then run this script to update the values in train_info.csv

import csv
import os, fnmatch

datadir = 'train/'
trainfile = 'train_info.csv'
newtrainfile = 'train_info_mod.csv'

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def fix_csvfile():
    with open(trainfile, 'rb') as csvfile:
        with open(newtrainfile, 'wt') as outfile:

	        outfile.truncate()
	        reader = csv.reader(csvfile)
	        writer = csv.writer(outfile)
	    
	        skip = True
	        for row in reader:
	            if (skip):
	                writer.writerow(row)
	                skip = False
	                continue
	        

	            filename = row[0].split('.')

	            basename = filename[0]
	            extension = filename[1]

	            f_name = find(basename + '.*', datadir)
	            f = f_name[0]
	            row[0] = f
	            writer.writerow(row)