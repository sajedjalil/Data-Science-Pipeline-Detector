# -*- coding: utf-8 -*-
# !/usr/bin/python

# Since some images fail to download we can form an index file for downloaded images only
#
# NOTE: This file does not run on Kaggle
#
# 1. Download images to some directory
# 2. Run this script pointing index file, images directory and output index file as arguments
# 3. Get index file with only downloaded images
#
# Average time running 1:30
#
# May be improved by multiprocessing

import sys
import os
import csv
import tqdm


def main():
    if len(sys.argv) != 4:
        print('Syntax: {} <data_file.csv> <out_dir/> <output_file.csv>'.format(
            sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir, out_file) = sys.argv[1:]

    final = []
    with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        print('Reading file...')
        for row in tqdm.tqdm(reader):
            filename = os.path.join(out_dir, '{}.jpg'.format(row[0]))
            if os.path.exists(filename):
                final.append(row)

    with open(out_file, 'w') as csvfile:
        writer = csv.writer(
            csvfile, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
        writer.writerow(['id', 'url', 'landmark_id'])
        print('Writing file...')
        for row in tqdm.tqdm(final, total=len(final)):
            writer.writerow([row[0], row[1], int(row[2])])

if __name__ == '__main__':
    main()
