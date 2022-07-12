# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# move downloaded  images to ids named dirs 

import sys, os, csv

target_path = 'googlelandmark/train/'
origin_path = 'googlelandmark/download/'
data_file = 'googlelandmark/train.csv'


def move_img():
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  urls = []
  ids = []
  landmarks = []
  i = 0
  j = 0
  lines = ''
  for line in csvreader:
    if i > 0:  # skip table header
      # print(line)
      ids.append(line[0])
      urls.append(line[1])

      if line[2] not in landmarks:
        landmarks.append(line[2])
      if not os.path.exists(target_path + line[2]):
        os.makedirs(target_path + line[2])
        print(target_path + line[2])

      img_path = origin_path + line[0] + '.jpg'
      new_path = target_path + line[2] + os.sep + line[0] + '.jpg'
      if os.path.exists(img_path):
        os.rename(img_path,  new_path)
        print(img_path, new_path)
        j = j + 1
    i = i + 1
  print('all lines: ', i)
  print('rename files: ', j)


def count_file(path):
  for d in os.listdir(path):
    count = len([s for s in os.listdir(path + d)])
    if count > 20:
      print(d + "," + str(count))


if __name__ == '__main__':
#   move_img()
#   count_file(target_path)
  print("ok")