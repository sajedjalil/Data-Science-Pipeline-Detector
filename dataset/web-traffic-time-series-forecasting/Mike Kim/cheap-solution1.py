import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import csv
from cmath import exp, pi
from collections import Counter
import random
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
random.seed(6)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.


GLOBAL_MEDIAN = 145
GLOBAL_PRED_LEN = 10#31+31+13# 75

DATES = ['2017-08-' + str(j) for j in range(22,32)]#22..31 inclusive, len is 10

def int_float(x):
    try:
       return int(x)
    except Exception:
       return float(x)

def median(lst):
    n = len(lst)
    if n < 1:
            return GLOBAL_MEDIAN
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def imputed(x,median_x):
    output_list = []
    for j in row[1:]:
        if j!='':
          output_list.append(int_float(j))
        else:
          output_list.append(median_x)
    return output_list

def build_model_dict(input_vals, list_mod_vals = [7,30,365], offset = 2):
    model_dict = {}
    both = [(i+offset,j) for i,j in enumerate(input_vals)]
    for val1 in list_mod_vals:
        for val2 in both:
            key1 = str(val1)
            key2 = str(val2[0] % val1)
            key3 = key1+"_"+key2
            if model_dict.get(key3) is None:
                model_dict[key3] = [val2[1]]
            else:
                model_dict[key3] += [val2[1]]
    return model_dict


#input_vals = range(80,100,2)
#tmp = build_model_dict(input_vals)
#print(tmp.keys())


#train_2 has  804 columns including Page, then dates
def predict_from_dict(new_row, list_mod_vals = [3,30,365], offset=2+804-1): #804+2offset-1page
    # an check via [(i+2,j) for i,j in enumerate(range(803))]
    median_list = []
    both = [(i+offset,j) for i,j in enumerate(new_row)]
    for val2 in both:
        tmp_list = []
        for val1 in list_mod_vals:
            key1 = str(val1)
            key2 = str(val2[0] % val1)
            key3 = key1+"_"+key2
            if model_dict.get(key3) is not None:
                tmp_list += model_dict[key3]
        median_list += [median(tmp_list)]
    return median_list
    
    

PATH = "../input/"
PERIODS = [7,28,30,31,365] #[7,28,29,30,31,365,366]

print('make key dict')
key_dict_id_page = {}
fname = PATH+"key_2.csv"
with open(fname, 'r') as csvfile:
   csvreader = csv.reader(csvfile, delimiter=',', quotechar="\"")
   header = next(csvreader)
   for row in csvreader:
      #page,id
      page = row[0].replace("\"",'')
      key_dict_id_page[hash(row[1])] = hash(page)


no_count = 0
count = 0
print('start ts_dict')
ts_dict = {}
gt_dict = {}
med_dict = {}
fname = PATH+"train_2.csv"
with open(fname, 'r') as csvfile:
   csvreader = csv.reader(csvfile, delimiter=',', quotechar="\"")
   header = next(csvreader)
   print(header[784])
   #print(header)
   for row in csvreader:
      count += 1
      if count % 2000==0:
         print(count)
         #break
      page = row[0]
      page = page.replace("\"",'')

      row10 = row[0:(len(row)-GLOBAL_PRED_LEN)]

      tmpvec = [int_float(j) for j in row10[1:] if j!='']
      tmp_median = median(tmpvec)
      tmpvec = imputed(row10[1:],tmp_median)
      model_dict = build_model_dict(tmpvec, list_mod_vals =PERIODS, offset = 2)
      new_row = row[(len(row)-GLOBAL_PRED_LEN):]
      tmpval = predict_from_dict(new_row, list_mod_vals = PERIODS, offset=2+784-1)
      for i,j in enumerate(DATES):
            ts_dict[hash(page+'_'+DATES[i])] = tmpval[i]
            gt_dict[hash(page+'_'+DATES[i])] = row[i+784]
            med_dict[hash(page+'_'+DATES[i])] = tmp_median




print('make file')

missing=0
y_true = []
y_pred = []
with open('cvsub.csv', 'a') as the_file:
   the_file.write('hash,pred,truth,med\n')
   for row in gt_dict.keys():
        row_hash = str(row)
        row_pred = str(ts_dict[row])
        row_truth = str(gt_dict[row])
        row_med = str(med_dict[row])
        the_file.write(row_hash+','+row_pred+','+row_truth+','+row_med + '\n')

print(missing)

tmp = pd.read_csv('cvsub.csv')
print(tmp.head(3))

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)
    
tmp_smape = smape(tmp['truth'],tmp['pred'])
print(tmp_smape)

tmp_smape = smape(tmp['truth'],tmp['med'])
print(tmp_smape)
#45.1771946128 is smape [7,30,31,365]


#45.1537489392
#46.5768373774
