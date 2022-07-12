#!/usr/bin/env python
#NIH Seizure code - Kaggle Competition
#Copyright 2016 Daniel Grunberg
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#Compares EEG files with time shift to find matches, just matches on channel 1
#export PYTHONUNBUFFERED=1 for unbuffered output
#
import os
import glob
import re
import sys
import numpy as np, h5py 
import pandas as pd
import math
import scipy.signal
import matplotlib.pyplot as plt
import argparse
import collections
from scipy.io import loadmat

parser = argparse.ArgumentParser(
    description="compare data files",
    epilog='''
    If files are specified on command line, will compare and plot those
''')

parser.add_argument("--version", action="version", version="0.01")
parser.add_argument("--datadir", help="data directory (default ../input)", default="../input/")
parser.add_argument("--limit", help="do 50 files, just for a test", action='store_true')
parser.add_argument("--patient", help="patient to do (default 1)", type=int, default=1)
parser.add_argument("--files", help="list files to use instead of directories", nargs='+')
parser.add_argument("--full", help="compare all files against each other (default 0 vs. 1s)", action='store_true')
options = parser.parse_args()
data_dir=options.datadir
if data_dir[-1]!='/':
    data_dir=data_dir+'/'

#########################################################
def read_file(file, seq=True):
    #return train data as np.array and sequence number
    #raise exception if file does not exist
    matdata = loadmat(file)
    d=matdata['dataStruct']
    if seq:
        s=d['sequence'][0,0][0,0]
    else:
        s=0
    ret=d['data'][0,0]
    return ret, s
#########################################################
def remove_dropouts(data):
    #data is 240,000 x 16
    dropout=(data==0.0).all(axis=1)
    num_dropouts=dropout.sum()
    #print 'number of dropouts {}'.format(num_dropouts)
    return data[~dropout, :]
#########################################################
def count_valid(data):
    #data is 240,000 x 16
    return (data!=0.0).any(axis=1).sum()
#########################################################
def compare(file1, file2):
    try:
        data1, seq1 = read_file(file1, False)
    except IOError:
        raise ValueError('file {} does not exist, skipping'.format(file1))
    except Exception as ex:
        raise ValueError('{} file {} problem, skipping'.format(ex, file1))
    try:
        data2, seq2 = read_file(file2, False)
    except IOError:
        raise ValueError('file {} does not exist, skipping'.format(file2))
    except Exception as ex:
        raise ValueError('{} file:{} problem, skipping'.format(ex, file2))
    #compute rmse between channels with time variance
    #data should be 240,000 by 16
    #print 'file {} min {} max {}'.format(file1, data1.min(), data1.max())
    data1=remove_dropouts(data1)
    data2=remove_dropouts(data2)
    #make them be the same size
    rows=min(data1.shape[0], data2.shape[0])
    #print 'got {} rows'.format(rows)
    data1=data1[0:rows,:]
    data2=data2[0:rows,:]
    #plt.plot(data1[:,0])
    #plt.plot(data2[:,0]+300)
    lowest=1e6
    off=-1000
    for var in range(10):
        if var==0:
            num=data1.size
            e=(abs(data1-data2)).mean()
            if e < lowest:
                lowest=e
                off=var
        else:
            #negative shift
            d1=data1[var:, :]
            d2=data2[0:-var, :]
            #print d1.shape, d2.shape
            num=d1.size
            e=(abs(d1-d2)).mean()
            if e < lowest:
                lowest=e
                off=var
            #positive shift
            d2=data2[var:, :]
            d1=data1[0:-var, :]
            num=d1.size
            e=(abs(d1-d2)).mean()
            if e < lowest:
                lowest=e
                off=var
    if lowest < 2.0:
        head1,tail1=os.path.split(file1)  # tail is filename
        head2,tail2=os.path.split(file2)  # tail is filename
        print('MATCH {:15} {:15} mean diff {:.5f} offset {:3}'.format(tail1, tail2, lowest, off))
############################################
### MAIN
if options.files is None:
    for t in ['train']:
        for patient in [options.patient]:
            files='{}{}_{}/*.mat'.format(data_dir, t, patient)
            print(files)
            file_list = sorted(glob.glob(files))
            print('found {} files'.format(len(file_list)))
    #Just do the first 20 files for testing
    if options.limit:
        file_list=file_list[0:50]
else:
    file_list=list()
    for f in options.files:
        mo=re.match('(\d)_', f)
        patient=mo.group(1)
        file_list.append(data_dir + 'train_' + patient + '/' + f)
    print('using file list:', file_list)

ddd=collections.OrderedDict()
for i, file in enumerate(file_list):
    head,tail=os.path.split(file)  # tail is filename
    try:
        data, seq = read_file(file, False)
    except IOError:
        raise ValueError('file {} does not exist, skipping'.format(file))
    except Exception as ex:
        print('{} file {} problem, skipping'.format(ex, file))
        continue
    p=count_valid(data)
    if p < 500:
        print('got {} rows for {}, skipping'.format(p, file))
        continue
    #resample
    s = scipy.signal.resample(data[:,0], 15000)  # resample to 15000 (25 Hz)
    ddd[tail]=np.float32(s)
    if i%100==0: print(i)
print('Read in data from files')
#generate 2 lists to compare: file_list_a and file_list_b
#if same_list is True, we will do a triangular compare so that we do not
#compare
if options.full:
    file_list_b=file_list_a=ddd.keys()
    same=True
    print('doing full cross compare')
else:
    file_list_a=list()
    file_list_b=list()
    for k in ddd.keys():
        mo=re.match('(\d+)_(\d+)_(\d+).mat', k)
        klass=mo.group(3)
        if klass=='0':
            file_list_a.append(k)
        elif klass=='1':
            file_list_b.append(k)
        else:
            print('unknown klass:', klass)
    print('comparing _0 to _1 files only 0:{} 1:{}'.format(len(file_list_a), len(file_list_b)))
    same=False
    
#now do the cross correlation
#either compare _0 to _1's or do full cross compare (--full option)
for i, file1 in enumerate(file_list_a):
    head1,tail1=os.path.split(file1)  # tail is filename
    print(i)
    if same:
        start=i+1
    else:
        start=0
    for j in range(start, len(file_list_b)):
        file2=file_list_b[j]
        head2,tail2=os.path.split(file2)  # tail is filename        
        v=np.correlate(ddd[tail1], ddd[tail2], mode='same')
        q=np.argmax(v)
        d=q-15000/2
        #print 'files {} {} {}'.format(file1,file2,d)
        if v[q] > 1.0e6:
            #found something big
            if d==0:
                err=abs(ddd[tail1] - ddd[tail2]).mean()  
            elif d>0:
                err=abs(ddd[tail1][d:] - ddd[tail2][:-d]).mean()
            else:
                err=abs(ddd[tail2][-d:] - ddd[tail1][:d]).mean()
            flag=False
            c = (v > 0.5*v[q]).sum()
            if (c < 5):
                #This is a peak in correlation and real
                flag=True
            if flag:
                print('MATCHED {:15} {:15} {:10} {:10.3g} {:6} c {:6} err {:6}'.format(tail1, tail2, q-15000/2, v[q], d, c, err))
        if options.files:
            plt.subplot(211)
            r=ddd[tail1].shape[0]
            plt.plot(np.arange(r), ddd[tail1])
            plt.plot(d+np.arange(r),ddd[tail2]+600)
            plt.subplot(212)
            plt.plot(v)
            plt.show()
