#!/usr/bin/python2.7
import os
#os.system("ls ../input")
#os.system("echo \n\n")
#os.system("head ../input/*")
import csv
import random

submission = list(csv.reader(open('../input/sample_submission.csv')))
coupon_list_test = open('../input/coupon_list_test.csv', 'rb')
next(coupon_list_test)
coupon_id_hash = [str(row.split(b',')[-1][:-1], 'ascii') for row in coupon_list_test]
for row in submission[1:]:
    row[1] = ''.join(map(lambda s: ' ' + s, random.sample(coupon_id_hash, 10)))
submission_file = open('submission.csv', 'w')
for row in submission:
    submission_file.write(row[0])
    submission_file.write(',')
    submission_file.write(row[1])
    submission_file.write('\n')
os.system('chmod +x submission.csv')