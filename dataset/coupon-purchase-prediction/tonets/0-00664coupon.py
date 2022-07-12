#!/usr/bin/env python
# -*- coding: utf-8 -*-
f = open('../input/sample_submission.csv', 'r')
line = f.readlines()
f.close()
f = open('submission.csv', 'w')
f.write(line[0].strip()+"\n")
del line[0]
for l in line:
    l = l.strip()
    user = l.split(',')[0]
    f.writelines(user + ",2fcca928b8b3e9ead0f2cecffeea50c1\n")
f.close()