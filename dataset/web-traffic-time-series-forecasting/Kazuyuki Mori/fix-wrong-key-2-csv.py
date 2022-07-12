#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import pandas as pd
import os

def load_key2_csv(fixfile="./key_2_fix.csv"):
    if not os.path.exists(fixfile):
        fix_csv("../input/key_2.csv", fixfile)
    return pd.read_csv(fixfile)

def fix_csv(infile="../input/key_2.csv", outfile="./key_2_fix.csv"):
    print("{} -> {}".format(infile, outfile))
    print("...")
    f = codecs.open(infile, 'r', 'utf8')
    fix = codecs.open(outfile, 'w', 'utf8')
    line = f.readline().strip()
    fix.write(line+"\n")
    while True:
        line = f.readline().strip()
        if len(line) == 0:
            break
        s = line.split(u',')
        if len(s) > 2:
            s = [','.join(s[:-1]),s[-1]]
        s[0] = s[0].replace('"','""') # escape double quote
        fix.write(u'"' + s[0] + u'","' + s[1] + u'"\n')
    f.close()
    fix.close()

    print("Success")
    print("="*30)
    print("Usage: ")
    print("""key = pd.read_csv('{}')""".format(outfile))
    print("="*30)

def main():
    load_key2_csv()

if __name__ == '__main__':
    main()

