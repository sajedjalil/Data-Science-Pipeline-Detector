# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import csv


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


color_code = {
    'po': Color.RED,
    'ao': Color.BLUE,
    'bo': Color.YELLOW
}

with open('../input/test_stage_1.tsv') as data:
    reader = csv.reader(data, delimiter='\t')
    next(reader)
    for i, _ in enumerate(reader, 1):
        print(i, ":", end='')
        offsets = [('po', int(_[3]), _[2]), ('ao', int(_[5]), _[4]), ('bo', int(_[7]), _[6])]
        offsets.sort(key=lambda x: x[1])
        seek = 0
        for element in offsets:
            print(_[1][seek:int(element[1])] + color_code[element[0]] + element[2] + Color.END, end='')
            seek = int(element[1]) + len(element[2])
        else:
            print(_[1][seek:], _[8])

# Any results you write to the current directory are saved as output.