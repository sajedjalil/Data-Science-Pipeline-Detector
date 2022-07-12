# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import hashlib

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
    
    
print("category_names", md5("../input/category_names.csv"))
print("sample_submission", md5("../input/sample_submission.csv"))
print("test", md5("../input/test.bson"))
print("train", md5("../input/train.bson"))
print("train_example", md5("../input/train_example.bson"))