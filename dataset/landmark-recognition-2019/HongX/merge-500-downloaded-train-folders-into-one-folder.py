from glob import glob
import shutil
import os

dst = '/path/to/dst_folder/'
for f1 in glob( '/path/to/download_folder/*'):
    print(f1)
    if len(f1.split('/')[-1])==1:
        for f2 in glob(f1+'/*'):
            for f3 in glob(f2+'/*'):
                src = f3+'/'
                files = os.listdir(src)
                for f in files:
                    shutil.move(src+f, dst)
                os.rmdir(f3)
            os.rmdir(f2)
        os.rmdir(f1)