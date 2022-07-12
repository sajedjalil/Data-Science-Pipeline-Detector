
# Any results you write to the current directory are saved as output.
import scipy.io as sio
import numpy as np
import glob
import os

def get_data(file):
	matfile = sio.loadmat(file)
	data = (matfile['dataStruct']['data'][0,0]).T
	return data

f1 = '../input/train_1/*_1.mat'
ff1 = glob.glob(f1)
ff1.sort()
f2 = '../input/train_1/*_0.mat'
ff2 = glob.glob(f2)
ff2.sort()
count = 0
st = 100
fi = 200
for i in ff2:
        d2 = get_data(i)
        for j in ff1:
                print(j)
                d1 = get_data(j)
                fcor = np.fft.ifft(np.fft.fft(d1[0,:])*np.conj(np.fft.fft(d2[0,:])))
                loc = fcor.argmax()
                if loc > 240000-fi:
                        continue
                one = np.sum(d1[0,loc+st:loc+fi])
                two = np.sum(d2[0,st:fi])
                sigma1 = np.std(d1[0,loc+st:loc+fi])
                sigma2 = np.std(d2[0,st:fi])
                diff = np.abs(np.sum(d1[0,loc+st:loc+fi] - d2[0,st:fi]))
                diff_sig = np.std(d1[0,loc+st:loc+fi] - d2[0,st:fi])
                # this is a little over kill ....
                if ((sigma1/sigma2 > 0.999) and (sigma1/sigma2 < 1.001) and (np.abs(one) > 0) and (np.abs(two) > 0) and (np.abs(one - two)/100. < 1) and (diff_sig < 1.0)):
                        count = count + 1
                        print(os.path.basename(i), os.path.basename(j), count)
                        print(loc, np.sum(d1[0,loc+st:loc+fi] - d2[0,st:fi]), diff_sig)
