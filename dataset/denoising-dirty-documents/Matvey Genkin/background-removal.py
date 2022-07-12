import numpy as np
from scipy import signal
import skimage
import skimage.io
import skimage.morphology
import skimage.filters
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import sklearn
import csv

im_num_test, im_num_train = [], []
for path in os.listdir("../input/test"):
    im_num_test.append(int(path[:-4]))
for path in os.listdir("../input/train"):
    im_num_train.append(int(path[:-4]))
IM_SAVE_CT = 0
def show(img):
    global IM_SAVE_CT
    plt.imshow(img, cmap = cm.Greys_r)
    plt.savefig("output"+str(IM_SAVE_CT)+".png")
    IM_SAVE_CT += 1

def RMSE(im_list_1, im_list_2):
    px1 = np.hstack([im.flatten() for im in im_list_1])
    px2 = np.hstack([im.flatten() for im in im_list_2])
    return np.sqrt(np.sum((px1-px2)**2)/float(px1.size))    

def denoise(im):
    out = np.zeros(im.shape, dtype=im.dtype)
    for index, value in np.ndenumerate(im):
        out[index] = 1 if value>0.1 else 0
    return out

def find_best_threshold(im, truth):
    rng = [.1, .01]
    choices = np.linspace(0,1,10, endpoint=False)
    error = np.zeros((10))
    m = 0
    for i in range(2):
        for j in range(10):
            out = (im > choices[j])
            error[j] = RMSE([out], [truth])
            print("thresh "+str(choices[j])+" error "+str(error[j]))
        m = np.argmin(error)
        choices =  np.linspace(choices[m]-.5*rng[i], choices[m]+.5*rng[i], 10, endpoint=False)
    return choices[m]
        
def plot_thresh_RMSE(im, truth):
    thresh_list = np.linspace(0,1,endpoint=False)
    RMSE_list = list(map(lambda thresh : RMSE([im*(im>thresh*np.ones(im.shape, dtype=im.dtype))], [truth]), thresh_list))
    plt.figure(int(random.random()*200))
    plt.plot(thresh_list, RMSE_list)
    plt.savefig("plot"+str(random.random())[3:6]+".png")

def make_submission_csv():
    global im_num_test
    f = open("submit.csv", "wt")
    w = csv.DictWriter(f, fieldnames=["id","value"])
    w.writeheader()
    for i in im_num_test:
        test_im = skimage.img_as_float(skimage.io.imread("../input/test/"+str(i)+".png", as_grey=True))
        pred_im = denoise(test_im)
        for index, value in np.ndenumerate(pred_im):
            w.writerow({"id":str(i)+"_"+str(index[0])+"_"+str(index[1]),"value":str(value)})


im_train, im_train_cleaned, im_pred= [], [], []
for i in im_num_train:
    im_train.append(skimage.img_as_float(skimage.io.imread("../input/train/"+str(i)+".png", as_grey=True)))
    im_train_cleaned.append(skimage.img_as_float(skimage.io.imread("../input/train_cleaned/"+str(i)+".png", as_grey=True)))
print(str(RMSE(im_train, im_train_cleaned))+" with no cleaning")

for i in range(len(im_train)):
    im_pred.append(denoise(im_train[i]))

t, m = [], []
for i in range(len(im_train)):
    tryim = skimage.filters.gaussian_filter(im_train[i], sigma=0.4)
    trueim = im_train_cleaned[i]
    thresh = find_best_threshold(tryim, trueim)*np.ones((tryim.shape))
    clsd = np.logical_not(skimage.morphology.closing((tryim < thresh), selem = skimage.morphology.disk(1)))
    #plot_thresh_RMSE(tryim, trueim)
    #show(tryim)
    #show(trueim)
    #show((tryim>thresh))
    #show(clsd)
    #show(trueim)
    t.append(RMSE([(tryim>thresh)*tryim], [trueim]))
    m.append(RMSE([clsd*tryim], [trueim]))
    #print("just threshold       "+str(RMSE([(tryim>thresh)*tryim], [trueim])))
    #print("closed morph         "+str(RMSE([clsd*tryim], [trueim])))
    #print("just original image  "+str(RMSE([tryim], [trueim])))
print(str(np.mean(np.array(t)))+" mean rmse with just threshold")
print(str(np.mean(np.array(m)))+" mean rmse with threshold and morphological closing")

#make_submission_csv()