import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Input,Dropout,Dense,Activation,Convolution2D,MaxPooling2D,ZeroPadding2D,Flatten
from keras.optimizers import Adam,Adagrad,RMSprop,rmsprop,SGD
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras import backend as K
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
import time


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

ROWS = 48
COLS = 48
CHANNEL = 3
FOLDS = 5
classes = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
train_dir = '../input/train/'
test_dir = '../input/test_stg1/'
fish_train = [] ## path : fish class / file name
target = [] ## fish species

def get_img(img_fish):
    fish_dir = train_dir+'{}'.format(img_fish)
    images = [img_fish + '/' + str(i) for i in os.listdir(fish_dir)]
    return images

def read_img(imgs):
    img = cv2.imread(imgs,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(ROWS,COLS),interpolation = cv2.INTER_CUBIC)
    return img

def prepare_data():
    data = np.ndarray((len(fish_train),ROWS,COLS,CHANNEL),dtype = np.uint8)
    img_length = len(fish_train)
    for i,file in enumerate(fish_train):
        ## file path : FISH_CLASS / file name
        data[i] = read_img(train_dir + file)
        if i % 1000 ==0 :
            print('%d of %d  ' % (i,img_length))
    return data

print('Reading Image ....')
start_time = time.time()
for fish in classes:
    fish_img = get_img(fish) ## fish class / file name
    fish_train.extend(fish_img)
    
    fish_class = np.tile(fish,len(fish_img))
    target.extend(fish_class)
    print('%d photos of fish %s' %(len(fish_img),fish))
end_time = time.time()
print('Time is %f min' %((end_time - start_time) / 60))

print('\nProcessing ....')
start_time = time.time()
data = prepare_data()
print('All Image Shape : {}'.format(data.shape))
end_time = time.time()
print('Time is %f min' % ((end_time - start_time) / 60))
    

target = LabelEncoder().fit_transform(target)
target = np_utils.to_categorical(target)

X_train,X_valid,y_train,y_valid = train_test_split(data,target,
                                                   test_size=0.2,
                                                   random_state=42)

optimizer  = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

def normalization(pixel):
    return (pixel - K.mean(pixel)) / K.std(pixel)

def create_model():
    model = Sequential()
    model.add(Activation(activation = normalization,input_shape=(ROWS,COLS,CHANNEL)))
    model.add(ZeroPadding2D((1,1),dim_ordering='th'))
    model.add(Convolution2D(4,3,3,border_mode='same',activation='relu',dim_ordering='th'))
    model.add(ZeroPadding2D((1,1),dim_ordering='th'))
    model.add(Convolution2D(4,3,3,border_mode='same',activation='relu',dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),dim_ordering='th'))
    
    model.add(ZeroPadding2D((1,1),dim_ordering='th'))
    model.add(Convolution2D(8,3,3,border_mode='same',activation='relu',dim_ordering='th'))
    model.add(ZeroPadding2D((1,1),dim_ordering='th'))
    model.add(Convolution2D(8,3,3,border_mode='same',activation='relu',dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),dim_ordering='th'))
    
    '''
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16,3,3,border_mode='same',activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16,3,3,border_mode='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    '''
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.499))
    
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.499))
    
    model.add(Dense(len(classes),activation='sigmoid'))
    
    
    ## vgg-16
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #model.compile(optimizer,loss=objective)
    
    return model

print('Starting....')
start_time = time.time()
class History(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
        self.val_losses = []
    
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
    
def run_model():
    
    nb_epoch = 10
    batch_size = 64
    num_fold = 0
    score =0 
    sum_score =0 
    
    
    #### reading test file
    print('Reading test file ....')
    start_time = time.time()
    test_files = [i for i in os.listdir(test_dir)]
    x_test = np.ndarray((len(test_files),ROWS,COLS,CHANNEL),dtype=np.uint8)
    for i,file in enumerate(test_files):
        x_test[i] = read_img(test_dir + file)
    print('Finished....')
    print('Time is %f min' %((time.time() - start_time) / 60))
    
    
    
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = 3,
                               verbose=1,mode='auto')

    kf = KFold(len(data),n_folds=FOLDS,shuffle=True,random_state=42)

    
    start_time = time.time()
    history = History() ## you can do the data visualization to fine-tunning
   
    '''
    model = create_model()
    model.fit(X_train,y_train,
              batch_size=batch_size,nb_epoch=nb_epoch,
              shuffle=True,validation_split=0.2,
              callbacks=[history,early_stopping])
    y_pred = model.predict(x_test)
    '''
   
    
    for train_idx,test_idx in kf:
        X_train = data[train_idx]
        X_valid = data[test_idx]
        y_train = target[train_idx]
        y_valid = target[test_idx]
        model = create_model()
        num_fold +=1
        print('Start KFold {} from {}'.format(num_fold,FOLDS))
        print('Split train : {}'.format(len(X_train)))
        print('Split valid : {}'.format(len(X_valid)))
    
        model.fit(X_train,y_train,batch_size=batch_size,
                  nb_epoch = nb_epoch,validation_data=(X_valid,y_valid),
                  shuffle=True,callbacks = [history,early_stopping])
                  
        y_pred = model.predict(x_test,batch_size=batch_size)
        #score = log_loss(y_valid,y_pred)
        sum_score += y_pred
        #sum_score += score * len(test_idx)
    sum_score /= FOLDS
    print('Log loss train independent avg : {}'.format(sum_score))
   
   
    
    start_time = time.time()
    print('Writing ...')
    submission = pd.DataFrame(sum_score,columns=classes)
    submission.insert(0,'image',test_files)
    submission.to_csv('VGG_cnn_model.csv',index=False)
    print('Finished...')
    print('Time is %f min' %((time.time() - start_time) / 60))

if __name__ == '__main__':
    run_model()


