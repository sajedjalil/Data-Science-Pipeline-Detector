import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold,train_test_split
import cv2
from skimage.transform import resize
import gc
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras import Model
from keras import backend as K


debug=False

num_folds = 3
lr = 0.1
img_size_ori = 101
img_size_target = 128
batch=10 if debug else 64
nrows=100 if debug else None
model_depth = 2 if debug else 5
ep = 50 if debug else 200


def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

def resample(img,img_size_target):
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    # print(y_true_in.shape)

    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def predict_generator(df_file_names, img_size, batch_size):
    number_of_batches = np.ceil(len(df_file_names) / batch_size)
    print(len(df_file_names), number_of_batches)
    counter = 0
    int_counter = 0

    while True:
            beg = batch_size * counter
            end = batch_size * (counter + 1)
            batch_files = df_file_names[beg:end]
            image_list = []

            for file in batch_files['id'].values:
                int_counter += 1
                try : 
                	image = np.array(load_img("../input/test/images/{}.png".format(file), grayscale=True,target_size=(img_size,img_size))).reshape((img_size,img_size,1)) /255.
                except Exception as e :
                	image = np.array(load_img("../input/train/images/{}.png".format(file),grayscale=True,target_size=(img_size,img_size))).reshape((img_size,img_size,1)) /255.

                image_list.append(image)

            counter += 1

            image_list = np.array(image_list)

            yield (image_list)

def holdout_generator(df_file_names, img_size, batch_size):
    number_of_batches = np.ceil(len(df_file_names) / batch_size)
    print(len(df_file_names), number_of_batches)
    counter = 0
    int_counter = 0

    while True:
            beg = batch_size * counter
            end = batch_size * (counter + 1)
            batch_files = df_file_names[beg:end]
            image_list = []

            for file in batch_files['id'].values:
                int_counter += 1
                image = np.array(load_img("../input/train/images/{}.png".format(file), grayscale=True,target_size=(img_size,img_size))).reshape((img_size,img_size,1)) /255.
                image_list.append(image)

            counter += 1

            image_list = np.array(image_list)

            yield (image_list)

def batch_generator_train(df_file_names, img_size, batch_size, is_train=True, shuffle=True,flip_lr = True,random_crop = False):
    number_of_batches = np.ceil(len(df_file_names) / batch_size)

    counter = 0
    while True:
        batch_files = df_file_names[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file in batch_files['id'].values:
            image = np.array(load_img("../input/train/images/{}.png".format(file),grayscale=True,target_size=(img_size,img_size))).reshape((img_size,img_size,1)) / 255. #cv2.resize(cv2.imread(file), (img_size,img_size)) / 255.
            mask = np.array(load_img("../input/train/masks/{}.png".format(file),grayscale=True,target_size=(img_size,img_size))).reshape((img_size,img_size,1)) / 255.

            if flip_lr : 
                r = np.random.randint(2,dtype=int)
                if r == 1 : 
                	image = cv2.flip(image, 0).reshape((img_size,img_size,1))
                	mask = cv2.flip(mask,0).reshape((img_size,img_size,1))

            image_list.append(image)
            mask_list.append(mask)
            

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            counter = 0

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0],nrows=nrows)
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

if debug : 
	test_df = test_df[:nrows]

counter = 0

callbacks = [EarlyStopping(monitor='val_loss',patience=10),
ModelCheckpoint('weights.h5',monitor='val_loss',save_best_only=True),
ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)]
kf = KFold(n_splits=num_folds)


ids,holdout_idx = train_test_split(train_df)
ids.reset_index(inplace=True)
holdout_idx.reset_index(inplace=True)
best_thres = 0
preds_test = np.zeros(shape=(len(test_df),img_size_target,img_size_target,1))

for train_idx,val_idx in kf.split(ids) :
	model = UNet((img_size_target,img_size_target,1),start_ch=16,depth=model_depth,batchnorm=True)
	model.compile(optimizer=Adam(lr=lr),loss = 'binary_crossentropy',metrics=[dice_coef,'accuracy'])

	model.fit_generator(batch_generator_train(df_file_names=train_df.loc[train_idx], img_size=img_size_target, batch_size=batch, is_train=True, shuffle=True),
		steps_per_epoch=np.ceil(len(train_idx)/batch),epochs=ep,callbacks=callbacks,
		validation_data=batch_generator_train(df_file_names=train_df.loc[val_idx], img_size=img_size_target, batch_size=batch, is_train=True, shuffle=True),
		validation_steps=np.ceil(len(val_idx)/batch))
	
	model.load_weights('weights.h5')
	holdout_preds = model.predict_generator(holdout_generator(holdout_idx,img_size=img_size_target,batch_size=batch),steps=np.ceil(len(holdout_idx)/batch))
	holdout_preds = np.array(list(resample(x,img_size_ori) for x in holdout_preds))

	thresholds = np.linspace(0, 1, 50)
	holdout_y = []
	for n in holdout_idx['id'].values : 
		ho_mask = np.array(load_img('../input/train/masks/{}.png'.format(n),grayscale=True,target_size=(img_size_ori,img_size_ori))).reshape((img_size_ori,img_size_ori,1)) / 255.
		holdout_y.append(ho_mask)

	holdout_y = np.array(holdout_y)

	ious = np.array(list(iou_metric_batch(holdout_y, np.int32(holdout_preds > threshold)) for threshold in thresholds))

	threshold_best_index = np.argmax(ious[9:-10]) + 9
	iou_best = ious[threshold_best_index]
	threshold_best = thresholds[threshold_best_index]
	print(' Got best iou of ',iou_best,' at threshold ', threshold_best)

	best_thres += threshold_best / num_folds
	preds_test += model.predict_generator(predict_generator(test_df,img_size=img_size_target,batch_size=batch),steps=np.ceil(len(test_df)/batch)) / num_folds
	
	del model;gc.collect()

pred_dict = {idx: RLenc(np.round(resample(preds_test[i],img_size_target = img_size_ori) > threshold_best),img_size_ori) for i, idx in enumerate(test_df['id'].values)}

if not debug : 
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_{}.csv'.format(counter))
    
    counter += 1