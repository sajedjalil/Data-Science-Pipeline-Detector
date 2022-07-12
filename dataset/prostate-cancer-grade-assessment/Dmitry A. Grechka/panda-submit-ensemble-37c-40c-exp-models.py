import time
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import tensorflow_addons as tfa
import math
import os
from skimage import io
import cv2

#tf.debugging.set_log_device_placement(False)

isTestSetRun = os.path.exists('../input/prostate-cancer-grade-assessment/test_images')
testCsvPath = '../input/prostate-cancer-grade-assessment/test.csv'
if isTestSetRun:
    input_dir = '../input/prostate-cancer-grade-assessment/test_images'
    testIdents = list(pd.read_csv(testCsvPath).image_id)
else:
    input_dir = '../input/prostate-cancer-grade-assessment/train_images'
    testIdents = [x[0:-5] for x in os.listdir(input_dir) if x.endswith(".tiff")][0:64]
checkpoint_path = '../input/panda-37c/train2/fold_3/weights.hdf5'
out_file = 'submission.csv'
backboneFrozen = True
batchSize = 4
trainSequenceLength = 64
initial_downscale_factor = 1
tileSize = 256
outImageSize = 224
prefetchSize = multiprocessing.cpu_count() + 1

modelsToEnsamble = [
    ('../input/panda-37c/train2/fold_2/weights.hdf5', 0.0),
    ('../input/panda-37c/train2/fold_3/weights.hdf5', 0.0),
    ('../input/panda-40c/train2/fold_4/weights.hdf5', 0.0),
    ('../input/panda-40c/train2/fold_5/weights.hdf5', 0.0)
]

cpuCores = multiprocessing.cpu_count()
print("Detected {0} CPU cores".format(cpuCores))

inputFiles = ["{0}.tiff".format(x) for x in testIdents if os.path.exists(os.path.join(input_dir,"{0}.tiff".format(x)))]
inputIdents = [x[0:-5] for x in inputFiles]

imageCount = len(inputIdents)
print("Found {0} files for inference".format(imageCount))
fullPaths = [os.path.join(input_dir,"{0}.tiff".format(x)) for x in inputIdents]



def GetInferenceDataset(fullPaths, rotDegree):
    filenameDs = tf.data.Dataset.from_tensor_slices(fullPaths)

    def TrimBlackPaddings(image):
        shape = image.shape
        rowsAggregated = np.amax(image,axis=(0,2))
        rowIndices = np.where(rowsAggregated != 0)[0]
        if len(rowIndices) == 0:
            print("WARN: entire black image in TrimBlackPaddings")
            return image # entire black image
        firstRow,lastRow = rowIndices[0], rowIndices[-1]

        colsAggregated = np.amax(image, axis=(1,2))
        colIndices = np.where(colsAggregated != 0)[0]
        if len(colIndices) == 0:
            print("WARN: entire black image in TrimBlackPaddings")
            return image # entire black image

        firstCol, lastCol = colIndices[0], colIndices[-1]
        
        return image[firstCol:(lastCol+1), firstRow:(lastRow+1), :]

    def EnlargeForRotationTF(imageTensor):
        radiansDegree = rotDegree / 180.0 * math.pi
        def EnlargeForRotation(image):
            h,w,_ = image.shape
            diag = math.sqrt(h*h + w*w)
            alphaCos = w / diag
            alpha = math.acos(alphaCos)
            beta = alpha - radiansDegree

            w2 = diag * abs(math.cos(beta))
            h2 = diag * abs(math.sin(beta))

            beta2 = alpha + radiansDegree
            w3 = diag * abs(math.cos(beta2))
            h3 = diag * abs(math.sin(beta2))

            w_ext = int(max(w2, w3, w))
            h_ext = int(max(h2, h3, h))

            padH = h_ext - h
            padW = w_ext - w 

            # print("init shape {0} {1}. rot {2}; pad h {3}; pad w {4}".format(h,w,rotDegree, padH, padW))

            #print("padding")
            paddedImage = np.pad(image, (
                (padH // 2, padH // 2),
                (padW // 2, padW // 2),
                (0,0)))
            return paddedImage
        return tf.numpy_function(EnlargeForRotation, [imageTensor], tf.uint8)

    def getNotEmptyTiles(image, tileSize, precomputedTileIndices=None, emptyCutOffMaxThreshold = 25):
        '''Returns the list of non-empty tile indeces (tile_row_idx,tile_col_idx) and corresponding list of tile npArrays).
        Each index list element specifies the zero based index (row_idx,col_idx) of square tile which contain some data (non-empty)'''

        # negate image (as lots of further processing operations add zero paddings)
        # we need black background

        h,w,_ = image.shape
        vertTiles = math.ceil(h / tileSize)
        horTiles = math.ceil(w / tileSize)
        indexResult = []
        dataResult = []
        tileIntensity  = []

        def extractTileData(row_idx,col_idx):
            tileYstart = row_idx*tileSize
            tileYend = min((row_idx+1)*tileSize,h)
            tileXstart = col_idx*tileSize
            tileXend = min((col_idx+1)*tileSize,w)
            tile = image[
                tileYstart:tileYend,
                tileXstart:tileXend,
                :] # all 3 color channels
            return tile

        def coerceTileSize(tile):
            """In case tile is smaller than requested size, it is padded with white content"""
            # we may need to pad the edge tile to full requested tile size
            th,tw,_ = tile.shape
            xPad = tileSize - tw
            yPad = tileSize - th
            if (xPad>0) | (yPad>0):
                # we pad to the end
                tile = np.pad(tile,[[0,yPad],[0,xPad],[0,0]],constant_values = 0)
            return tile

        # we analyze all tiles for the content
        for row_idx in range(0,vertTiles):
            for col_idx in range(0,horTiles):
                tile = extractTileData(row_idx, col_idx)
                # if the tile contains pure white pixels (no information) we skip it (do not return)
                #print("row {0} col {1} min_v {2}".format(row_idx,col_idx,tileMin))
                
                tileMax = np.nanmax(tile)
                if tileMax < emptyCutOffMaxThreshold: # too black! there is no tissue areas
                    continue
            
                tile = coerceTileSize(tile)

                indexResult.append((row_idx,col_idx))
                dataResult.append(tile)
                
                tileMean = np.nanmean(tile)
                tileIntensity.append(tileMean)
            
        # sorting the tiles according to intensity
        resIdx = []
        resData = []
        sortedIntencites = []
        for (idxElem,dataElem,sortedIntence) in sorted(zip(indexResult, dataResult, tileIntensity), key=lambda pair: -pair[2]):
            resIdx.append(idxElem)
            resData.append(dataElem)
            sortedIntencites.append(sortedIntence)

        indexResult = resIdx
            
        return indexResult,resData

    def getImageMean_withoutPureBlack(image):
        black = np.array([0,0,0])
        cleared = np.where(image==black,np.NaN,image)
        result = np.nanmean(cleared)
        if np.isnan(result):
            result = 0.0 # pure black
        return result

    def getImageMean_withoutBlack(image,blackThreshold):
        #print("image shape is {0}".format(image.shape))
        imageBrightness = np.nanmax(image, axis=-1)

        blackMap = imageBrightness < blackThreshold
        blackMap = np.expand_dims(blackMap,axis=-1)
        blackMap = np.tile(blackMap, (1,1,3))
        if not(np.any(blackMap)):
            # nonNan exists
            cleared = np.where(blackMap,np.NaN,image)
            return np.nanmean(cleared)
        else:
            return np.NaN # pure black
        
    def getImageContrast_withoutPureBlack(image, regTerm=0.0, precomputedMu = None):
        """(0,0,0) pixels are excluded from contrast evaluation"""
        if precomputedMu is None:
            mu = getImageMean_withoutPureBlack(image)
        else:
            mu = precomputedMu
        diff = image - mu
        return getImageContrast(image,regTerm=regTerm, precomputedDiff=diff)

    def getImageContrast(image, regTerm=0.0, precomputedDiff = None):
        """Entire image contrast as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
        if precomputedDiff is None:
            mu = getImageMean_withoutPureBlack(image)
            diff = image - mu
        else:
            diff = precomputedDiff
        
        squaredDiff = diff*diff
        meanSqDiff = np.mean(squaredDiff)
        return math.sqrt(regTerm + meanSqDiff)

    def GCN(image,lambdaTerm = 0.0, eps=1e-8, precomputedContrast = None, precomputedMean = None):
        """Global Contrast Normalization as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
        if precomputedMean is None:
            mu = np.mean(image)
        else:
            mu = precomputedMean
        
        diff = image - mu
        if precomputedContrast is None:
            initialContrast = getImageContrast(image, lambdaTerm, diff)
        else:
            initialContrast = precomputedContrast
        return diff / max(eps, initialContrast)

    def GCNtoRGB_uint8(gcnImage, cutoffSigmasRange = 3.0):
        """gcnImage is considered to have 0 mean and stddev == 1.0"""
        #print("GCNtoRGB_uint8 call")
        rescaled = (gcnImage + cutoffSigmasRange)/(2.0 * cutoffSigmasRange) * 255.0
        return np.round(np.minimum(np.maximum(rescaled,0.0),255.0)).astype(np.uint8)

    def getTiles(im):
        #print("getTiles: image type {0}. shape {1}. dtype {2}".format(type(im),im.shape, im.dtype))
        
        _,tiles = getNotEmptyTiles(im, tileSize, emptyCutOffMaxThreshold=10)

        #if len(tiles) > trainSequenceLength:
        #    tiles = tiles[0:trainSequenceLength]

        if len(tiles) == 0:
            print("WARN: Image {0} resulted in 0 tiles. producing blank (black) single tile TfRecords file".format(ident))
            tiles = [ np.zeros((tileSize,tileSize,3),dtype=np.uint8) ]
        else:
            # filtering out non green
            nonRedTiles = []
            for tile in tiles:
                red = np.mean(tile[:,:,0]) # supposing RGB, not BGR
                green = np.mean(tile[:,:,1])
                #print("[R:{0}\tG:{1}; ratio {2}".format(red,green, green/red))
                if (red > 0) and (green / red >= 1.2): # green must be at least 1.5 times more than red (to remove white marker tiles)
                    nonRedTiles.append(tile)
            if len(nonRedTiles)>0:
                tiles = nonRedTiles
            else:
                print("WARN: omited non-green tiles filtering, as the filtering impose the empty tile set")

            # normalizing with contrasts
            contrasts = []
            means = []
            #print("normalzing {0} tiles".format(len(tiles)))
            for tile in tiles:        
                mu = getImageMean_withoutBlack(tile, 40.0)
                if not np.isnan(mu):
                    contrast = getImageContrast_withoutPureBlack(tile, precomputedMu=mu)
                    means.append(mu)
                    contrasts.append(contrast)
            if len(means)>0: # whole image is not entirly black
                meanContrast = np.mean(contrasts)    
                meanMean = np.mean(means)
                if meanMean > 0.0:
                    for j in range(0,len(tiles)):
                        tiles[j] = GCNtoRGB_uint8(GCN(tiles[j], lambdaTerm=0.0, precomputedContrast=meanContrast, precomputedMean=meanMean), cutoffSigmasRange=2.0)
        
        tilePack = np.stack(tiles)
        #print("result of getTiles: shape {0}. dtype {1}".format(tilePack.shape, tilePack.dtype))

        return tilePack


    def coerceSeqSize(imagePack, trainSequenceLength):
        imagePackShape = tf.shape(imagePack)
        outputShape = [
            trainSequenceLength,
            imagePackShape[1],
            imagePackShape[2],
            imagePackShape[3]
        ]
        T = imagePackShape[0]

        availableIndices = tf.range(T)


        # if T is less than trainSequenceLength we need to duplicate the layers
        seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T), tf.int32)
        notTooShortIndicies = \
        tf.cond(seqRepCount > 1, \
            lambda : tf.tile(availableIndices, [seqRepCount]), \
            lambda : availableIndices)

        # if T is greater than trainSequenceLength we need to truncate it
        notTooLongIndices = notTooShortIndicies[0:trainSequenceLength]
        #notTooLong = tf.IndexedSlices(imagePack,notTooLongIndices, dense_shape = outputShape)
        notTooLong = tf.gather(imagePack, notTooLongIndices)
        shapeSet = tf.reshape(notTooLong,outputShape)
        return shapeSet

    def augment(imagePack):
        def augmentSingle(image):
            augSwitches = tf.cast(tf.math.round(tf.random.uniform([3],minval=0.0, maxval=1.0)),dtype=tf.bool)
            image = tf.cond(augSwitches[0], lambda: tf.image.rot90(image), lambda: image)
            image = tf.cond(augSwitches[1], lambda: tf.image.flip_left_right(image), lambda: image)
            image = tf.cond(augSwitches[2], lambda: tf.image.flip_up_down(image), lambda:image)
            return image
        return tf.map_fn(augmentSingle, imagePack)


    def trimBlackMarginsTF(imageTensor):
        withoutMargins = tf.numpy_function(TrimBlackPaddings,[imageTensor], Tout=tf.uint8)
        return withoutMargins

    @tf.function
    def rotateTF(imageTensor):
        withoutMargins = trimBlackMarginsTF(imageTensor)

        if abs(rotDegree % 360.0) > 0.1:
            enlarged = EnlargeForRotationTF(withoutMargins)
            degRad = rotDegree / 180.0 * math.pi
            rotated = tfa.image.rotate(enlarged, degRad, interpolation="BILINEAR")
            withoutMargins2 = trimBlackMarginsTF(rotated)
            return withoutMargins2
        else:
            return withoutMargins
    

    def decodeTiffTF(pathTensor):
        def decodeTiff(path):
            path = path.decode("utf-8")
            image = io.MultiImage(path)[1] 
            #print("decoded image type {0}. shape {1}. dtype {2}".format(type(image),image.shape, image.dtype))
            return image
        return tf.numpy_function(decodeTiff,[pathTensor], Tout=(tf.uint8))

    def loadAsTilePackTF(hiResImageTensor):
        def loadAsTilePack(hiResImage):
            tiles = getTiles(hiResImage)
            return tiles
        tensorPack = tf.numpy_function(loadAsTilePack, [hiResImageTensor], Tout=tf.uint8)
        return tf.reshape(tensorPack,[-1, tileSize, tileSize, 3])

    def debugPrinterTF(tensor):
        def debugPrinter(data):
            print("debug printer. shape {0}. dtype {1}".format(data.shape, data.dtype))
            return data
        return tf.numpy_function(debugPrinter,[tensor],tf.uint8)

    def resizePackTF(tilePack):
        shaped = tf.reshape(tilePack,[-1, tileSize, tileSize, 3])
        resized = tf.image.resize(shaped, [outImageSize, outImageSize], method=tf.image.ResizeMethod.GAUSSIAN)
        conveted = tf.cast(tf.round(resized),tf.uint8)
        return conveted

    @tf.function
    def negativeImage(image):
        #print("in negate: image is {0}".format(image))
        return 255 - image

    def process(tilePack):
        #return coerceSeqSize(tilePack,trainSequenceLength)
        return augment(coerceSeqSize(tilePack,trainSequenceLength))

    #print("filenameDs:")
    #print(filenameDs)

    #tf.data.experimental.AUTOTUNE
    #.map(debugPrinterTF) \

    imagesDs = filenameDs \
        .map(decodeTiffTF, num_parallel_calls=cpuCores *2, deterministic=True) \
        .map(negativeImage, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic= True) \
        .map(rotateTF, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic= True) \
        .map(loadAsTilePackTF, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic= True) \
        .map(resizePackTF, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic= True) \
        .map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic= True) \
        .batch(batchSize, drop_remainder=False) \
        .prefetch(prefetchSize)

    return imagesDs


def constructModel(seriesLen, DORate=0.2, l2regAlpha = 1e-3):
    imageSize = 224
    netInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="input")
    denseNet = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(imageSize, imageSize, 3),
        # it should have exactly 3 inputs channels,
        # and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value
        pooling=None)  # Tx8x8x1024 in case of None pooling
    
    converted = tf.keras.applications.densenet.preprocess_input(netInput)
    print("converted input shape {0}".format(converted.shape))
    cnnOut = tf.keras.layers.TimeDistributed(denseNet, name='cnns')(converted)  # Tx7x7x1024 in case of None pooling
    print("cnn out shape {0}".format(cnnOut.shape))
    cnnPooled = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((7, 7)), name='cnnsPooled')(
        cnnOut)  # Tx1x1x1024
    cnnPooledReshaped = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((1024,)), name='cnnsPooledReshaped')(
        cnnPooled)  # Tx1024
    cnnPooledReshapedDO = tf.keras.layers.Dropout(rate=DORate, name='cnnsPooledReshapedDO')(
        cnnPooledReshaped)  # Tx1024
    perSliceDenseOut = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(256,
        activation="selu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)), name='perSliceDenseOut')(
        cnnPooledReshapedDO)  # 128.   1024*128  parameters
    perSliceDenseOutDO = tf.keras.layers.Dropout(rate=DORate, name='perSliceDenseOutDO')(
        perSliceDenseOut)
    perSliceDenseOut2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128,
        activation="selu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)), name='perSliceDenseOut2')(
        perSliceDenseOutDO)  # 128.   1024*128  parameters
    perSliceDenseOutDO2 = tf.keras.layers.Dropout(rate=DORate, name='perSliceDenseOutDO2')(
        perSliceDenseOut2)
    #gru1 = tf.keras.layers.GRU(128, return_sequences=True)
    #gru1back = tf.keras.layers.GRU(128, return_sequences=True, go_backwards=True)
    #gru1out = tf.keras.layers.Bidirectional(gru1, backward_layer=gru1back, name='rnn1')(perSliceDenseOutDO)
    #gru1outDO = tf.keras.layers.Dropout(rate=DORate, name='rnn1DO')(gru1out)

    #, batch_input_shape=(1, seriesLen, 128)
    # , implementation=1

    rnnOut = \
        tf.keras.layers.GRU(
            96, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha),
            return_sequences=True)(perSliceDenseOutDO2)
    rnnOutDO = tf.keras.layers.Dropout(rate=DORate,name='rnnDO')(rnnOut)
    rnnOut2 = \
        tf.keras.layers.GRU(
            64, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha))(rnnOutDO)
    rnnOutDO2 = tf.keras.layers.Dropout(rate=DORate,name='rnn2DO')(rnnOut2)
    # predOut = \
    #     tf.keras.layers.Dense(6,name="resLogits",activation="linear",
    #     kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)
    #     )(rnnOutDO)
    predOut = \
        tf.keras.layers.Dense(1,name="unitRes",activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)
        )(rnnOutDO2)
    predOutScaled = \
        tf.keras.layers.Lambda(lambda x: x*5.0, name="scaledRes")(predOut)


    return tf.keras.Model(name="PANDA_A", inputs=netInput, outputs=predOutScaled), denseNet

model,backbone = constructModel(trainSequenceLength, DORate=0.4, l2regAlpha = 1e-4)
print("model constructed")
print(model.summary())
if backboneFrozen:
    backbone.trainable = False
else:
    backbone.trainable = True


def predict(checkpointPath, rotDegree):
    if os.path.exists(checkpointPath):
        print("Loading pretrained weights {0}".format(checkpointPath))
        model.load_weights(checkpointPath, by_name=True)
        print("Loaded pretrained weights {0}".format(checkpointPath))
    else:
        print("Pretrained weights file does not exist: {0}".format(checkpointPath))
        exit(1)

    imagesDs = GetInferenceDataset(fullPaths, rotDegree)
        
    t1 = time.time()
    #iterated = list(imagesDs.take(3).as_numpy_iterator())
    print("predicting using {0} with deg {1}".format(checkpointPath, rotDegree))
    predicted = np.squeeze(model.predict(imagesDs,verbose=1))
    t2 = time.time()
    print("predicted shape is {0}".format(predicted.shape))
    print("predicted {0} samples in {1} sec. {2} sec per sample".format(imageCount,t2-t1,(t2-t1)/imageCount))
    return predicted

modelsCount = len(modelsToEnsamble)
accumulatedPrediction = None
perModelPrediction = []
for (checkpoint_path,rotDegree) in modelsToEnsamble:
    predicted = predict(checkpoint_path, rotDegree)
    perModelPrediction.append(predicted)
    if accumulatedPrediction is None:
        accumulatedPrediction = np.copy(predicted)
    else:
        accumulatedPrediction = np.add(accumulatedPrediction, predicted)
ensembledPrediction = accumulatedPrediction / float(modelsCount)

predictedList = np.round(ensembledPrediction).astype(np.int32).tolist()
resDf = pd.DataFrame.from_dict({'image_id':inputIdents , 'isup_grade': predictedList})
resDf.sort_values(by=['image_id'], inplace=True, ascending=True)
resDf.to_csv(out_file, index=False)

perModelDict = {'image_id':inputIdents , 'isup_grade': predictedList, 'ensembled':ensembledPrediction.tolist()}
for i in range(0,modelsCount):
    perModelDict["isup_grade_{0}".format(i)] = perModelPrediction[i].tolist()
resDf2 = pd.DataFrame.from_dict(perModelDict)
resDf2.sort_values(by=['image_id'], inplace=True, ascending=True)
resDf2.to_csv("per_model_predictions.csv", index=False)

print('Done')