
import warnings, gc
gc.collect()

# warnings.filterwarnings('always')
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')

import math, os, cv2, time, torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import sklearn as skl
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import torch
from torch import nn
import torch.utils.data as utils
import torchvision as tv
import torchvision.models as models


# Device configuration
computeDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weightedAvgContrast(a, w):
    average = np.average(a, weights=w)
    diff = np.average(abs(a - average), weights=w)
    return diff


def sigmoid(x, base=math.e):
    return 1 / (1 + base ** (-x))


def gammaToLinear(x):
    return x**2.2


def linearToGamma(x):
    return x**(1/2.2)


def normalize(a, weights=None):
    array = a
#	shape = array.shape
#	array = scipy.stats.yeojohnson(array.flatten())[0].reshape(shape)
    array -= np.average(array, weights=weights)
    array /= weightedAvgContrast(array, weights)
    return array


def scaleAndCropMiddle(openCVimg, length):
    height, width, channels = openCVimg.shape
    va = height / width
    ha = width / height
    if va > ha:
        dim = (length, int(va * length))
    if ha > va:
        dim = (int(ha * length), length)
    else:
        dim = (length, length)
    resized = cv2.resize(openCVimg, dim, cv2.INTER_AREA)
    height, width, channels = resized.shape
    y = int((height - length) / 2)
    x = int((width - length) / 2)
    return resized[y:y+length, x:x+length]


def weightedMedian(data, weights):
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median


def adjustExposure(imgLinear, w):
    return imgLinear / (np.average(imgLinear, weights=w) + weightedMedian(imgLinear.flatten(), weights=w.flatten())) * (0.5 ** 2.2)


def preprocess(imagesAsArrays, size, supersampleFactor=1):

    imageCount = len(imagesAsArrays)
    processed = []

    for i, im in enumerate(imagesAsArrays):

        img = scaleAndCropMiddle(im, size * supersampleFactor)
        B, G, R = cv2.split(img)
        B, G, R = [np.array(channel).astype(np.float64) for channel in (R, G, B)]
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(YCrCb)
        Y, Cr, Cb = [np.array(channel).astype(np.float64) for channel in (Y, Cr, Cb)]
        Y /= 255
        Cr /= 255
        Cb /= 255

        Y = gammaToLinear(Y)

        #  Y = np.clip(cv2.addWeighted(Y, 1, cv2.GaussianBlur(Y, (0, 0), size / 10), -0.5, 0.5), 0, None)  # Ben Graham's preprocessing method

        #  plt.imshow(Y, cmap='gray', vmin=0, vmax=1)
        #  plt.show()

        lumaWeights = np.copy(Y)
        blackThreshold = 0.01

        for row, rowData in enumerate(Y):
            for dot, dotValue in enumerate(rowData):
                lumaWeights[row][dot] = sigmoid((dotValue - blackThreshold) / blackThreshold**2)

        Y = adjustExposure(Y, lumaWeights)
        Y = linearToGamma(Y)

        chromaWeights = np.copy(Y)
        blackThreshold = 0.1

        for row, rowData in enumerate(Y):
            for dot, dotValue in enumerate(rowData):
                lumaWeights[row][dot] = sigmoid((dotValue - blackThreshold) / blackThreshold**2)
                chromaWeights[row][dot] = Y[row][dot] #max(0, math.log(Y[row][dot] + (1 / 255), 255) + 1)

        Y = gammaToLinear(Y)

        Y = normalize(Y, lumaWeights)
        Cb = normalize(Cb, chromaWeights)
        Cr = normalize(Cr, chromaWeights)
        Y *= lumaWeights
        Cb *= chromaWeights
        Cr *= chromaWeights

        Y = sigmoid(Y)
        Cr = sigmoid(Cr)
        Cb = sigmoid(Cb)

        R /= 255
        G /= 255
        B /= 255
        R = gammaToLinear(R)
        G = gammaToLinear(G)
        B = gammaToLinear(B)
        R = adjustExposure(R, lumaWeights)
        G = adjustExposure(G, lumaWeights)
        B = adjustExposure(B, lumaWeights)
        R = linearToGamma(R)
        G = linearToGamma(G)
        B = linearToGamma(B)
        R = np.tanh(R)
        G = np.tanh(G)
        B = np.tanh(B)

        """
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0][0].imshow(cv2.cvtColor((cv2.merge([Y, Cr, Cb]) * 255).astype(np.uint8), cv2.COLOR_YCR_CB2RGB), vmin=0, vmax=1)
        axs[1][0].imshow((cv2.merge([R, G, B]) * 255).astype(np.uint8), vmin=0, vmax=1)
        axs[0][1].imshow(Cb, cmap='plasma', vmin=0, vmax=1)
        axs[1][1].imshow(Cr, cmap='plasma', vmin=0, vmax=1)
        plt.show()
        """

        processed.append([
            cv2.resize((cv2.merge([Cr, Cb, Y]) * 255).astype(np.uint8), (size, size), cv2.INTER_AREA)
        ,	cv2.resize((cv2.merge([B, G, R]) * 255).astype(np.uint8), (size, size), cv2.INTER_AREA)
        ,	cv2.resize(cv2.cvtColor((cv2.merge([Y, Cr, Cb]) * 255).astype(np.uint8), cv2.COLOR_YCR_CB2BGR), (size, size), cv2.INTER_AREA)
        ])

        print((i + 1) / imageCount)

    return processed


def export(processedImages, rootPath, groupNames, colorEncodingNames=('YCbCr', 'RGB1', 'RGB2'), extension='png'):
    workload = len(processedImages)
    for index, imageSet in enumerate(processedImages):
        newFolder = rootPath + groupNames[index]
        os.mkdir(newFolder)
        for colorTypeIndex, singleImage in enumerate(imageSet):
            cv2.imwrite(newFolder + colorEncodingNames[colorTypeIndex] + '.' + extension, singleImage)

        print(f'Exported {index + 1} / {workload}')


def combineSeparateImageChannels(dataset):
    return np.array([cv2.merge(imageColorChannelGroup) for imageColorChannelGroup in dataset])


def engineerInputData(streamingOn, sourceFiles, referenceIDs, exportPath=False, dim=256, AA=2):
    output = []

    if streamingOn:
        for sourceImagePath in sourceFiles:
            gc.collect()
            imgSingleton = [cv2.imread(sourceImagePath)]
            imgSingleton = preprocess(imgSingleton, dim, supersampleFactor=AA)
            export(imgSingleton, exportPath, referenceIDs) if exportPath else None
            output.append(imgSingleton[0])

    else:
        megabatch = [cv2.imread(sourceImagePath) for sourceImagePath in sourceFiles]
        megabatch = preprocess(megabatch, dim, supersampleFactor=AA)
        export(megabatch, exportPath, referenceIDs) if exportPath else None
        output = megabatch
        gc.collect()

    return combineSeparateImageChannels(output)


def augmentTrainSet(trainSet, labels):
    assert len(trainSet) == len(labels)
    augmentation = []
    correspondingLabels = []
    for i, image in enumerate(list(trainSet)):
        gc.collect()
        perms = [
            np.rot90(image, k=1)
        ,   np.rot90(image, k=2)
        ,   np.rot90(image, k=3)
        ]
        augmentation.extend(perms)
        correspondingLabels.extend([labels[i]] * len(perms))
    print(np.array(labels).shape)
    print(np.array(correspondingLabels).shape)
    print(np.array(trainSet).shape)
    print(np.array(augmentation).shape)
    return np.concatenate((trainSet, augmentation)), np.concatenate((labels, correspondingLabels))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == '__main__':

    trainCSV = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

    pathStruct = '../input/9channel-engineered-training-images-256256/processed train images/Processed Train Images/{}/{}.png'
    allTrainImgData = [combineSeparateImageChannels([[cv2.imread(pathStruct.format(imageID, colorSpace)) for colorSpace in ('RGB1', 'RGB2', 'YCbCr')]])[0] for imageID in trainCSV['id_code']]
    
    print('Got training data')
    
    gc.collect()

    NUM_CLASSES = 5
    diagnosis = np.array([[int(y > i) for i in range(0, NUM_CLASSES - 1)] for y in trainCSV['diagnosis']]).astype(np.float64)
    #diagnosis = np.array([[int(y == i) for i in range(0, NUM_CLASSES)] for y in trainCSV['diagnosis']]).astype(np.float64)

    X_train, X_valid, y_train, y_valid = skl.model_selection.train_test_split(allTrainImgData, diagnosis, train_size=0.01)
    
    print('Split data into train and vaildation sets')
    
    gc.collect()

    X_train, y_train = augmentTrainSet(X_train, y_train)
    print('Augmented training data')
    
    X_train, y_train = unison_shuffled_copies(X_train, y_train)
    print('Shuffled training data')

    gc.collect()


    tensor_X_train = torch.stack([torch.Tensor(i) for i in X_train])  # transform to torch tensors
    tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])
    trainDataset = utils.TensorDataset(tensor_X_train, tensor_y_train)

    tensor_X_valid = torch.stack([torch.Tensor(i) for i in X_valid])  # transform to torch tensors
    tensor_y_valid = torch.stack([torch.Tensor(i) for i in y_valid])
    validDataset = utils.TensorDataset(tensor_X_valid, tensor_y_valid)

    print('Initialized Torch tensor datasets')

    # Data loader
    train_loader = utils.DataLoader(dataset=trainDataset,
                                               batch_size=100,
                                               shuffle=True)

    valid_loader = utils.DataLoader(dataset=validDataset,
                                              batch_size=100,
                                              shuffle=False)

    print('Initialized Torch data loaders')

    # Fully connected neural network with one hidden layer
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(9, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.drop_out = nn.Dropout()
            self.fc1 = nn.Linear(64 ** 2 * 256, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 5)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            return out


    model = ConvNet().to(computeDevice)

    print('Initialized model')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.reshape(-1, 28 * 28).to(computeDevice)
            labels = labels.to(computeDevice)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Validation Set's Quadratic Weighted Kappa: ")


    del X_train, y_train, X_valid, y_valid
    gc.collect()

    print('Training and validation finished')

    testCSV = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
    X_test = engineerInputData(True, [f'../input/aptos2019-blindness-detection/test_images/{im}.png' for im in testCSV['id_code']], testCSV['id_code'])
    gc.collect()


    print('Transformed test data')


    submission = pd.DataFrame({
        'id_code' : testCSV['id_code'].tolist(),
        'diagnosis' : diagResults
    })

    ##########

    Output = submission.to_csv('submission.csv', index=False, header=True)
