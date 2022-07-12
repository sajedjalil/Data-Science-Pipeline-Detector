

import pandas as pd #For reading csv files.
import numpy as np
import matplotlib.pyplot as plt #For plotting.
from datetime import datetime
import PIL.Image as Image #For working with image files.
import os
#Importing torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset,DataLoader #For working with data.
from torch.utils.tensorboard import SummaryWriter
from torchvision import models,transforms #For pretrained models,image transformations.

"""
在kaggel运行的
加了预处理
加了summary
加了loss曲线图
加了准确率曲线图
加了混淆矩阵图，成功
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
print(device) #Prints the device we're using.

writer = SummaryWriter(comment='ResNet', filename_suffix="_1")



# path = "/kaggle/input/aptos2019-blindness-detection/"
path = "../input/aptos2019-blindness-detection/"
weights = "../input/diabeticretinopathy/DR/model/resnet50-0676ba61.pth"
# weights = "results/06-16_17-30/DR_best.pth"
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
log_dir = os.path.join(BASE_DIR,"results", time_str)
class_names = ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

train_df = pd.read_csv(f"{path}train.csv")
print(f'No.of.training_samples: {len(train_df)}')

test_df = pd.read_csv(f'{path}test.csv')
print(f'No.of.testing_samples: {len(test_df)}')

seg_ratio = 0.9
nums_data = int(len(train_df))
seg_point = int(nums_data*seg_ratio)
index_data = list(np.arange(nums_data))
random.shuffle(index_data)
data_set = []
for i in range(nums_data):
    data_set.append([train_df['id_code'][index_data[i]],train_df['diagnosis'][index_data[i]]])
train_index,valid_index = data_set[:seg_point],data_set[seg_point:]

#Histogram of label counts.
train_df.diagnosis.hist()
plt.xticks([0,1,2,3,4])
plt.grid(False)
plt.savefig(os.path.join(log_dir,'histogram_label.png'))
plt.show()
plt.close()
# As you can see,the data is imbalanced.
# So we've to calculate weights for each class,which can be used in calculating loss.

from sklearn.utils import class_weight  # For calculating weights for each class.

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3, 4]),
                                                  y=train_df['diagnosis'].values)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(class_weights)  # Prints the calculated weights for the classes.

#For getting a random image from our training set.
num = int(np.random.randint(0,len(train_df)-1,(1,))) #Picks a random number.
sample_image = (f'{path}train_images/{train_df["id_code"][num]}.png')#Image file.
sample_image = Image.open(sample_image)
plt.imshow(sample_image)
plt.axis('off')
plt.title(f'Class: {train_df["diagnosis"][num]}') #Class of the random image.
plt.show()
plt.close()



def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


class dataset(Dataset):  # Inherits from the Dataset class.
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class.

    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''

    def __init__(self, df, data_path, image_transform=None, train=True):  # Constructor.
        super(Dataset, self).__init__()  # Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)  # Returns the number of samples in the dataset.

    def __getitem__(self, index):
        image_id = self.df[index][0]
        image = Image.open(f'{self.data_path}/{image_id}.png')  # Image.
        if self.image_transform:
            image = self.image_transform(image)  # Applies transformation to the image.

        if self.train:
            label = self.df[index][1]  # Label.
            return image, label  # If train == True, return image & label.

        else:
            return image  # If train != True, return image.




class dataset_test(Dataset):  # Inherits from the Dataset class.
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class.

    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''

    def __init__(self, df, data_path, image_transform=None, train=True):  # Constructor.
        super(Dataset, self).__init__()  # Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)  # Returns the number of samples in the dataset.

    def __getitem__(self, index):
        image_id = self.df['id_code'][index]
        image = Image.open(f'{self.data_path}/{image_id}.png')  # Image.
        if self.image_transform:
            image = self.image_transform(image)  # Applies transformation to the image.

        if self.train:
            label = self.df['diagnosis'][index]  # Label.
            return image, label  # If train == True, return image & label.

        else:
            return image  # If train != True, return image.


train_transform = transforms.Compose([transforms.Resize([512,512]),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomCrop(512, padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Transformations to apply to the image.

valid_transform = transforms.Compose([transforms.Resize([512,512]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Transformations to apply to the image.

# data_set = dataset(train_df,f'{path}train_images',image_transform=train_transform)
#
#
# #Split the data_set so that valid_set contains 0.1 samples of the data_set.
# train_set,valid_set = torch.utils.data.random_split(data_set,[3302,360])

train_set = dataset(train_index,f'{path}train_images',image_transform=train_transform)
valid_set = dataset(valid_index,f'{path}train_images',image_transform=valid_transform)

train_dataloader = DataLoader(train_set,batch_size=32,shuffle=True) #DataLoader for train_set.
valid_dataloader = DataLoader(valid_set,batch_size=32,shuffle=False) #DataLoader for validation_set.


#Since we've less data, we'll use Transfer learning.
model = models.resnet50(pretrained=False) #Downloads the resnet34 model which is pretrained on Imagenet dataset.

state_dict = torch.load(weights, map_location=device)  # load checkpoint
model.load_state_dict(state_dict)

#Replace the Final layer of pretrained resnet34 with 4 new layers.
model.fc = nn.Sequential(
                         nn.Linear(2048,128),
                         nn.ReLU(inplace=True),
                         nn.Dropout(0.1),
                         nn.Linear(128,5),
                    )

# 模型
fake_img = torch.randn(1, 3, 512, 512)
writer.add_graph(model, fake_img)

from torchsummary import summary
summary(model, input_size=(3, 512, 512), device="cpu")

model = model.to(device) #Moves the model to the device.


def train(dataloader, model, loss_fn, optimizer,iter_count_train):
    '''
    train function updates the weights of the model based on the
    loss using the optimizer in order to get a lower loss.

    Args :
         dataloader: Iterator for the batches in the data_set.
         model: Given an input produces an output by multiplying the input with the model weights.
         loss_fn: Calculates the discrepancy between the label & the model's predictions.
         optimizer: Updates the model weights.

    Returns :
         Average loss per batch which is calculated by dividing the losses for all the batches
         with the number of batches.
    '''

    model.train()  # Sets the model for training.

    total = 0
    correct = 0
    running_loss = 0
    conf_mat = np.zeros((5, 5))

    for batch, (x, y) in enumerate(dataloader):  # Iterates through the batches.
        iter_count_train += 1
        output = model(x.to(device))  # model's predictions.
        loss = loss_fn(output, y.to(device))  # loss calculation.

        running_loss += loss.item()

        total += y.size(0)
        predictions = output.argmax(
            dim=1).cpu().detach()  # Index for the highest score for all the samples in the batch.
        correct += (
                    predictions == y.cpu().detach()).sum().item()  # No.of.cases where model's predictions are equal to the label.

        # 混淆矩阵
        for j in range(len(y)):
            cate_i = y[j].cpu().numpy()
            pre_i = predictions[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1

        # 记录数据，保存于event file
        writer.add_scalars("Loss", {"Train": loss.item()}, iter_count_train)
        writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count_train)

        optimizer.zero_grad()  # Gradient values are set to zero.
        loss.backward()  # Calculates the gradients.
        optimizer.step()  # Updates the model weights.

    avg_loss = running_loss / len(dataloader)  # Average loss for a single batch

    print(f'\nTraining Loss = {avg_loss:.6f}', end='\t')
    print(f'Accuracy on Training set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    return avg_loss,100 * (correct / total),conf_mat


def validate(dataloader, model, loss_fn,iter_count_valid):
    '''
    validate function calculates the average loss per batch and the accuracy of the model's predictions.

    Args :
         dataloader: Iterator for the batches in the data_set.
         model: Given an input produces an output by multiplying the input with the model weights.
         loss_fn: Calculates the discrepancy between the label & the model's predictions.

    Returns :
         Average loss per batch which is calculated by dividing the losses for all the batches
         with the number of batches.
    '''

    model.eval()  # Sets the model for evaluation.

    total = 0
    correct = 0
    running_loss = 0
    conf_mat = np.zeros((5, 5))
    with torch.no_grad():  # No need to calculate the gradients.

        for x, y in dataloader:
            iter_count_valid += 1
            output = model(x.to(device))  # model's output.
            loss = loss_fn(output, y.to(device)).item()  # loss calculation.
            running_loss += loss

            total += y.size(0)
            predictions = output.argmax(dim=1).cpu().detach()
            correct += (predictions == y.cpu().detach()).sum().item()

            # 混淆矩阵
            for j in range(len(y)):
                cate_i = y[j].cpu().numpy()
                pre_i = predictions[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Valid": loss}, iter_count_valid)
            writer.add_scalars("Accuracy", {"Valid": correct / total}, iter_count_valid)

    avg_loss = running_loss / len(dataloader)  # Average loss per batch.

    print(f'\nValidation Loss = {avg_loss:.6f}', end='\t')
    print(f'Accuracy on Validation set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    return avg_loss,100 * (correct / total),conf_mat


def optimize(train_dataloader, valid_dataloader, model, loss_fn, optimizer, nb_epochs):
    '''
    optimize function calls the train & validate functions for (nb_epochs) times.

    Args :
        train_dataloader: DataLoader for the train_set.
        valid_dataloader: DataLoader for the valid_set.
        model: Given an input produces an output by multiplying the input with the model weights.
        loss_fn: Calculates the discrepancy between the label & the model's predictions.
        optimizer: Updates the model weights.
        nb_epochs: Number of epochs.

    Returns :
        Tuple of lists containing losses for all the epochs.
    '''
    # Lists to store losses for all the epochs.
    train_losses = []
    valid_losses = []
    train_acces = []
    valid_acces = []
    best_acc = 0
    valid_acc = 0
    iter_count_train = 0
    iter_count_valid = 0
    for epoch in range(nb_epochs):
        print(f'\nEpoch {epoch + 1}/{nb_epochs}')
        print('-------------------------------')
        train_loss,train_acc,mat_train = train(train_dataloader, model, loss_fn, optimizer,iter_count_train)  # Calls the train function.
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        valid_loss,valid_acc,mat_valid = validate(valid_dataloader, model, loss_fn,iter_count_valid)  # Calls the validate function.
        valid_losses.append(valid_loss)
        valid_acces.append(valid_acc)

        # 每个epoch，记录梯度，权值
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == nb_epochs - 1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == nb_epochs - 1)
        if epoch > (nb_epochs / 2) and best_acc < valid_acc:
            # if epoch % save_freq==0 and best_acc < acc_valid:

            best_acc = valid_acc


            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "DR_best.pth")
            torch.save(checkpoint, path_checkpoint)

    print('\nTraining has completed!')

    return train_losses, valid_losses,train_acces,valid_acces,valid_acc


loss_fn   = nn.CrossEntropyLoss(weight=class_weights) #CrossEntropyLoss with class_weights.
optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
nb_epochs = 44
#Call the optimize function.
train_losses, valid_losses,train_acces,valid_acces,valid_acc = optimize(train_dataloader,valid_dataloader,model,loss_fn,optimizer,nb_epochs)


#Plot the graph of train_losses & valid_losses against nb_epochs.
epochs = range(nb_epochs)
plt.plot(epochs, train_losses, 'g', label='Training loss')
plt.plot(epochs, valid_losses, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(log_dir,'Training and Validation loss.png'))
plt.show()
plt.close()

epochs = range(nb_epochs)
plt.plot(epochs, train_acces, 'g', label='Training Accuracy')
plt.plot(epochs, valid_acces, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(log_dir,'Training and Validation Accuracy.png'))
plt.show()
plt.close()

#Save the model

checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": nb_epochs,
                          "best_acc": valid_acc}
torch.save(checkpoint,log_dir+'DR_last.pth')

writer.close()
test_set = dataset_test(test_df,f'{path}test_images',image_transform = valid_transform,train = False )

test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False) #DataLoader for test_set.


def test(dataloader, model):
    '''
    test function predicts the labels given an image batches.

    Args :
         dataloader: DataLoader for the test_set.
         model: Given an input produces an output by multiplying the input with the model weights.

    Returns :
         List of predicted labels.
    '''

    model.eval()  # Sets the model for evaluation.

    labels = []  # List to store the predicted labels.

    with torch.no_grad():
        for batch, x in enumerate(dataloader):
            output = model(x.to(device))

            predictions = output.argmax(dim=1).cpu().detach().tolist()  # Predicted labels for an image batch.
            labels.extend(predictions)

    print('Testing has completed')

    return labels


labels = test(test_dataloader,model) #Calls the test function.

labels = np.array(labels)
submission_df = pd.DataFrame({'id_code':test_df['id_code'],'diagnosis':labels}) #DataFrame with id_code's and predicted labels.

submission_df.to_csv('submission.csv',index=False) #csv file for submission.
print("Your submission was successfully saved!")