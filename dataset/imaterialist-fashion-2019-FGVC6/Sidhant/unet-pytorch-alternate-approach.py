""" This is an attempt of using Unet for segmentation of the categories. Performance is poor, I think it's 
because convergence is very slow. Bottlneck slowing down training is rle conversions in each batch.
The input into the Unet is 3xNxM image and the output is a 46xMxN array. 46 are the number of classes. 
Essentially it is a binary classifcation  problem but over a 46 channel output. The reason I picked such an 
approach is that it allows for overlapping classification segments. - eg. masks of tshirt overlaps with pockets, 
sleeve etc. The class imbalance caused the network to only output 0's, so I used a pos_weights in the 
loss function. After doing this, the network no longer predicts all 0's as it did earlier but still performance is 
poor. Feel free to discuss about the approach, why it doesn't work etc.

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pathlib import Path
import json
import os
import cv2
from fastai import *
from fastai.vision import * 
from fastai.datasets import *
from skimage import data,io, transform
from torch.utils.data import * 
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from optparse import OptionParser
import numpy as np
import torch.backends.cudnn as cudnn
from torch import optim
import time 
from torch.autograd import Function, Variable


# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.
path='../input/imaterialist-fashion-2019-FGVC6/'
#a = '../input/train/' +'10320336070_91.jpg'
df = pd.read_csv(path + 'train.csv')
#b=np.array(cv2.imread(a))
#b.shape
df.head(2)

def create_label(x):
    ids = x.split("_")
    return int(ids[0])
    
df['label']=df['ClassId'].apply(create_label)
# remove files
df = df[df['ImageId'] != '2ab8c02ce17612733ddee218b4ce1fd1.jpg']
df = df[df['ImageId'] != 'f4d6e71fbffc3e891e5009fef2c8bf6b.jpg']
df = df[df['ImageId'] != '00c344097e4bb8031790989c21ae6fe7.jpg']
df = df[df['ImageId'] != '010b56cf612e31e9b2f1321dbb655fc2.jpg']
df = df[df['ImageId'] != '01643fd559e2461301fcf5e6feb26b87.jpg']
df = df[df['ImageId'] != '02426e0dd340abf3bac9a14ec8674d6a.jpg']
file_list=['00c344097e4bb8031790989c21ae6fe7.jpg', '010b56cf612e31e9b2f1321dbb655fc2.jpg', '01643fd559e2461301fcf5e6feb26b87.jpg', '02426e0dd340abf3bac9a14ec8674d6a.jpg', '043d9048aa5c4a0a62ff7e484961e2f9.jpg', '0612d2b6a7a43b0875fdb3ecc1535e81.jpg', '0630dcf8c8dc83292c7e1bbbf5049322.jpg', '06ce98624c1d695f9c6eeaa8151bb049.jpg', '06f99b8fc0db3c289a5f85af68a59d6d.jpg', '07c91e24128080597fb8c7a3b7676261.jpg', '07d7a7ea62a3b2fe8ecfe1e9fcfc6034.jpg', '0bed403b9f373435ff4a922655c1c254.jpg', '0bf3f45a408a7edd93ac420d9bba98ee.jpg', '0e8317503f7f0c432ab62c5e7e11b975.jpg', '0edd937827a774360d932e961da6490a.jpg', '0f904e377f7d69ae2d3d467d6c0a4d89.jpg', '13621e8904b3cefdfec28275c90124f5.jpg', '14495fa1389e145a61307abf0e9f3211.jpg', '168e021ef9322b0741065b8f5645b234.jpg', '181d3546d933e9f02c75b1ebd9456d70.jpg', '194a218e260601798bb657805050d7e1.jpg', '1a415982b6177b0d30fbceadeaab3c07.jpg', '1b54ff04705d6da0e3dc06a496a0e165.jpg', '1c23b0472587d84d82ff17366be5ccb8.jpg', '1c48b6c1069d148c13fa04dd6986b064.jpg', '1d79d8c8fe947e834b5d4f6b3819ed73.jpg', '25a4c80360bf787f945ec6a253b78ef9.jpg', '28b4d38bf9dd2b925a5aeaa6ccbb3696.jpg', '290d0b780ae364cbc50acc62c44928cb.jpg', '2ba44450278ee435bf25e37912d008dc.jpg', '2ba69a4e9de0cdf1caa419b8c2338461.jpg', '2be0d3aaf6b0302c5d6bb839337423d9.jpg', '2db8150211e0497ae6d5c292bb050fc8.jpg', '2ecb189c6d816c6ddcffba959163f7cd.jpg', '307aba17bfc9a40b9f15453e1c613815.jpg', '31baef0fe1708afd43d5a4f06e7b85d6.jpg', '32bc97a2ce6c759bcecadf10d1e8c7c6.jpg', '353f3e104f0b21a40d88fa18722bacc3.jpg', '35732509497752d8265ce060fdc2b994.jpg', '373f5cfc1c53ec12a5e006ba05b275f9.jpg', '389fc5e52e8107468158e587a6d14657.jpg', '3a23c1f989e1e985e445b61630024468.jpg', '3ad916fe62d4fc6ff1b6f5a7f7b67810.jpg', '3aecc673739c40129fff8451fa574c7b.jpg', '3b0c357baa75d048a3de82f22e99f127.jpg', '3b21276d079ad7b955361db32ae39619.jpg', '3bf09e807864233aea4b7f68d2419b31.jpg', '3d8aa95253a15f64b297be8b92357e26.jpg', '3e49f7d28159d2f785dd4011506851d2.jpg', '3fec43fc4dba5825b4ee73ea255a59eb.jpg', '437fe04fbbf7bcba2abbff73365e82ad.jpg', '444071ff7ebcd773aa0be52c717b68b0.jpg', '454e5544d797bb92a8d8b4f98fd9e3ca.jpg', '45834cef5e546cc85e55e4d997302eca.jpg', '45eac4eb5dfbab3c9674cee7086b14cf.jpg', '4614ac83a00791d55db59f693c005b03.jpg', '49400fc45fb42b01eb009373311797e8.jpg', '4a41639fa7ca74ce4912a0c8cfe9c1bd.jpg', '4b3633b82822771b025bf3c831102f93.jpg', '4c3c57f3adb8e28fd139b72cc6e68858.jpg', '4cff61856e37febd097371dc114b5ffd.jpg', '4e0dcb3913b4eb8d48c797a0d8fba17c.jpg', '4e11531bffbd53642119f548f31853dd.jpg', '4e763310ce1b32cb75b947aa8f5f6362.jpg', '4ed300689a88f5390899f5595ff6fd9c.jpg', '4f50776e37df104d39adfdc09dd0afb6.jpg', '4f56a281939323e1dc0df74c283c1ab2.jpg', '5048006be849fab075ae6b8993b470ba.jpg', '51907caa20154ecbb5acf2106f2b84cb.jpg', '554ac5888146fa9ac4e5983e20648b63.jpg', '56ccaa8a7be3f0359ee96f144247c165.jpg', '58940147f49f38e2ad95fb58cd2a9d41.jpg', '5a08d9cc11a1615ba72e8be1e6d5b622.jpg', '5c4b35c7273024e854562ce62f80d48b.jpg', '5d189f285cbf9a1486b194981ffa51dd.jpg', '606fba83abc0aa45c45b09412478c382.jpg', '6097c1908548b52d28bfe8f6c1d994d8.jpg', '626e99262d421589537c6aa6c4813719.jpg', '6278033b375bc00406fa9be7227be597.jpg', '639b27f89cb826e03dfd56493ed8502d.jpg', '63dd8124bd6958cd0778414f3cecbeca.jpg', '69ee86f19ea0819cec434976b83eed02.jpg', '6a1b81a712a77ecce1b10ddb13397c6f.jpg', '6a9df31253b53353ca996ee100aae9e3.jpg', '6ab8540b7a1043ca1a6b8d5d4c9b3e2b.jpg', '6dc17d9effb634502634334a9be22691.jpg', '6ded716c6314544d9bfed0b8aa867287.jpg', '6e39b130ab342ab1acb1bc3669394685.jpg', '6e41af61cfd1bc2a585e755562d6ee87.jpg', '72980eef8eef838cde3ed3138553f15c.jpg', '72d014fe1fd96b31c46c7f53e7a083a7.jpg', '72f444f2a5e7c9e7845cc59ae053274c.jpg', '73972461600162bbf4de2e0524d0b7c8.jpg', '75f83e3eec8ab3bcc9e24259d66dbbfb.jpg', '7629537133cc78169233295aa7313109.jpg', '76b9efc4ac56a841ccd30a740c4e1d33.jpg', '78f5ece22b866e48849b43f207648379.jpg', '7c9f38457cbc8383f8493e10e32a8560.jpg', '7d956f0919f2eab63e481e9321ce2028.jpg', '7eefdcf40f3ae9f581556328675cf23b.jpg', '7f1f10d13863b0eae9b90d1ade1ca6b3.jpg', '8319f1f2d65bfa0e8f0264cb5e25175b.jpg', '85973934cc8179397219cc471f1ef553.jpg', '85b784018b93aa1fd420d7a9346ce5a2.jpg', '85f9da5d7595227361f5e056d8413e34.jpg', '864deb800f07cc0a1b2147e76aba1a88.jpg', '86d5d1652aa75cae4bfcffbdc0f3b079.jpg', '8ab58cce71e25db9223baea0451271b5.jpg', '8bb29b5a3643259cb1cfbe6971d0836f.jpg', '8f2c6a6526e4abd6df89f9dd765ca229.jpg', '90c76ec621bfe5e146f6bf7867738519.jpg', '90f703cb6177ad22714f668175b52335.jpg', '91cf59be512913adef9eee2da1b0c269.jpg', '9310d80ba15517563fcf0be4e33a04e9.jpg', '94b357abf2a45614ce99fe0ff30ad84d.jpg', '960f703e599cfb761c87bf569cae5eb5.jpg', '9719923d907686b6336860f425795ded.jpg', '98d75a91c03c542169787db37f43fbf2.jpg', '98f1b69c43f3f725af6bd6c9cde2a1a1.jpg', '9a0e484947074c73a62a88cfa67d7f7b.jpg', '9b27fc68c88ba4b875b0248d554461f7.jpg', '9b541f60dae448f8f1a9c29923c2d077.jpg', '9c54ef5f0b988f11980ab5865bd81805.jpg', '9fc56a7f086ac7de8c4ca42f585e650a.jpg', 'a0e34b1aa32884889401665a7c0a1c30.jpg', 'a46cff6199db3c3721488f4699c49c73.jpg', 'a5a46fd45494837955548232de15fa2e.jpg', 'a8595e67c8f8844279448496a3cbb1d0.jpg', 'a87797c302e92b96c818f048b2ae97c0.jpg', 'aa6f2dfc6b486dd5675e5c9576bd16e9.jpg', 'ab8844a15b640f20cc2095761a694f51.jpg', 'ac7f353fd7481926e8fa9eafed609948.jpg', 'ad1c0b2ba1924b57aaca1455b177fee3.jpg', 'aefa496028070ea6650455ec5fb9fc5d.jpg', 'af211d9f824b68b739eb8212620e6010.jpg', 'b41b41b8b4925f559e3f103bc9330acd.jpg', 'b42c56e786adc1804b5f362d54d70c8d.jpg', 'b780a56a468959b0fefe3ddd0fd1d571.jpg', 'b9ccf0f2032d9ce6ce01b221bf0fbe11.jpg', 'be179362ec6e2c434d3ee5a329efaca3.jpg', 'bf399a62f828acc5a800c5ae48803dde.jpg', 'c10940b2a58cb0869fdc72c6990e652e.jpg', 'c265cbab6a72fc283a9020684b929a30.jpg', 'c40d31d5b711b675bc66773c78bf2622.jpg', 'c55be4778c06e3249befa776419394f4.jpg', 'c7dc4f9965747edbd2ab59031959ae47.jpg', 'c92b03faba0a9463bb6a74a5b5c5913f.jpg', 'cd9d6cb647805c29fde6c3f998b6d0e4.jpg', 'cda3e1e3d731c330fb9c8eefb86cd99c.jpg', 'd03a54446b39c5755fc582050aaa5e2c.jpg', 'd1aabb4dc18bb6fb8b4466f4186aab2e.jpg', 'd205af78e74c4c0f14507a1d397adfb3.jpg', 'd300a9d86a20779cad295c64b5642c6e.jpg', 'd45378eb460b4bfeec151d7222d80c22.jpg', 'd7cd2b11f7dd2e4542d46422414aa9c9.jpg', 'd7d5c6f18d5b7a0b50189e2779054c01.jpg', 'da5c45a740bf6ef6990c33e022d61720.jpg', 'dd74c855a639e7738c63eb12b649c789.jpg', 'dec6af5377787ffa5de4d8f00290b1d9.jpg', 'e0093ef481c23f3b8d7f3203c6c17dd9.jpg', 'e08880a8535f902a9ffbaa40e3fa15f1.jpg', 'e23c1598f18e94ed442e78a1de4ece52.jpg', 'e6eb5fb61bb403fc0703c7e756b6f8a6.jpg', 'e7f88ef91e050501c102f2cf9e504b10.jpg', 'e992328bebcf8036ef2875604d099372.jpg', 'eb8322ef97eff6e825e10eb6878d597d.jpg', 'ed3c389f88a1acd56d0ad6e16d9d029b.jpg', 'ed4a786260b04e174f438a97f1dec272.jpg', 'ed66525b4fa242ddb3df3675ceaec44d.jpg', 'efe649a932e4dba137453f80a5685199.jpg', 'f0fc6875cfab6fbf3b31fef14e52c62d.jpg', 'f1e8a09f995ebd45458b96f40d95ad9c.jpg', 'f246a7bddb701dfdc83cbce180997d9f.jpg', 'f3d1641902df5be81c219b3c73be5ecb.jpg', 'f5a39d5af14a9107aefcfee4a6cdd3e5.jpg', 'f69e112fb22010e615d0fb1c5660b0d2.jpg', 'f6f23192a1a4c0315779e2d3d01da952.jpg', 'f850ef92c8c07bf24da06f9058b5085d.jpg', 'f8b4e899fed1c56d21f5edbaab4edf01.jpg', 'fa4296a6ff06ec9da479e221a4a663f6.jpg', 'fbbdae3b820fc5bb2084c9cf7347b415.jpg', 'fbf4b07fffa61fa2650649ec19debf39.jpg', 'fc97fa2a6368f84336dd2e1dc3da7679.jpg', 'fcb72ae3118383fc91364d6181f5c681.jpg', 'fdd0daec3624d1d7acf8048484a7efe2.jpg', 'ff943feb34677ab7380b172f5a879061.jpg']
# removing unneccesary files (with grayscale input, model accepts 3 channel input
for word in file_list:
    df = df[df['ImageId']!=word]
    
# function to write train history     
def write_dict(train_dict):
    json2 = json.dumps(train_dict)
    f = open("train_hist.json","w")
    f.write(json2)
    f.close()

# creating the target - n_classes x 512 x 512 
def create_target(x,df, resize_to):
    x = Path(x)
    file_name = x.stem + x.suffix
    df = df[df['ImageId'] == file_name]
    masks=[]
    ids=[]
    # resize mask to 
    new_h, new_w = resize_to
    # target array 
    target = np.zeros((46,new_h,new_w),dtype=np.uint8)
    for i in range(len(df)):
    
        rle_mask = df.iloc[i]['EncodedPixels']
        height= df.iloc[i]['Height']
        width= df.iloc[i]['Width']
        label=df.iloc[i]['label']
        # open rle mask 
        a = vision.image.open_mask_rle(rle_mask, shape=(height, width)).resize((1,new_h,new_w))
        # removing extra dimension from mask 
        mask = np.squeeze(np.array(a.data, dtype=np.uint8))
        # creating the target. masks of class #n occupies nth channel 
        target[label,:, :] = np.maximum(target[label,:, :], mask)
    
    return target

# Creating pytorch dataset 
class SegmentationDataset(Dataset):
    
    def __init__(self, df, root_dir, resize_shape):
        self.df = df
        self.image_names = df['ImageId'].unique()
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        #self.transform = transform
    
    def __len__(self):
        return len(self.image_names)
            
    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir,self.image_names[idx])
        # loading training image 
        img = io.imread(img_path)
        shape = img.shape
        try:
            # switching axis such that shape is (n_channels, w,h)
            img=np.rollaxis(img, 2, 0) 
        except: 
            print('Img size is wrong, it is', shape)
            print(img_path)
            sys.exit(1)
            
        
        h,w = self.resize_shape
        # resize image to 
        img = transform.resize(img, (3,h,w))
        # creating target mask
        segmented_image = create_target(img_path, self.df, self.resize_shape)

        return [img,segmented_image]
        
        
path_train = os.path.join(path,'train')


# defining blocks for unet -https://github.com/milesial/Pytorch-UNet
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
# full assembly of the sub-parts to form the complete net - https://github.com/milesial/Pytorch-UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        #print('input into dice', input.view(-1).size())
        #print('target into dice', target.view(-1).size())
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# evalutae 
def eval_train(net, batch, mask_preds, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    
    imgs = batch[0]
    true_masks = batch[1]
    #print('input image to eval_train size', imgs.size())
    #print('true mask input to eval_train size', true_masks.size())

    #img = torch.from_numpy(img).unsqueeze(0)
    #true_mask = torch.from_numpy(true_mask).unsqueeze(0)

    if gpu:
        imgs = imgs.cuda()
        true_masks = true_masks.cuda()

    # converting predictions to 1 or 0 
    mask_preds = (mask_preds > 0.5).float()
    
    for i in range(len(batch)):
        tot += dice_coeff(mask_preds[i,:,:,:], true_masks[i,:,:,:]).item()
        return tot / (i + 1)
        
# actually calcuates recall     
def true_positive_rate(target, pred_mask):
    target=np.array(target)
    # converting probs to 0 and 1 
    pred_mask = np.array((pred_mask > 0.5).float())
    # gets all the number of true positive predictions 
    nb_tp = np.sum(np.logical_and(pred_mask == 1, target ==1))
    nb_tn = np.sum(np.logical_and(pred_mask == 1, target == 0))
    recall = nb_tp/(nb_tn + nb_tp)
    
    nb_p = np.sum(target == 1)
    return nb_tp/(nb_tn + nb_tp)
     
        
def train_net(net,
              epochs=5,
              batch_size=2,
              lr=0.025,
              val_percent=0.05,
              save_cp=False,
              gpu=False,
              img_scale=0.5, check_dir=None):

    dir_checkpoint = 'checkpoints/'


    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_dataset),
               len([0]), str(save_cp), str(gpu)))

    N_train = len(train_dataset)
    N_train_batches = len(train_dataloader)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    """change image height if needed"""
    
    pos_weight = 600*torch.ones((batch_size, 46,512,512)).view(-1).cuda()
    

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses = []
    times = []
    train_dices = []
    
    #hist_path = os.path.join(check_dir,'train_hist.json')
    #print(hist_path)
    #train_history = json.load(open(hist_path))
    # the step of the checkpoint 
    #last_step = train_history['train_step'][-1]
    #print(last_step)
    train_history = {'loss': [], 'train_acc': [], 'train_dice': [], 'val_acc': [], 'train_step':[], 'val_dice': [], 'val_loss': [], 'val_step': []}
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        
        epoch_loss = 0

        for i, batch in enumerate(train_dataloader):
            #if epoch + 1 == 1:
            #    if i < last_step:
            #        continue 
            
            start=time.time()
            imgs = batch[0].type(torch.FloatTensor) 
            target_masks = batch[1].type(torch.FloatTensor)
            del batch
            
            if gpu:
                imgs = imgs.cuda()
                target_masks = target_masks.cuda()
            
            # predicting
            masks_pred = net(imgs)
            
            # flatten 
            masks_probs_flat = masks_pred.view(-1)
            target_masks_flat = target_masks.view(-1)
            # calculate loss 
            #print(target_masks.size())
            loss = criterion(masks_probs_flat, target_masks_flat)
            del masks_probs_flat
            del target_masks_flat
            # loss     
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            masks_pred = torch.sigmoid(masks_pred)
            end = time.time()
            times.append(end-start)
            # every 10 steps, eval 
            if i % 10 == 0:
                # train dice score
                train_dice = eval_train(net, (imgs,target_masks), masks_pred, gpu)
                del imgs
                # true postive rate
                
                acc=true_positive_rate(target_masks.type(torch.ByteTensor), (masks_pred > 0.5).byte())
                del target_masks
                del masks_pred
                # appending loss 
                #print(type(loss.item()))
                train_history['loss'].append(loss.item())
                train_history['train_acc'].append(acc)
                train_history['train_dice'].append(train_dice)
                train_history['train_step'].append(i + epoch*N_train_batches)
               
                #print stats 
                print('{0:.4f} --- loss: {1:.6f} -- train acc: {2:.3f} -- train dice: {4:.3f} in {3:.3f} seconds'.format(
                                        i * batch_size / N_train, loss.item(), acc,  np.mean(times), train_dice))
                # reset timer 
                times = []
                
                
                # every 100 steps savemodel 
                if i % 500 == 0:
                    if save_cp:
                        torch.save(net.state_dict(),'CP{}.pth'.format(epoch + 1))
                        print('Checkpoint {} saved !'.format(epoch + 1))
                        
                    print('\nEval..')
                    # eval
                    net.eval()
                    val_losses =[] 
                    val_dices = []
                    val_accs = []
                    
                    for k, batch in enumerate(val_dataloader):
                        # evaluate only on 10 batches 
                        if k == 12:
                            break
                        
                        val_imgs = batch[0].type(torch.FloatTensor) 
                        target_masks = batch[1].type(torch.FloatTensor)

                        if gpu:
                            val_imgs = val_imgs.cuda()
                            target_masks = target_masks.cuda()
                        # predict 
                        masks_pred = net(val_imgs)
                        # flatten
                        masks_probs_flat = masks_pred.view(-1)
                        target_masks_flat = target_masks.view(-1)
                        #print(masks_probs_flat.size())
                        #print(target_masks_flat.size())
                        # calculate loss 
                        val_loss = criterion(masks_probs_flat, target_masks_flat).item()
                        val_losses.append(val_loss)
                        
                        del masks_probs_flat
                        del target_masks_flat
                        #eval 
                        val_dice = eval_train(net, (val_imgs, target_masks), masks_pred,gpu=gpu)
                        val_dices.append(val_dice)
                        val_accs.append(true_positive_rate(target_masks, masks_pred))
                        del val_imgs
                        del target_masks
                    
                    train_history['val_dice'].append(np.mean(val_dices))   
                    train_history['val_acc'].append(np.mean(val_accs))
                    train_history['val_loss'].append(np.mean(val_loss))
                    train_history['val_step'].append(i)
                    
                    write_dict(train_history)
                    print('val loss: {0:.5f}, val acc: {1:0.5}, val dice: {2:0.5}'.format(np.mean(val_loss), np.mean(val_accs), np.mean(val_dices)))
                    
                net.train()
            
        print('Epoch finished ! Loss: {}'.format(epoch_loss))

# eval set 
val_sta = 73502
val_end = 79507
# train set 
a = df.iloc[:val_sta]
b = df.iloc[val_end:]
train_df = a.append(b)

train_dataset=SegmentationDataset(train_df, path_train, (512,512))
val_dataset=SegmentationDataset(df[val_sta:val_end],path_train, (512,512))
del train_df
del df        
        
bs=4
train_dataloader= DeviceDataLoader.create(train_dataset, bs=bs, num_workers=2, shuffle=True)
val_dataloader = DeviceDataLoader.create(val_dataset, bs=bs, shuffle=True)
# load checkpoints 
#check_dir = '../input/pytorch-unet2'
#check = check_dir + '/CP1.pth'
# unet 
net=UNet(n_channels=3, n_classes=46)

#net.load_state_dict(torch.load(check))
net.cuda()
net.train()
""" uncomment train_net below to train """
#train_net(net=net, save_cp=True, batch_size=bs, gpu=True, check_dir=None)