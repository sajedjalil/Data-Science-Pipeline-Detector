

#Train a model from scratch in PyTorch and run evaluation
import sys
import io
import os
import re
import subprocess
import math
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import OrderedDict
from collections import Counter
import random
import Levenshtein
import torch
import glob
import re
import torch.nn as nn
from PIL import Image

def read_characters(lines,**kwargs):
    vocab = dict()
    num_words = 0
    for line in lines:
        line = line.strip()
        for char in line:
            if char not in vocab:
                vocab[char] = num_words
                num_words += 1
    for word in [kwargs['start_token'],kwargs['end_token'],kwargs['unk_token']]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab

def load_batch(data,**kwargs):
    files = ['../input/bms-molecular-translation/train/{0:s}/{1:s}/{2:s}/{3:s}.png'.format(f[0],f[1],f[2],f) for f in data['image_id'].values]
    image_data = torch.from_numpy(load_images(files,**kwargs))
    text = data[args['targets']].values
    sequence_data = torch.from_numpy(load_characters(text,**kwargs))
    return image_data.float().to(kwargs['device']), sequence_data.long().to(kwargs['device'])

def load_sample_batch(data,**kwargs):
    files = ['../input/bms-molecular-translation/'+args['mode']+'/{0:s}/{1:s}/{2:s}/{3:s}.png'.format(f[0],f[1],f[2],f) for f in data['image_id'].values]
    image_data = torch.from_numpy(load_images(files,**kwargs))
    return image_data.float().to(kwargs['device'])

def load_images(files,**kwargs):
    output = 255*np.ones((len(files),1,args['ysize'],args['xsize']),dtype='uint8')
    maxx = kwargs['xsize']
    maxy = kwargs['ysize']
    maxratio = maxx / maxy
    for i,file in enumerate(files):
        img = Image.open(file).convert('L')
        x, y = img.size
        ratio = x/y
        if ratio >= maxratio:
            img = img.resize((maxx,int(y*maxx/x)))
        else:
            img = img.resize((int(x*maxy/y),maxy))
        x, y = img.size
        output[i,0,int((maxy-y)/2):int((maxy-y)/2)+y,int((maxx-x)/2):int((maxx-x)/2)+x] = np.uint8(np.asarray(img))
    return output

def load_characters(text,**kwargs):
    output = len(kwargs['vocab'])*np.ones((max([len(t) for t in text])+2,len(text)),dtype='int')
    utoken_value = kwargs['vocab'][kwargs['unk_token']]
    for i,t in enumerate(text):
        words = [kwargs['start_token']] + list(t.strip()) + [kwargs['end_token']]
        for jdx,word in enumerate(words):
            output[jdx,i] = kwargs['vocab'].get(word,utoken_value)
    return output

#Train the model for an epoch
def train_model(train_data,encoder,decoder,optimizer,criterion,**kwargs):
    trainlen = len(train_data)
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        encoder = encoder.train()
        decoder = decoder.train()
        for b in range(nbatches):
            #Obtain batch
            X, Y = load_batch(train_data.iloc[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])],**kwargs)
            Yt = Y[1:Y.size(0)]
            Y = Y[0:Y.size(0)-1]
            mask = torch.clamp(len(kwargs['vocab'])-Yt,max=1)
            seq_length = torch.sum(mask,dim=0).cpu()
            ordered_seq_length, dec_index = seq_length.sort(descending=True)
            X = X[dec_index]
            Y = Y[:,dec_index]
            Yt = Yt[:,dec_index]
            mask = mask[:,dec_index]
            #Propagate
            embeddings = encoder(X)
            decoder.init_hidden(hidden=torch.unsqueeze(embeddings,0))
            posteriors = decoder(Y,ordered_seq_length)
            #Flatten outputs and targets
            flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
            flat_mask = mask.view(mask.size(0)*mask.size(1))
            flat_Y = Yt.view(Yt.size(0)*Yt.size(1))
            flat_Y = flat_Y * flat_mask
            #Compute non reduced loss
            flat_loss = criterion(flat_posteriors,flat_Y)
            flat_loss = flat_loss*flat_mask.float()
            #Get the averaged/reduced loss using the mask
            mean_loss = torch.sum(flat_loss)/torch.sum(flat_mask)
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
            #Track loss
            if total_backs == 100:
                total_loss = total_loss*0.99+mean_loss.detach().cpu().numpy()
            else:
                total_loss += mean_loss.detach().cpu().numpy()
                total_backs += 1
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()
    return total_loss/(total_backs+1)

def sample_model(data,encoder,decoder,**kwargs):
    num_seq = len(data)
    nbatches = math.ceil(num_seq/kwargs['batch_size'])
    output_idx = np.zeros((args['sample_length'],num_seq))
    with torch.no_grad():
        encoder = encoder.eval()
        decoder = decoder.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = load_sample_batch(data.iloc[b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])],**kwargs)
                embeddings = encoder(X)
                decoder.init_hidden(hidden=torch.unsqueeze(embeddings,0))
                X = kwargs['vocab'][kwargs['start_token']]*torch.ones((1,X.shape[0]),dtype=torch.long).to(kwargs['device'])
                output_idx[0,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])] = X.cpu().numpy()
                seq_length = torch.ones((X.shape[1]),dtype=torch.long)
                for i in range(1,args['sample_length']):
                    #Forward the last symbol
                    posteriors = decoder(X,seq_length)
                    #posteriors[:,:,args['vocab'][args['unk_token']]] = -100.0
                    X = torch.argmax(posteriors,dim=2,keepdim=False)
                    output_idx[i,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])] = X.cpu().numpy()
                    if args['early_stopping']:
                        stop_criterion = output_idx[:,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])] == kwargs['vocab'][kwargs['end_token']]
                        if np.sum(np.max(stop_criterion,axis=0)) == X.shape[1]:
                            #output_idx = output_idx[0:i+1,:]
                            break
                pbar.set_description(f'Doing inference')
                pbar.update()
    return output_idx

def indexes_to_characters(input_idx,**kwargs):
    output = ''
    for i in input_idx:
        if kwargs['inv_vocab'][i] == kwargs['end_token']:
            break
        elif kwargs['inv_vocab'][i] not in [kwargs['start_token'],kwargs['unk_token']]:
            output += args['inv_vocab'][i]
    return output

class CNNEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CNNEncoder, self).__init__()
        #Arguments
        self.xsize = kwargs['xsize']
        self.ysize = kwargs['ysize']
        self.num_blocks = kwargs['num_blocks']
        self.channels = kwargs['channels']
        self.input_channels = (kwargs['input_channels'] if 'input_channels' in kwargs else 1)
        self.reduce_size = (kwargs['reduce_size'] if 'reduce_size' in kwargs else False)
        self.dropout = kwargs['dropout']
        self.embedding_size = kwargs['embedding_size']
        self.vocab = kwargs['vocab']
        self.num_classes = len(self.vocab)

        #Gaussian normalise the input
        tmp_xsize = self.xsize
        tmp_ysize = self.ysize
        #Convolutional blocks
        for i in range(1,self.num_blocks+1):
            if self.reduce_size or i ==1:
                setattr(self,'convblock'+str(i),ConvBlock((self.channels if i>1 else self.input_channels), self.channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, dropout=self.dropout, residual=False))
                tmp_xsize = int(tmp_xsize/2)
                tmp_ysize = int(tmp_ysize/2)
            setattr(self,'convblock'+str(i)+'residual',ConvBlock(self.channels, self.channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=int(self.channels/2), dropout=self.dropout, residual=True))
            
        #Flatten the output
        self.flatten = nn.Flatten()
        #Reduce to embedding layer
        self.linear = nn.Linear(int(tmp_xsize*tmp_ysize*self.channels), self.embedding_size, bias=False)
        #Batch normalise
        self.batchnorm = nn.BatchNorm1d(self.embedding_size, momentum=0.9)

    def forward(self, x):
        out = (x - 128.0) / 128.0
        for i in range(1,self.num_blocks+1):
            if self.reduce_size or i==1:
                conv = getattr(self,'convblock'+str(i))
                out = conv(out)
            conv = getattr(self,'convblock'+str(i)+'residual')
            out = conv(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.batchnorm(out)
        return out

#Residual convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dropout=0.2, residual=False):
        super(ConvBlock, self).__init__()
        #2D convolution
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        #Batch normalisation
        self.bn = nn.BatchNorm2d(out_c, momentum=0.9)
        #Activation
        self.prelu = nn.PReLU(out_c)
        #Dropout
        self.dropout = nn.Dropout3d(p=dropout)
        self.residual = residual
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.dropout(out)
        #Residual connection
        if self.residual:
            out = out + x
        return out
    
class LSTMDecoder(nn.Module):
    def __init__(self, **kwargs):
        
        super(LSTMDecoder, self).__init__()
        #Base variables
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab)
        self.out_dim = len(self.vocab)
        self.embed_dim = kwargs['character_embedding_size']
        self.start_token = kwargs['start_token']
        self.end_token = kwargs['end_token']
        self.unk_token = kwargs['unk_token']
        self.hid_dim = kwargs['embedding_size']
        self.n_layers = 1
        
        #Define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #Define the lstm layer
        self.lstm = nn.LSTM(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=self.n_layers)
        #Define the output layer
        self.linear = nn.Linear(self.hid_dim,self.out_dim)
        #Define the softmax layer
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, lengths):
        #Inputs are size (LxBx1)
        #Forward embedding layer
        emb = self.embed(inputs)
        #emb = inputs.new_zeros(inputs.shape[0],inputs.shape[1],self.embed_dim).float()
        #Embeddings are size (LxBxself.embed_dim)

        #Pack the sequences for GRU
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #Forward the GRU
        packed_rec, self.hidden = self.lstm(packed,self.hidden)
        #Unpack the sequences
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        #Hidden outputs are size (LxBxself.hidden_size)
        
        #Flatten for the output layer
        flat_rec = rec.view(rec.size(0)*rec.size(1), rec.size(2))
        #Forward the output layer and the softmax
        flat_out = self.softmax(self.linear(flat_rec))
        #Unflatten the output
        out = flat_out.view(emb.size(0),emb.size(1),flat_out.size(1))
        #Outputs are size (LxBxself.in_dim)
        
        return out
    
    def init_hidden(self, bsz=None, hidden=None):
        #Initialise the hidden state
        if hidden is not None:
            weight = next(self.parameters())
            self.hidden = (hidden,weight.new_zeros(self.n_layers, hidden.shape[1], self.hid_dim))
        elif bsz is not None:
            weight = next(self.parameters())
            self.hidden = (weight.new_zeros(self.n_layers, bsz, self.hid_dim),weight.new_zeros(self.n_layers, bsz, self.hid_dim))

    def detach_hidden(self):
        #Detach the hidden state
        self.hidden=(self.hidden[0].detach(),self.hidden[1].detach())

    def cpu_hidden(self):
        #Set the hidden state to CPU
        self.hidden=(self.hidden[0].detach().cpu(),self.hidden[1].detach().cpu())

    def cut_hidden(self,valid):
        #Reduce batch size in hidden state
        self.hidden=(self.hidden[0][:,0:valid,:].contiguous(),self.hidden[1][:,0:valid,:].contiguous())

#Initialise all random numbers for reproducibility
def init_random(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True

#Arguments
args = {
    'sample_length': 1024,
    'xsize': 384,
    'ysize': 256,
    'batch_size': 128,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'start_token': '*s*',
    'end_token': '*\s*',
    'unk_token': '*UNK*',
}

#Initialise RNGs
init_random(**args)

#Read models
encoder = torch.load('../input/moleculetranslation-cnnencoderlstmdecoder-training/cnn_encoder.pytorch').to(args['device'])
decoder = torch.load('../input/moleculetranslation-cnnencoderlstmdecoder-training/lstm_decoder.pytorch').to(args['device'])

#Get target vocabulary
args['vocab'] = decoder.vocab
args['inv_vocab'] = {args['vocab'][v]:v for v in args['vocab']}

#Process test data
print('Inference...')
args['mode'] = 'test'
args['early_stopping'] = True
test_data = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')
output = []
for i in range(100):
    st = int(i*len(test_data)/100)
    en = min(int((i+1)*len(test_data)/100),len(test_data))
    samples_idx = sample_model(test_data.iloc[st:en],encoder,decoder,**args)
    output += [indexes_to_characters(samples_idx[:,i],**args) for i in range(samples_idx.shape[1])]
    print('Completed {0:d}%'.format(i+1))
test_data['InChI'] = output
test_data.to_csv('submission.csv',index=False)
