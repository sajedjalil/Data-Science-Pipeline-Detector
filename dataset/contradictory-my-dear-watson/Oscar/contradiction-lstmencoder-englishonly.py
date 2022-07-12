import sys
import os
import math
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True

def normalise(text):
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = text.upper()
    words=[]
    for w in text.strip().split():
        if w.startswith('HTTP'):
            continue
        while len(w)>0 and w[0] not in chars:
            w = w[1:]
        while len(w)>0 and w[-1] not in chars:
            w = w[:-1]
        if len(w) == 0:
            continue
        words.append(w)
    text=' '.join(words)
    return text

def read_vocabulary(train_text, **kwargs):
    vocab = dict()
    counts = dict()
    num_words = 0
    for line in train_text:
        line = (list(line.strip()) if kwargs['characters'] else line.strip().split())
        for char in line:
            if char not in vocab:
                vocab[char] = num_words
                counts[char] = 0
                num_words+=1
            counts[char] += 1
    num_words = 0
    vocab2 = dict()
    if not kwargs['characters']:
        for w in vocab:
            if counts[w] >= args['min_count']:
                vocab2[w] = num_words
                num_words += 1
    vocab = vocab2
    for word in [kwargs['start_token'],kwargs['end_token'],kwargs['unk_token']]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab

def load_data(premise, hypothesis, targets=None, cv=False, **kwargs):
    assert len(premise) == len(hypothesis)
    num_seq = len(premise)
    max_words = max([len(t) for t in premise+hypothesis])+2
    dataset = len(kwargs['vocab'])*torch.ones((2,max_words,num_seq),dtype=torch.long)
    labels = torch.zeros((num_seq),dtype=torch.uint8)
    idx = 0
    utoken_value = kwargs['vocab'][kwargs['unk_token']]
    for i,line in tqdm(enumerate(premise),desc='Allocating data memory',disable=(kwargs['verbose']<2)):
        words = (list(line.strip()) if kwargs['characters'] else line.strip().split())
        if len(words)==0 or words[0] != kwargs['start_token']:
            words.insert(0,kwargs['start_token'])
        if words[-1] != kwargs['end_token']:
            words.append(kwargs['end_token'])
        for jdx,word in enumerate(words):
            dataset[0,jdx,idx] = kwargs['vocab'].get(word,utoken_value)
        line=hypothesis[i]
        words = (list(line.strip()) if kwargs['characters'] else line.strip().split())
        if len(words)==0 or words[0] != kwargs['start_token']:
            words.insert(0,kwargs['start_token'])
        if words[-1] != kwargs['end_token']:
            words.append(kwargs['end_token'])
        for jdx,word in enumerate(words):
            dataset[1,jdx,idx] = kwargs['vocab'].get(word,utoken_value)
        if targets is not None:
            labels[idx] = targets[i]
        idx += 1

    if cv == False:
        return dataset, labels

    idx = [i for i in range(num_seq)]
    random.shuffle(idx)
    trainset = dataset[:,:,idx[0:int(num_seq*(1-kwargs['cv_percentage']))]]
    trainlabels = labels[idx[0:int(num_seq*(1-kwargs['cv_percentage']))]]
    validset = dataset[:,:,idx[int(num_seq*(1-kwargs['cv_percentage'])):]]
    validlabels = labels[idx[int(num_seq*(1-kwargs['cv_percentage'])):]]
    return trainset, validset, trainlabels, validlabels

class LSTMEncoder(nn.Module):
    def __init__(self, **kwargs):
        
        super(LSTMEncoder, self).__init__()
        #Base variables
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab)
        self.start_token = kwargs['start_token']
        self.end_token = kwargs['end_token']
        self.unk_token = kwargs['unk_token']
        self.characters = kwargs['characters']
        self.embed_dim = kwargs['embedding_size']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        
        #Define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #Define the lstm layer
        self.lstm = nn.LSTM(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=self.n_layers)
    
    def forward(self, inputs, lengths):
        #Inputs are size (LxBx1)
        #Forward embedding layer
        emb = self.embed(inputs)
        #Embeddings are size (LxBxself.embed_dim)

        #Pack the sequences for GRU
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #Forward the GRU
        packed_rec, self.hidden = self.lstm(packed,self.hidden)
        #Unpack the sequences
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        #Hidden outputs are size (LxBxself.hidden_size)
        
        #Get last embeddings
        out = rec[lengths-1,list(range(rec.shape[1])),:]
        #Outputs are size (Bxself.hid_dim)
        
        return out
    
    def init_hidden(self, bsz):
        #Initialise the hidden state
        weight = next(self.parameters())
        self.hidden = (weight.new_zeros(self.n_layers, bsz, self.hid_dim),weight.new_zeros(self.n_layers, bsz, self.hid_dim))

    def detach_hidden(self):
        #Detach the hidden state
        self.hidden=(self.hidden[0].detach(),self.hidden[1].detach())

    def cpu_hidden(self):
        #Set the hidden state to CPU
        self.hidden=(self.hidden[0].detach().cpu(),self.hidden[1].detach().cpu())
        
class Predictor(nn.Module):
    def __init__(self, **kwargs):
        
        super(Predictor, self).__init__()
        self.hid_dim = kwargs['hidden_size']*2
        self.out_dim = 3
        #Define the output layer and softmax
        self.linear = nn.Linear(self.hid_dim,self.out_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,input1,input2):
        #Outputs are size (Bxself.hid_dim)
        inputs = torch.cat((input1,input2),dim=1)
        out = self.softmax(self.linear(inputs))
        return out

def train_model(trainset,trainlabels,encoder,predictor,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[2]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        encoder = encoder.train()
        for b in range(nbatches):
            #Data batch
            X1 = trainset[0,:,b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            mask1 = torch.clamp(len(kwargs['vocab'])-X1,max=1)
            seq_length1 = torch.sum(mask1,dim=0)
            ordered_seq_length1, dec_index1 = seq_length1.sort(descending=True)
            max_seq_length1 = torch.max(seq_length1)
            X1 = X1[:,dec_index1]
            X1 = X1[0:max_seq_length1]
            rev_dec_index1 = list(range(seq_length1.shape[0]))
            for i,j in enumerate(dec_index1):
                rev_dec_index1[j] = i
            X2 = trainset[1,:,b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            mask2 = torch.clamp(len(kwargs['vocab'])-X2,max=1)
            seq_length2 = torch.sum(mask2,dim=0)
            ordered_seq_length2, dec_index2 = seq_length2.sort(descending=True)
            max_seq_length2 = torch.max(seq_length2)
            X2 = X2[:,dec_index2]
            X2 = X2[0:max_seq_length2]
            rev_dec_index2 = list(range(seq_length2.shape[0]))
            for i,j in enumerate(dec_index2):
                rev_dec_index2[j] = i
            Y = trainlabels[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            #Forward pass
            encoder.init_hidden(X1.size(1))
            embeddings1 = encoder(X1,ordered_seq_length1)
            encoder.detach_hidden()
            encoder.init_hidden(X2.size(1))
            embeddings2 = encoder(X2,ordered_seq_length2)
            embeddings1 = embeddings1[rev_dec_index1]
            embeddings2 = embeddings2[rev_dec_index2]
            posteriors = predictor(embeddings1,embeddings2)
            loss = criterion(posteriors,Y)
            #Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Estimate the latest loss
            if total_backs == 100:
                total_loss = total_loss*0.99+loss.detach().cpu().numpy()
            else:
                total_loss += loss.detach().cpu().numpy()
                total_backs += 1
            encoder.detach_hidden()
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()
    return total_loss/(total_backs+1)

def evaluate_model(testset,encoder,predictor,**kwargs):
    testlen = testset.shape[2]
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    predictions = np.zeros((testlen,))
    with torch.no_grad():
        encoder = encoder.eval()
        for b in range(nbatches):
            #Data batch
            X1 = testset[0,:,b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            mask1 = torch.clamp(len(kwargs['vocab'])-X1,max=1)
            seq_length1 = torch.sum(mask1,dim=0)
            ordered_seq_length1, dec_index1 = seq_length1.sort(descending=True)
            max_seq_length1 = torch.max(seq_length1)
            X1 = X1[:,dec_index1]
            X1 = X1[0:max_seq_length1]
            rev_dec_index1 = list(range(seq_length1.shape[0]))
            for i,j in enumerate(dec_index1):
                rev_dec_index1[j] = i
            X2 = testset[1,:,b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            mask2 = torch.clamp(len(kwargs['vocab'])-X2,max=1)
            seq_length2 = torch.sum(mask2,dim=0)
            ordered_seq_length2, dec_index2 = seq_length2.sort(descending=True)
            max_seq_length2 = torch.max(seq_length2)
            X2 = X2[:,dec_index2]
            X2 = X2[0:max_seq_length2]
            rev_dec_index2 = list(range(seq_length2.shape[0]))
            for i,j in enumerate(dec_index2):
                rev_dec_index2[j] = i
            #Forward pass
            encoder.init_hidden(X1.size(1))
            embeddings1 = encoder(X1,ordered_seq_length1)
            encoder.init_hidden(X2.size(1))
            embeddings2 = encoder(X2,ordered_seq_length2)
            embeddings1 = embeddings1[rev_dec_index1]
            embeddings2 = embeddings2[rev_dec_index2]
            posteriors = predictor(embeddings1,embeddings2)
            #posteriors = model(X,ordered_seq_length)
            estimated = torch.argmax(posteriors,dim=1)
            predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])] = estimated.detach().cpu().numpy()
    return predictions
        
#Arguments
args = {
    'cv_percentage': 0.1,
    'epochs': 20,
    'batch_size': 128,
    'embedding_size': 16,
    'hidden_size': 64,
    'num_layers': 1,
    'learning_rate': 0.01,
    'seed': 0,
    'start_token': '<s>',
    'end_token': '<\s>',
    'unk_token': '<UNK>',
    'verbose': 1,
    'characters': False,
    'min_count': 15,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    }

#Read data
train_data = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
test_data = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
#Extract only English language cases
train_data = train_data.loc[train_data['language']=='English']
test_data = test_data.loc[test_data['language']=='English']
#Extract premises and hypothesis
train_premise = [normalise(v) for v in train_data.premise.values]
train_hypothesis = [normalise(v) for v in train_data.hypothesis.values]
test_premise = [normalise(v) for v in test_data.premise.values]
test_hypothesis = [normalise(v) for v in test_data.hypothesis.values]
train_targets = train_data.label.values
print('Training: {0:d} pairs in English. Evaluation: {1:d} pairs in English'.format(len(train_premise),len(test_premise)))
print('Label distribution in training set: {0:s}'.format(str({i:'{0:.2f}%'.format(100*len(np.where(train_targets==i)[0])/len(train_targets)) for i in [0,1,2]})))

batch_sizes = [64,128,256]
min_counts = [5,15,25]

it_idx = 0
valid_predictions = dict()
test_predictions = dict()
valid_accuracies = dict()

for batch_size in batch_sizes:
    for min_count in min_counts:
        args['batch_size'] = batch_size
        args['min_count'] = min_count
    
        random_init(**args)

        #Make vocabulary and load data
        args['vocab'] = read_vocabulary(train_premise+train_hypothesis, **args)
        #print('Vocabulary size: {0:d} tokens'.format(len(args['vocab'])))
        trainset, validset, trainlabels, validlabels = load_data(train_premise, train_hypothesis, train_targets, cv=True, **args)
        testset, _ = load_data(test_premise, test_hypothesis, None, cv=False, **args)

        #Create model, optimiser and criterion
        encoder = LSTMEncoder(**args).to(args['device'])
        predictor = Predictor(**args).to(args['device'])
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(predictor.parameters()),lr=args['learning_rate'])
        criterion = nn.NLLLoss(reduction='mean').to(args['device'])

        #Train epochs
        best_acc = 0.0
        for ep in range(1,args['epochs']+1):
            loss = train_model(trainset,trainlabels,encoder,predictor,optimizer,criterion,**args)
            val_pred = evaluate_model(validset,encoder,predictor,**args)
            test_pred = evaluate_model(testset,encoder,predictor,**args)
            acc = 100*len(np.where((val_pred-validlabels.numpy())==0)[0])/validset.shape[2]
            if acc >= best_acc:
                best_acc = acc
                best_epoch = ep
                best_loss = loss
                valid_predictions[it_idx] = val_pred
                valid_accuracies[it_idx] = acc
                test_predictions[it_idx] = test_pred
        print('Run {0:d}. Best epoch: {1:d} of {2:d}. Training loss: {3:.2f}, validation accuracy: {4:.2f}%, test label distribution: {5:s}'.format(it_idx+1,best_epoch,args['epochs'],best_loss,best_acc,str({i:'{0:.2f}%'.format(100*len(np.where(test_pred==i)[0])/len(test_pred)) for i in [0,1,2]})))
        it_idx += 1

#Do the score combination
best_epochs = np.argsort([valid_accuracies[ep] for ep in range(it_idx)])[::-1]
val_pred = np.array([valid_predictions[ep] for ep in best_epochs[0:5]])
val_pred = np.argmax(np.array([np.sum((val_pred==i).astype(int),axis=0) for i in [0,1,2]]),axis=0)
test_pred = np.array([test_predictions[ep] for ep in best_epochs[0:5]])
test_pred = np.argmax(np.array([np.sum((test_pred==i).astype(int),axis=0) for i in [0,1,2]]),axis=0)
acc = 100*len(np.where((val_pred-validlabels.numpy())==0)[0])/validset.shape[2]
print('Ensemble. Cross-validation accuracy: {0:.2f}%, test label distribution: {1:s}'.format(acc,str({i:'{0:.2f}%'.format(100*len(np.where(test_pred==i)[0])/len(test_pred)) for i in [0,1,2]})))
#Set all predictions to the majority category
df_out = pd.DataFrame({'id': pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')['id'], 'prediction': np.argmax([len(np.where(train_targets==i)[0]) for i in [0,1,2]])})
#Set only English language cases to the predicted labels
df_out.loc[df_out['id'].isin(test_data['id']),'prediction']=test_pred
df_out.to_csv('/kaggle/working/submission.csv'.format(it_idx,acc),index=False)