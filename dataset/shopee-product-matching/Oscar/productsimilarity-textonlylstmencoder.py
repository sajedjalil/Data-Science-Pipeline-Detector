#Train and evaluate with PyTorch
import sys
import os
import math
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import copy
import torch
import torch.nn as nn

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
def correct_group_labels(train_data):
    labels = {}
    for l,gl in train_data.groupby('label_group'):
        labels[l] = list(np.unique(gl.index))
    labels_to_merge = []
    for l,gl in train_data.groupby('image_phash'):
        if len(gl) > 1:
            if len(np.unique(gl['label_group'])) > 1:
                to_merge = list(np.unique(gl['label_group'].values))
                seen = False
                for i,lm in enumerate(labels_to_merge):
                    if len(set(lm) & set(to_merge)) > 0:
                        labels_to_merge[i] = list(np.unique(labels_to_merge[i] + to_merge))
                        seen = True
                        break
                if not seen:
                    labels_to_merge.append(to_merge)
    for l in labels_to_merge:
        tgt = l[0]
        for src in l[1:]:
            labels[tgt] = labels[tgt] + labels[src]
            del[labels[src]]
    for l in labels:
        train_data.loc[train_data.index.isin(labels[l]),'label_group'] = l
    return train_data

def split_train_cv(train_data,label,**kwargs):
    labels = list(np.unique(train_data[label]))
    random.shuffle(labels)
    train_labels = labels[:-int(len(labels)*kwargs['cv_percentage'])]
    cv_labels = labels[-int(len(labels)*kwargs['cv_percentage']):]
    cv_data = train_data.loc[train_data[label].isin(cv_labels)]
    train_data = train_data.loc[train_data[label].isin(train_labels)]
    train_data = train_data.drop_duplicates(subset=['label_group','title'])
    train_data = train_data.reset_index(drop=True)
    cv_data = cv_data.reset_index(drop=True)
    return train_data, cv_data
    
def read_characters(lines,**kwargs):
    counts = dict()
    for line in lines:
        line = line.strip()
        for char in line:
            if char not in counts:
                counts[char] = 0
            counts[char]+=1
    total_counts = sum([counts[c] for c in counts])
    num_words = 0
    vocab = {}
    for c in counts:
        if counts[c] >= args['min_count']:
            vocab[c] = num_words
            num_words += 1
    final_counts = sum([counts[c] for c in vocab])
    print('Min counts: {0:d}. Original: {1:d} characters / {2:d} counts. Final: {3:d} characters / {4:d} counts. Coverage: {5:.1f}%'.format(args['min_count'],len(counts),total_counts,len(vocab),final_counts,100*final_counts/total_counts))
    for word in [kwargs['start_token'],kwargs['end_token'],kwargs['unk_token']]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab

def load_data(text, **kwargs):
    num_seq = len(text)
    max_words = max([len(list(t.strip()))+2 for t in text])
    dataset = len(kwargs['vocab'])*torch.ones((max_words,num_seq),dtype=torch.long)
    utoken_value = kwargs['vocab'][kwargs['unk_token']]
    idx = 0
    for line in tqdm(text,desc='Allocating data memory',disable=(kwargs['verbose']<2)):
        words = list(line.strip())
        if words[0] != kwargs['start_token']:
            words.insert(0,kwargs['start_token'])
        if words[-1] != kwargs['end_token']:
            words.append(kwargs['end_token'])
        for jdx,word in enumerate(words):
            dataset[jdx,idx] = kwargs['vocab'].get(word,utoken_value)
        idx += 1
    assert idx == num_seq
    return dataset

def make_triplets(data,**kwargs):
    indices = list(data.index)
    triplets = []
    for i in tqdm(indices,desc='Creating triplets',disable=(kwargs['verbose']<2)):
        label = data.iloc[i]['label_group']
        for j in list(data.loc[data['label_group'].isin([label])].index):
            if i!=j:
                rnd_index = []
                for _ in range(kwargs['num_triplets']):
                    rnd = random.choice(indices)
                    while data.iloc[rnd]['label_group'] == label or rnd in rnd_index:
                        rnd = random.choice(indices)
                    triplets.append([i,j,rnd,kwargs['targets'][label],kwargs['targets'][label],kwargs['targets'][data.iloc[rnd]['label_group']]])
                    rnd_index.append(rnd)
    random.shuffle(triplets)
    return np.array(triplets)

class LSTMEncoder(nn.Module):
    def __init__(self, **kwargs):
        
        super(LSTMEncoder, self).__init__()
        #Base variables
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab)
        self.start_token = kwargs['start_token']
        self.end_token = kwargs['end_token']
        self.unk_token = kwargs['unk_token']
        self.embed_dim = kwargs['embedding_size']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.targets = kwargs['targets']
        self.out_dim = len(self.targets)
        
        #Define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #Define the lstm layer
        self.lstm = nn.LSTM(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=self.n_layers)
        #L2 normalise
        self.l2norm = L2Norm()
        #Define the output layer
        self.linear = nn.Linear(self.hid_dim,self.out_dim)
        #Define the softmax layer
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, lengths, return_embeddings = False):
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
        #out = self.l2norm(out)
        
        if not return_embeddings:
            out = self.softmax(self.linear(out))
        
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
        
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis
    def forward(self,x):
        norm = torch.norm(x, 2, self.axis, True)
        output = torch.div(x, norm)
        return output
       
def euclidean_norm(x,y = None):
    if y is None:
        output = np.zeros((x.shape[0],x.shape[0]))
        idx = np.array(range(x.shape[0]))
        for i in range(x.shape[0]):
            tmp = np.sqrt(np.sum(np.square(x[:x.shape[0]-i,:]-x[i:,:]),axis=1))
            output[idx[:x.shape[0]-i],idx[i:]] = tmp
        output = np.triu(output,1) +  np.transpose(np.triu(output,0))
    else:
        output = np.zeros((x.shape[0],y.shape[0]))
        for i in range(x.shape[0]):
            output[i,:] = np.sqrt(np.sum(np.square(x[i:i+1]-y),axis=1))
    return output

def train_model(trainset,train_triplets,encoder,optimizer,criterion,**kwargs):
    trainlen = train_triplets.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        encoder = encoder.train()
        for b in range(nbatches):
            #Data batch
            indices = train_triplets[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])]
            anchors = trainset[:,indices[:,0]].clone().long().to(kwargs['device'])
            positive = trainset[:,indices[:,1]].clone().long().to(kwargs['device'])
            negative = trainset[:,indices[:,2]].clone().long().to(kwargs['device'])
            embeddings = {}
            for X,i in [(anchors,'anchors'),(positive,'positive'),(negative,'negative')]:
                mask = torch.clamp(len(kwargs['vocab'])-X,max=1)
                seq_length = torch.sum(mask,dim=0)
                ordered_seq_length, dec_index = seq_length.sort(descending=True)
                max_seq_length = torch.max(seq_length)
                X = X[:,dec_index]
                X = X[0:max_seq_length]
                encoder.init_hidden(X.size(1))
                embeddings[i] = encoder(X,ordered_seq_length.cpu(),return_embeddings=True)
                rev_dec_index = list(range(seq_length.shape[0]))
                for j,k in enumerate(dec_index):
                    rev_dec_index[k] = j
                embeddings[i] = embeddings[i][rev_dec_index]
            loss = criterion(embeddings['anchors'],embeddings['positive'],embeddings['negative'])
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

def evaluate_model(testset,encoder,**kwargs):
    testlen = testset.shape[1]
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    embeddings = np.zeros((testlen,kwargs['hidden_size']))
    with torch.no_grad():
        encoder = encoder.eval()
        for b in range(nbatches):
            #Data batch
            X = testset[:,b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            mask = torch.clamp(len(kwargs['vocab'])-X,max=1)
            seq_length = torch.sum(mask,dim=0)
            ordered_seq_length, dec_index = seq_length.sort(descending=True)
            max_seq_length = torch.max(seq_length)
            X = X[:,dec_index]
            X = X[0:max_seq_length]
            #Forward pass
            encoder.init_hidden(X.size(1))
            output = encoder(X,ordered_seq_length.cpu(),return_embeddings=True)
            #posteriors = model(X,ordered_seq_length)
            rev_dec_index = list(range(output.shape[0]))
            for i,j in enumerate(dec_index):
                rev_dec_index[j] = i
            embeddings[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])] = output[rev_dec_index].cpu().numpy()
    return embeddings

def compute_results(embeddings,labels,image_labels,**kwargs):
    scores = -1 * euclidean_norm(embeddings)
    #Items with the same image hash always have a maximum score of 0
    image_scores = np.array([[image_labels[i]==image_labels[j] for j in range(scores.shape[1])] for i in range(scores.shape[0])])
    scores = np.minimum(scores + 10*image_scores,0.0)
    pos_indices = np.array([[labels[i]==labels[j] for j in range(scores.shape[1])] for i in range(scores.shape[0])])
    idx_sort = np.argsort(scores,axis=1)[:,::-1]
    sorted_scores = np.array([scores[i,idx_sort[i]] for i in range(scores.shape[0])])
    sorted_pos_indices = np.array([pos_indices[i,idx_sort[i]] for i in range(scores.shape[0])])
    sorted_neg_indices = 1 - sorted_pos_indices
    thresholds = np.arange(np.max(scores),np.min(scores)-0.0001,-0.1)
    fpr = []
    tpr = []
    prec = []
    rec = []
    f1 = []
    for th in thresholds:
        selected_indices = (sorted_scores >= th)
        tp = np.sum(sorted_pos_indices * selected_indices, axis=1)
        fp = np.sum(sorted_neg_indices * selected_indices, axis=1)
        fpr.append(np.mean(fp/np.sum(sorted_neg_indices,axis=1)))
        tpr.append(np.mean(tp/np.sum(sorted_pos_indices,axis=1)))
        #Limit to best 50 for F1 computation
        selected_indices[:,50:] = 0
        unselected_indices = 1 - selected_indices
        tp = np.sum(sorted_pos_indices * selected_indices, axis=1)
        tn = np.sum(sorted_neg_indices * unselected_indices, axis=1)
        fp = np.sum(sorted_neg_indices * selected_indices, axis=1)
        fn = np.sum(sorted_pos_indices,axis=1) - tp
        prec.append(np.mean(tp / (tp + fp)))
        rec.append(np.mean(tp / (tp + fn)))
        f1.append(np.mean((2*tp) / (2*tp + fp + fn)))
    results = pd.DataFrame({'thresholds':thresholds,'tpr':tpr,'fpr':fpr,'prec':prec,'rec':rec,'f1':f1})
    return results

def calibrate_threshold(embeddings,labels,image_labels,target_samples,**kwargs):
    scores = -1 * euclidean_norm(embeddings)
    #Items with the same image hash always have a maximum score of 0
    image_scores = np.array([[image_labels[i]==image_labels[j] for j in range(scores.shape[1])] for i in range(scores.shape[0])])
    scores = np.minimum(scores + 10*image_scores,0.0)
    pos_indices = np.array([[labels[i]==labels[j] for j in range(scores.shape[1])] for i in range(scores.shape[0])])
    idx_sort = np.argsort(scores,axis=1)[:,::-1]
    sorted_scores = np.array([scores[i,idx_sort[i]] for i in range(scores.shape[0])])
    sorted_pos_indices = np.array([pos_indices[i,idx_sort[i]] for i in range(scores.shape[0])])
    sorted_neg_indices = 1 - sorted_pos_indices
    thresholds = np.arange(np.max(scores),np.min(scores)-0.0001,-0.005)
    prec = []
    rec = []
    f1 = []
    for th in thresholds:
        selected_indices = (sorted_scores >= th)
        #Limit to best 50 for F1 computation
        selected_indices[:,50:] = 0
        unselected_indices = 1 - selected_indices
        tp = np.zeros((scores.shape[0]))
        fp = np.zeros((scores.shape[0]))
        for i in range(scores.shape[0]):
            idx = 0
            for j in range(50):
                if selected_indices[i,j]:
                    if sorted_pos_indices[i,j]:
                        tp[i] += 1
                        idx += 1
                    else:
                        fp[i] += min(math.ceil(target_samples/scores.shape[0]),50-idx)
                        idx += math.ceil(target_samples/scores.shape[0])
                    if idx >= 50:
                        break
        fn = np.sum(sorted_pos_indices,axis=1) - tp
        prec.append(np.mean(tp / (tp + fp)))
        rec.append(np.mean(tp / (tp + fn)))
        f1.append(np.mean((2*tp) / (2*tp + fp + fn)))
    results = pd.DataFrame({'thresholds':thresholds,'prec':prec,'rec':rec,'f1':f1})
    return results

def generate_matches(test_data, th, **kwargs):
    testset = load_data(test_data['title'].values, **kwargs)
    test_ids = test_data['posting_id'].values
    image_labels = test_data['image_phash'].values
    test_embeddings = evaluate_model(testset,encoder,**kwargs)
    output = []
    batch_size = 512
    nbatches = math.ceil(test_embeddings.shape[0]/batch_size)
    for b in range(nbatches):
        test_scores = -1 * euclidean_norm(test_embeddings[b*batch_size:min(test_embeddings.shape[0],(b+1)*batch_size)],test_embeddings)
        image_scores = np.array([[image_labels[i]==image_labels[j] for j in range(test_scores.shape[1])] for i in range(b*batch_size,min(test_embeddings.shape[0],(b+1)*batch_size))])
        scores = np.minimum(test_scores + 10*image_scores,0.0)
        idx_sort = np.argsort(scores,axis=1)[:,::-1]
        idx_sort = idx_sort[:,:50]
        sorted_scores = np.array([scores[i,idx_sort[i]] for i in range(scores.shape[0])])
        selected_indices = (sorted_scores >= th)
        output += [" ".join([test_ids[idx_sort[i,j]] for j in range(selected_indices.shape[1]) if selected_indices[i][j]]) for i in range(selected_indices.shape[0])]
    return output

#Arguments
args = {
    'cv_percentage': 0.1,
    'epochs': 10,
    'batch_size': 128,
    'embedding_size': 32,
    'hidden_size': 128,
    'num_layers': 1,
    'learning_rate': 0.001,
    'seed': 0,
    'start_token': '*s*',
    'end_token': '*\s*',
    'unk_token': '*UNK*',
    'verbose': 1,
    'min_count': 1500,
    'num_triplets': 1,
    'margin': 1.0,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
}

#Set the seed for repeatable experiments
random_init(**args)

#Read training input
train_data = pd.read_csv('../input/shopee-product-matching/train.csv')

#Correct 'title' encodings and set to lowercase
train_data['title'] = [t.lower().encode('utf-8').decode('unicode_escape') for t in train_data['title'].values]
train_data = train_data.loc[train_data['title'].str.len()>0]
train_data = train_data.reset_index(drop=True)

#Merge group labels using 'image_phash'
old_groups = len(np.unique(train_data['label_group']))
train_data = correct_group_labels(train_data)
print('{0:d} groups initially, merged into {1:d} groups'.format(old_groups,len(np.unique(train_data['label_group']))))

#Separate train/cv sets by label_group value
train_data, cv_data = split_train_cv(train_data,'label_group',**args)
print('Train data: {0:d} samples, {1:d} labels. CV data: {2:d} samples, {3:d} labels'.format(len(train_data),len(np.unique(train_data['label_group'])),len(cv_data),len(np.unique(cv_data['label_group']))))

#Extract the character vocabulary from the train data and label targets
args['vocab'] = read_characters(train_data['title'].values, **args)
args['targets'] = {v:i for i,v in enumerate(np.unique(train_data['label_group']))}

#Load train and CV tensors in memory
trainset = load_data(train_data['title'].values, **args)
validset = load_data(cv_data['title'].values, **args)

#Determine training data triplets
train_triplets = make_triplets(train_data,**args)
print('{0:d} training triplets'.format(len(train_triplets)))

#Models and optimiser
encoder = LSTMEncoder(**args).to(args['device'])
optimizer = torch.optim.Adam(encoder.parameters(),lr=args['learning_rate'])
criterion = nn.TripletMarginLoss(margin=args['margin']).to(args['device'])

#Train with triplet loss
print('Training...')
for ep in range(1,args['epochs']+1):
    loss = train_model(trainset,train_triplets,encoder,optimizer,criterion,**args)
    val_embeddings = evaluate_model(validset,encoder,**args)
    results = compute_results(val_embeddings,cv_data['label_group'].values,cv_data['image_phash'].values,**args)
    auc = np.trapz(results['tpr'],x=results['fpr'])
    th, f1 = results.iloc[results['f1'].argmax()]['thresholds'],results.iloc[results['f1'].argmax()]['f1']
    print('Epoch {0:d} of {1:d}. Training loss: {2:.2f}, cross-validation AUC: {3:.3f}, best F1: {4:.3f} @ {5:.1f} threshold'.format(ep,args['epochs'],loss,auc,np.mean(f1),th))

#Read test data
print('Inferring...')
test_data = pd.read_csv('../input/shopee-product-matching/test.csv')
test_data['title'] = [t.lower().encode('utf-8').decode('unicode_escape') for t in test_data['title'].values]    

#Calibrate and estimate F1 for the number of samples
calibration = calibrate_threshold(val_embeddings,cv_data['label_group'].values,cv_data['image_phash'].values,len(test_data),**args)
th = calibration.loc[calibration['prec'] > calibration['prec'][0] * 0.9925]['thresholds'].iloc[-1]
f1 = calibration.loc[calibration['prec'] > calibration['prec'][0] * 0.9925]['f1'].iloc[-1]
prec = calibration.loc[calibration['prec'] > calibration['prec'][0] * 0.9925]['prec'].iloc[-1]
rec = calibration.loc[calibration['prec'] > calibration['prec'][0] * 0.9925]['rec'].iloc[-1]
print('Calibration: Threshold {0:.3f}, estimated F1 {1:.3f} (precision: {2:.3f}, recall: {3:.3f})'.format(th,f1,prec,rec))

#Generate output
output = generate_matches(test_data, th, **args)
test_data['matches'] = output
test_data = test_data[['posting_id','matches']]
test_data.to_csv('submission.csv',index=False)