#Autoregressive LSTM trained on InChI sequences without image input
import sys
import os
import math
import glob
import subprocess
import numpy as np
import pandas as pd
import random
import torch
import Levenshtein
import torch.nn as nn

#Clone and import RecurrentLM repository
if ~os.path.isdir('RecurrentLM'):
    subprocess.call(['git', 'clone', 'https://github.com/saztorralba/RecurrentLM'])
if 'RecurrentLM' not in sys.path:
    sys.path.append('RecurrentLM')
from utils.rnnlm_func import load_data, build_model, train_model, validate_model
from utils.lm_func import read_characters, count_sequences, read_sentences

def sample_model(model,**kwargs):
    output_idx = np.zeros((1,kwargs['max_length']))
    with torch.no_grad():
        model = model.eval()
        X = kwargs['vocab'][kwargs['start_token']]*torch.ones((1,1),dtype=torch.long).to(kwargs['device'])
        output_idx[0,0] = X.cpu().numpy()
        seq_length = torch.ones((1),dtype=torch.long)
        model.init_hidden(bsz=1)
        for i in range(1,kwargs['max_length']):
            #Forward the last symbol
            posteriors = model(X,seq_length)
            X = torch.argmax(posteriors,dim=2,keepdim=False)
            output_idx[0,i] = X.cpu().numpy()
    return output_idx

def sample_model_with_prompts(model,prompts,**kwargs):
    num_seq = prompts.shape[1]
    nbatches = math.ceil(num_seq/kwargs['batch_size'])
    mask = 1 + torch.clamp(prompts,max=0)
    inv_mask = 1 - mask
    output_idx = np.zeros((kwargs['max_length'],num_seq))
    with torch.no_grad():
        model = model.eval()
        for b in range(nbatches):
            batch = prompts[:,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])].clone().long().to(args['device'])
            X = batch[0:1,:]
            mask = 1 + torch.clamp(batch,max=0)
            inv_mask = 1 - mask
            output_idx[0,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])] = X.cpu().numpy()
            seq_length = torch.ones((X.shape[1]),dtype=torch.long)
            model.init_hidden(bsz=X.shape[1])
            for i in range(1,kwargs['max_length']):
                #Forward the last symbol
                posteriors = model(X,seq_length)
                best = torch.argmax(posteriors,dim=2,keepdim=False)
                if i < batch.shape[0]:
                    X = (batch[i:i+1,:] * mask[i:i+1,:]) + (best * inv_mask[i:i+1,:])
                else:
                    X = best
                output_idx[i,b*kwargs['batch_size']:min(num_seq,(b+1)*kwargs['batch_size'])] = X.cpu().numpy()
    return output_idx

def indexes_to_characters(input_idx,**kwargs):
    output = ''
    for i in input_idx:
        if args['inv_vocab'][i] == args['end_token']:
            break
        elif args['inv_vocab'][i] not in [args['start_token'],args['end_token']]:
            output += args['inv_vocab'][i]
    return output
    
#Arguments
args = {
    'train_samples': 500000,
    'cv_percentage': 0.05,
    'epochs': 10,
    'batch_size': 128,
    'embedding_size': 8,
    'hidden_size': 512,
    'num_layers': 1,
    'learning_rate': 0.001,
    'seed': 0,
    'bptt': sys.maxsize,
    'max_length': 512,
    'ltype': 'lstm',
    'start_token': '*s*',
    'end_token': '*\s*',
    'unk_token': '*UNK*',
    'verbose': 1,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    }

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
random_init(**args)

# Read training values, keep a subset for lower footprint
train_data = pd.read_csv('../input/bms-molecular-translation/train_labels.csv')
sentences = train_data.sample(args['train_samples'])['InChI'].values

#Read data
args['vocab'],args['characters'] = read_characters(lines=sentences,**args)
args['num_seq'],args['max_words'] = count_sequences(lines=sentences,**args)
trainset,validset = load_data(lines=sentences,cv=True, **args)

#Create model, optimiser and criterion
model = build_model(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
criterion = nn.NLLLoss(reduction='none').to(args['device'])

#Train epochs
print('Training...')
for ep in range(1,args['epochs']+1):
    loss = train_model(trainset,model,optimizer,criterion,**args)
    ppl = validate_model(validset,model,**args)
    if args['verbose'] == 1:
         print('Epoch {0:d} of {1:d}. Training loss: {2:.2f}, cross-validation perplexity: {3:.2f}'.format(ep,args['epochs'],loss,ppl))

#Generate most likely output from the model
args['inv_vocab'] = {args['vocab'][w]:w for w in args['vocab']}
best_output_idx = sample_model(model,**args)
best_output = indexes_to_characters(best_output_idx[0,:],**args)

#Compute Levenshtein distances to the validation samples
lev = []
for i in range(validset.shape[1]):
    target = indexes_to_characters(validset[:,i].numpy(),**args)
    lev.append(Levenshtein.distance(target,best_output))
print('Average Levenshtein distance to validation set: {0:.1f}'.format(np.mean(lev)))

#Use most likely output as simple prediction for all test_files
test_data = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')
test_data['InChI'] = best_output
test_data.to_csv('submission.csv',index=False)

#EXTRA EVALUATION: Test how well the model predicts the whole InChI sequence when seeded with the initial formula
#prompts = []
#for i in range(validset.shape[1]):
#    target = indexes_to_characters(validset[:,i].numpy(),**args)
#    prompts.append(np.array([args['vocab'][c] for c in list('/'.join(target.split('/')[0:2])+'/')]))
#t_prompts = -1 * torch.ones((max([len(p) for p in prompts]),len(prompts))).long()
#for i in range(len(prompts)):
#    t_prompts[0:len(prompts[i]),i] = torch.from_numpy(prompts[i])
#output_idx = sample_model_with_prompts(model,t_prompts,**args)
#lev = []
#for i in range(len(prompts)):
#    output = indexes_to_characters(output_idx[:,i],**args)
#    target = indexes_to_characters(validset[:,i].numpy(),**args)
#    lev.append(Levenshtein.distance(target,output))
#print('Average Levenshtein distance to validation set when using formula as prompt: {0:.1f}'.format(np.mean(lev)))