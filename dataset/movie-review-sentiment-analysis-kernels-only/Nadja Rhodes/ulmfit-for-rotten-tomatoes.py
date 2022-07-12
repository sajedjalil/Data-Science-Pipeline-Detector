# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from pathlib import Path
from fastai.text import *
from fastai.lm_rnn import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input/ulmfit-rt/ulmfit"))

ulmfit_path = Path('../input/ulmfit-rt/ulmfit')
tmp_path = ulmfit_path / 'tmp'
m_path = ulmfit_path / 'models'

c_path = Path('../input/movie-review-sentiment-analysis-kernels-only')

print('Loading data...')

trn_sent = np.load(tmp_path / f'trn_ids.npy')
val_sent = np.load(tmp_path / f'val_ids.npy')

trn_lbls = np.load(tmp_path / f'lbl_trn.npy')
val_lbls = np.load(tmp_path / f'lbl_val.npy')

trn_lbls = trn_lbls.flatten()
val_lbls = val_lbls.flatten()
trn_lbls -= trn_lbls.min()
val_lbls -= val_lbls.min()

c=int(trn_lbls.max())+1

itos = pickle.load(open(tmp_path / 'itos.pkl', 'rb'))
vs = len(itos)

bptt,em_sz,nh,nl,bs = 70,400,1150,3,64
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
dps = np.array([0.4,0.5,0.05,0.3,0.4])*1.0

trn_ds = TextDataset(trn_sent, trn_lbls)
val_ds = TextDataset(val_sent, val_lbls)
trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData('.', trn_dl, val_dl)

print('Loading model...')

m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn, models_name='')
learn.load(m_path / 'fwd_pretrain_aclImdb_clas_1')

learn.model.reset()
learn.model.eval()

print('Classifying...')

s_lbls = ['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive']
spacy_tok = spacy.load('en')
stoi = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos)})

def classify(model, text, print_res=True):    
    idxs = [stoi[w.text] if stoi[w.text] > 0 else 0 for w in spacy_tok(text.lower())]
    if -1 in idxs:
        raise ValueError(f'stoi bug: {list(zip(idxs, spacy_tok(text.lower())))}')
    preds = model(to_gpu(V(T([idxs])).transpose(0, 1)))[0]
    top_i = preds.topk(1)[1].item()
    if print_res:
        print(f'\'{text}\' - {s_lbls[top_i]} ({top_i})')
    return top_i

test_df = pd.read_csv(c_path / 'test.tsv', sep='\t')
test_phrases = test_df['Phrase']

sub = pd.read_csv(c_path / 'sampleSubmission.csv')
sub['Sentiment'] =  [classify(learn.model, test_phrases[i], print_res=False) for i in range(len(test_phrases))]

# Any results you write to the current directory are saved as output.
sub.to_csv("submission.csv", index=False)
