from contextlib import contextmanager
import os
import random
import re
import string
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.optimizer import Optimizer

n_row=None

EMBEDDING_FASTTEXT = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
TRAIN_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
SAMPLE_SUBMISSION = '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'

embed_size = 300
max_features = 100000
maxlen = 220

batch_size = 2048
train_epochs = 5
n_splits = 5

seed = 1029


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

misspell_dict = {" st ":" ",
"aren't": "are not",' shtty ':" shitty ", "can't": "cannot", "couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do not",
"hadn't": "had not", "hasn't": "has not", "haven't": "have not","he'd": "he would", "he'll": "he will", "he's": "he is",
"i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not","it's": "it is", "it'll": "it will", "i've": "I have",
"let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
"she's": "she is","shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
"they'll": "they will", "they're": "they are","they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not", 
"we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
"where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have",
"won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", 
"you've": "you have","'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying",
 ' 4r5e ':" arse ",' 5h1t ':" shit ",' 5hit ':" shit ",' a55 ':" ass ",' anus ':' anal ',' ar5e ':" arse ",' arrse ':' arse ',
 ' ass-fucker ':" ass fucker ",' asses ':" ass ",' assfucker ':" ass fucker ",
 ' assfukka ':"ass fucker",'assholes':"ass hole",' asswhole ':' ass hole ',' a_s_s ':" ass ",' b!tch ':' bitch ',
 ' b00bs ':' boobs ',' b17ch ':" bitch ",' b1tch ':' bitch ',' bi+ch ':" bitch ",' biatch ':' bitch ',
 ' bitchin ':' bitching ',' booobs ':' boobs ',' boooobs ':' boos ',' booooobs ':' boobs ',' booooooobs ':' boobs ',
 ' butthole ':' butt hole ',' buttmuch ':" butt much ",' buttplug ':" butt plug ",' c0ck ':' cock ',' c0cksucker ':" cock sucker ",
 ' cawk ':' cock ',' cl1t ':' clit ',' clitoris ':" clit ",' clits ':' clit ',' cock-sucker ':' cock sucker ',' cockface ':' cock face ',' cockhead ':' cock head ',
 ' cockmunch ':"cock munch ",' cockmuncher ':" muncher of cock ",' cocksuck ':" cock suck ",' cocksucked ':" cock sucked ",' cocksucker ':" cock sucker ",
 ' cocksucking ':" sucking cock ",' cocksucks ':" cock sucks ",' cocksuka ':" cock sucker ",
 ' cocksukka ':" cock sucker ",' cok ':" cock ",' cokmuncher ':" muncher of cock",' coksucka ':" cock sucker ",
 ' cunillingus ':' cunilingus ',' cunnilingus ':' cunilingus ',' cuntlick ':' cunt licker ',' cuntlicker ':" cunt licker ",' cuntlicking ':" cunt licking ",

 ' cyberfuc ':" cybe rfuck ",' cyberfuck ':" cyber fuck ",' cyberfucked ':" cyber fucked ",' cyberfucker ':" cyber fucker ",
 ' cyberfuckers ':" cyber fuckers ",' cyberfucking ':" cyber fucking ",' d1ck ':" dick ",
 ' dickhead ':" dick head ",' dlck ':" dick ",' dog-fucker ':"dog fucker ",' doggin ':' dogging ',
 ' duche ':" douche ",' dyke':" douche ",' ejaculatings ':' ejaculating ',
 ' ejakulate ':" ejaculate ",' f u c k ':" fuck ",' f u c k e r ':"fucker",' f4nny ':" fanny ",' faggitt ':" fag gitt ",
 ' fannyflaps ':" fanny flaps ",' fannyfucker ':" fanny fucker ",'fanyy':" fanny ",' fatass ':" fat ass ",' fcuk ':" fuck ",' fcuker ':" fucker ",
 ' fcuking ':" fucking ",' feck ':" fuck ",' fecker ':" fucker ",
}
"""
 ' fellate ':" fellation ",' fellatio ':" fellation ",' fingerfuck ':" finger fuck ",' fingerfucked ':" finger fucked ",' fingerfucker ':" finger fucker ",
 ' fingerfuckers ':" finger fuckers ",' fingerfucking ':" finger fucking ",' fingerfucks ':" finger fucks ",
 ' fistfuck ':" fist fuck ",' fistfucked ':" fist fucked ",' fistfucker ':" fist fucker ",' fistfuckers ':" fist fuckers ",' fistfucking ':" fist fucking ",
 ' fistfuckings ':" fist fuckings ",' fistfucks ':" fist fucks ",' fook ':" fuck ",
 ' fooker ':" fucker ",' fucka ':" fucker ",' fuckhead ':" fuck head ",' fuckin ':' fucking ',
 ' fuckingshitmotherfucker ':" fucking shit mother fucker ",' fuckme ':" fuck me ",' fuckwhit ':" fuck with it ",' fuckwit ':" fuck with it ",
 ' fudgepacker ':' fudge packer ',' fuk ':" fuck ",' fuker ':' fucker ',
 ' fukker ':" fucker ",' FFFUUU ':" fuck you ",' fukkin ':" fucking ",' fuks ':" fucks ",' fukwhit ':" fuck with it ",' fukwit ':" fuck with it ",
 ' fux ':' fucks ',' fux0r ':" fucker ",' f_u_c_k ':" fuck ",
 ' gaylord ':" gay lord ",' gaysex ':" gay sex ",' goatse ':" goat sex ",' hardcoresex ':" hardcore sex ",
 ' hoar ':" whore ",' hoare ':" whore ",' hoer ':" whore ",' hore ':" whore ",' hotsex ':" hot sex ",' jack-off ':" jack off ",' jackoff ':" jack off ",
 ' jerk-off ':" jack off",' knobead ':" knob head ",' knobed ':" knob head ",' knobend ':" knob head ",' knobhead ':" knob head ",
 
 ' knobjokey ':" knob jockey ",'kondum':"condom",'kondums':"condoms",' l3i+ch ':" bitch ",'l3itch':"bitch",
 ' m0f0 ':" mofo ",' m0fo ':" mofo ",' m45terbate ':" master bate ",' ma5terb8 ':"master bate",'ma5terbate':"master bate",
 ' master-bate ':" master bate ",' masterb8 ':" master bate ",
 ' masterbat* ':" master bate ",' masterbat3 ':" master bate ",' masterbate ':" master bate ",
 ' masterbation ':" master bation ",' masterbations ':" master bation ",' masturbate ':" master bate ",' mo-fo ':" mofo ",' mof0 ':' mofo ',' mothafuck ':" mother fucker ",
 ' mothafucka ':" mother fucker ",' mothafuckas ':" mother fuckers ",' mothafuckaz ':" mother fucker ",' mothafucked ':" mother fucker ",' mothafucker ':" mother fucker ",
 ' mothafuckers ':" mother fuckers ",' mothafuckin ':" mother fucking ",' mothafucking ':" mother fucking ",
 ' mothafuckings ':" mother fucking ",' mothafucks ':" mother fuck ",' motherfuck ':" mother fucker ",' motherfucked ':" mother fucker ",
 ' motherfucker ':" mother fucker ",' motherfuckers ':" mother fucker ",
 ' motherfuckin ':" mother fucker ",' motherfucking ':" mother fucker ",' motherfuckings ':" mother fucker ",' motherfuckka ':" mother fucker ",
 ' motherfucks ':" mother fucker ",' muff ':" mother fucker ",' mutha ':" mother ",' muthafecker ':" mother fucker ",
 ' muthafuckker ':" mother fucker ",' muther ':" mother fucker ",' mutherfucker ':" mother fucker ",' n1gga':"nigger ",' n1gger ':" nigger ",
 ' nigg3r ':" nigger ",' nigg4h ':" nigger ",' nigga ':" nigger ",' niggah ':" nigger ",' niggas ':" nigger ",
 
 ' niggaz ':' nigger ',' niggers ':" nigger ",' nobjocky ':' nob jokey ',' nobjokey ':' nob jokey ',' numbnuts ':" num bnuts ",' nutsack ':" nut suck ",
 ' orgasm ':" orgasim ",' orgasms ':" orgasims ",' p0rn ':" porn ",' penisfucker ':" penis fucker ",' phonesex ':" phone sex ",' phuck ':" fuck ",
 ' phuk ':" fuck ",' phuked ':" fucked ",' phuking ':" fucking ",' phukked ':" fucked ",' phukking ':" fucking ",' phuks ':" fuck ",' phuq ':" fuck ",
 ' pigfucker ':" pig fucker ",' pissflaps ':" piss flaps ",' pissin ':' pissing ',' pissoff ':" piss off ",' poop ':' porn ',' porno ':" porn ",
 ' pornography ':" porn ography ",' pornos ':" porn ",' pron ':" porn ",
 ' pusse ':" pussy ",' pussi ':" pussy ",' pussies ':" pussy ",' pussys ':" pussy ",' s hit ':" shit ",'s.o.b.':"son of bitch",' sh*t ':" shit ",
 ' sh!+ ':" shit ",' sh!t ':" shit ",' sh1t ':" shit ",
 ' shaggin':' shagging ',' shi+ ':' shit ',' shitdick ':" shit dick ",' shite ':" shit ",' shited ':" shit ",' shitey ':" shit ",
 ' shitfuck ':" shit fuck ",' shitfull ':" shit full ",' shithead ':" shit head ",
 ' shiting ':" shitting ",' shitings ':" shittings ",
 ' son-of-a-bitch ':" son of a bitch ",' s_h_i_t ':" shit ",' t1tt1e5 ':" titties ",' t1tties ':" titties ",' teets ':" tits ",' teez ':" tits ",
 ' tit ':" tits ",' titfuck ':" tits fuck ",' titt ':' tits ',' tittie5 ':" titties ",' tittiefucker ':" titties fucker ",' tittyfuck ':" titty fuck ",
 ' tittywank ':" titty wank ",' titwank ':" tits wank ",
 ' tw4t ':' twat ','twathead ':" twat head ",' twunt ':" twat ",' twunter ':" twater ",' v14gra ':" viagra ",' v1gra ':" viagra ",
 ' viagra ':' vagina ',' vulva ':' vagina ',
 ' w00se ':" woose ",'wang ':" wanker ",' wank ':' wanker ',' whoar ':' whore '," ur ":"you are"}
 """
print(len(misspell_dict.keys()))
def _get_misspell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    return misspell_dict, misspell_re


def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell(misspell_dict)
    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)
    

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']


def clean_text(x):
    x = str(x)
    for punct in puncts + list(string.punctuation):
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    return re.sub('\d+', ' ', x)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_fasttext(word_index):
    embeddings_index = dict(get_coefs(*o.strip().split(' ')) for o in open(EMBEDDING_FASTTEXT))

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_and_prec(n_row):
    train = pd.read_csv(TRAIN_DATA, nrows=n_row, index_col='id')
    test = pd.read_csv(TEST_DATA, nrows=n_row, index_col='id')
    # lower
    train['comment_text'] = train['comment_text'].str.lower()
    test['comment_text'] = test['comment_text'].str.lower()

    # clean misspellings
    train['comment_text'] = train['comment_text'].apply(replace_typical_misspell)
    test['comment_text'] = test['comment_text'].apply(replace_typical_misspell)

    # clean the text
    train['comment_text'] = train['comment_text'].apply(clean_text)
    test['comment_text'] = test['comment_text'].apply(clean_text)

    # clean numbers
    train['comment_text'] = train['comment_text'].apply(clean_numbers)
    test['comment_text'] = test['comment_text'].apply(clean_numbers)
    
    # fill up the missing values
    train_x = train['comment_text'].fillna('_##_').values
    test_x = test['comment_text'].fillna('_##_').values
    
    # tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_x))
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    # pad the sentences
    train_x = pad_sequences(train_x, maxlen=maxlen)
    test_x = pad_sequences(test_x, maxlen=maxlen)
    
    # get the target values
    train_y = (train['target'].values > 0.5).astype(int)

    # shuffling the data
    np.random.seed(seed)
    train_idx = np.random.permutation(len(train_x))

    train_x = train_x[train_idx]
    train_y = train_y[train_idx]

    return train_x, train_y, test_x, tokenizer.word_index
    

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        lstm_hidden_size = 120
        gru_hidden_size = 60
        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

        self.lstm = nn.LSTM(embed_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(gru_hidden_size * 6, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.unsqueeze(h_embedding.transpose(1, 2), 2)
        h_embedding = torch.squeeze(self.embedding_dropout(h_embedding)).transpose(1, 2)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    warnings.filterwarnings('ignore')
    
    with timer('load data'):
        train_x, train_y, test_x, word_index = load_and_prec(n_row)
        embedding_matrix = load_fasttext(word_index)
        
    with timer('train'):
        train_preds = np.zeros((len(train_x)))
        test_preds = np.zeros((len(test_x)))
    
        seed_torch(seed)
    
        x_test_cuda = torch.tensor(test_x, dtype=torch.long).cuda()
        test = torch.utils.data.TensorDataset(x_test_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x, train_y))
    
        for fold, (train_idx, valid_idx) in enumerate(splits):
            x_train_fold = torch.tensor(train_x[train_idx], dtype=torch.long).cuda()
            y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
    
            x_val_fold = torch.tensor(train_x[valid_idx], dtype=torch.long).cuda()
            y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()
    
            model = NeuralNet(embedding_matrix)
            model.cuda()
    
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    
            print(f'Fold {fold + 1}')
    
            for epoch in range(train_epochs):
                start_time = time.time()
    
                model.train()
                avg_loss = 0.
    
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    y_pred = model(x_batch)
    
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)
    
                model.eval()
                valid_preds_fold = np.zeros((x_val_fold.size(0)))
                test_preds_fold = np.zeros(len(test_x))
                avg_val_loss = 0.
    
                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    with torch.no_grad():
                        y_pred = model(x_batch).detach()
    
                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    
                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))
    
            for i, (x_batch,) in enumerate(test_loader):
                with torch.no_grad():
                    y_pred = model(x_batch).detach()
    
                test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    
            train_preds[valid_idx] = valid_preds_fold
            test_preds += test_preds_fold / len(splits)
            
        print(f'cv score: {roc_auc_score(train_y, train_preds):<8.5f}')
    
    with timer('submit'):
        test = pd.read_csv(TEST_DATA, nrows=n_row)
        test_ids = test.id
        submission = pd.DataFrame.from_dict({
        'id': test_ids,
        'prediction': test_preds})
        submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
