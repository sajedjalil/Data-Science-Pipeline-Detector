import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import dill

class CstTokenizer:
    
    def __init__(self,use_tqdm=False):
        VOCAB = ['<start>','<end>','(', ')', '+', ',', '-', '=', '/b', '/c', '/h', '/i', '/m', '/s', '/t',
          *[str(x) for x in range(10)],
          'B', 'Br', 'C', 'Cl', 'D', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si', 'T']
        VOCAB = sorted(VOCAB,reverse=True)
        word_index = dict([(v,i+1)for i,v in enumerate(VOCAB)])
        index_word = dict([(i+1,v)for i,v in enumerate(VOCAB)])
        self.VOCAB = VOCAB
        self.word_index = word_index
        self.index_word = index_word
        self.n_process = 4
        self.use_tqdm = use_tqdm
    
    def split_(self,s,w,mark,res=[]):
        counter = 0
        size = len(w)
        token = ''
        token_idx = []

        for i,c in enumerate(s):
            token += c
            token_idx.append(i)
            counter += mark[i]
            if len(token)<size:
                continue
            if len(token)>size:
                token = token[1:]
                counter -= mark[token_idx[0]]
                del token_idx[0]


            if counter == 0 and token == w:
                for idx in token_idx:
                    mark[idx]=1
                    counter+=1
                res.append((token_idx[0],w))
        return sorted(res,key=lambda x:x[0])

    def split(self,s):
        res=[]
        mark = [0]*len(s)
        for w in self.VOCAB:
            res = self.split_(s,w,mark,res=res)
        return [c for (_,c) in res]
    
    def tokenize_(self,sequence):
        results = self.split(sequence)
        return [self.word_index[r] for r in results]
    
    def tokenize(self,sequences):
        return [self.tokenize_(s) for s in sequences]
    
    def detokenize_(self,token):
        return ''.join([self.index_word[x] for x in token])
    
    def detokenize(self,tokens):
        with mp.Pool(processes=self.n_process) as pool:
            if self.use_tqdm:
                results = pool.map(self.detokenize_,tqdm(tokens))
            else:
                results = pool.map(self.detokenize_,tokens)
        return results
    
    
    def count_elements_(self,inchi):
        inchi = self.split(inchi.split('/')[1])
        count = ""
        total = 0
        i = 3
        for c in inchi[1:]:
            if c.isdigit():
                count += c
            else:
                total += int(count) if count else 1
                count = ""
        if count:
            total += int(count) if count else 1
        else:
            total += 1

        return total
    
    def count_elements(self,inchis):
        return [self.count_elements_(inchi) for inchi in tqdm(inchis)]
    
    
    
if __name__ == '__main__':
    tokenizer = CstTokenizer(use_tqdm=True)
    print(tokenizer.word_index)
    
    # tokenize label
    start = '<start>'
    end = '<end>'
    df = pd.read_csv('../input/bms-molecular-translation/train_labels.csv')
    
    

    labels = [f"{start}{s[9:]}{end}" for s in df['InChI']]
    # do some test
    print(tokenizer.detokenize(tokenizer.tokenize(labels[:2])))
    # tokenize the targets column
    labels = tokenizer.tokenize(labels)
    
    dill.dump(labels,open('labels.dill','wb'))
    
    # count_elements
    count_elements = tokenizer.count_elements(df.InChI.values)
    dill.dump(count_elements,open('count_elements.dill','wb'))
    
    
    
    
    
    
    
    
    
    
    
    
    