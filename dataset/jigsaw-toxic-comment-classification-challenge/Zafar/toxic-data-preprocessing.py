import pandas as pd
import numpy as np
import copy
import re
from keras.preprocessing.text import text_to_word_sequence
from nltk import WordNetLemmatizer


class BaseTokenizer(object):
    def process_text(self, text):
        raise NotImplemented

    def process(self, texts):
        for text in texts:
            yield self.process_text(text)


RE_PATTERNS = {
    ' american ':
        [
            'amerikan'
        ],

    ' adolf ':
        [
            'adolf'
        ],


    ' hitler ':
        [
            'hitler'
        ],

    ' fuck':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*', 
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'
        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
        ],

    ' ass hole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'
        ],

    ' bitch ':
        [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h'
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' trans gender':
        [
            'transgender'
        ],

    ' gay ':
        [
            'gay'
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'deek', 'd i c k'
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' bull shit ':
        [
            'bullsh\*t', 'bull\$hit'
        ],

    ' homo sex ual':
        [
            'homosexual'
        ],

    ' jerk ':
        [
            'jerk'
        ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots'
                                                                                      'i d i o t'
        ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'
        ],

    ' shit hole ':
        [
            'shythole'
        ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
        ],

    ' rape ':
        [
            ' raped'
        ],

    ' dumb ass':
        [
            'dumbass', 'dubass'
        ],

    ' ass head':
        [
            'butthead'
        ],

    ' sex ':
        [
            'sexy', 's3x', 'sexuality'
        ],


    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
        ],

    ' shut the fuck up':
        [
            'stfu'
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' mother fucker':
        [
            ' motha ', ' motha f', ' mother f', 'motherucker',
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],
}


class PatternTokenizer(BaseTokenizer):
    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]", patterns=RE_PATTERNS,
                 remove_repetitions=True):
        self.lower = lower
        self.patterns = patterns
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions

    def process_text(self, text):
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x.split()

    def process_ds(self, ds):
        ### ds = Data series

        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()
        # remove special chars
        if self.initial_filters is not None:
            ds = ds.str.replace(self.initial_filters, ' ')
        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
            ds = ds.str.replace(pattern, r"\1")

        for target, patterns in self.patterns.items():
            for pat in patterns:
                ds = ds.str.replace(pat, target)

        ds = ds.str.replace(r"[^a-z' ]", ' ')

        return ds.str.split()

    def _preprocess(self, text):
        # lower
        if self.lower:
            text = text.lower()

        # remove special chars
        if self.initial_filters is not None:
            text = re.sub(self.initial_filters, ' ', text)

        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            text = pattern.sub(r"\1", text)
        return text
        
def main():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    
    tokenizer = PatternTokenizer()
    train["comment_text"] = tokenizer.process_ds(train["comment_text"]).str.join(sep=" ")
    test["comment_text"] = tokenizer.process_ds(test["comment_text"]).str.join(sep=" ")
    train.to_csv("train_preprocessed.csv", index=False)
    test.to_csv("test_preprocessed.csv", index=False)
    
if __name__ == "__main__":
    main()