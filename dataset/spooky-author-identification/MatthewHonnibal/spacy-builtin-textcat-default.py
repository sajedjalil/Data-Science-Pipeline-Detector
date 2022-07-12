'''This kernel benchmarks spaCy 2's built-in text classification model, using fully
default settings, and no pre-trained vectors. There's one small trick for the task:
a simple data augmentation strategy, where we take slices of the training sentences
and add them as exra instances. This strategy would be pointless for a bag-of-words
model, but because we have ngrams via the convolutions, it helps.

The model code is pasted into the bottom of the script, so that you can
see it and play with different architectures. The model is rather intricate: it's
a convolutional neural network with a fairly tricky word embedding strategy, and
parametric attention. The model is also stacked with a unigram bag-of-words, although
I doubt that helps for this contest. The model is implemented using spaCy's ML library,
Thinc.
'''
import csv
import random
from math import log

from pathlib import Path
import spacy
from spacy.util import minibatch, decaying, compounding
import spacy.about


def read_data(loc):
    texts = []
    authors = []
    with loc.open('r') as file_:
        for row in csv.DictReader(file_, delimiter=','):
            texts.append(row['text'])
            authors.append(row['author'])
    return texts, authors


def format_data_for_spacy(texts, authors, all_authors):
    ys = []
    for true_author in authors:
        cats = {wrong_author: 0.0 for wrong_author in all_authors}
        cats[true_author] = 1.
        ys.append({'cats': cats})
    return list(zip(texts, ys))

    
def train(nlp, texts, authors, models_dir=None, use_default_model=False):
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat)
    label_set = set(authors)
    for label in label_set:
        textcat.add_label(label)
    train_data = format_data_for_spacy(texts, authors, label_set)
    random.shuffle(train_data)
    eval_data = train_data[:1000]
    train_data = train_data[len(eval_data):]
    optimizer = nlp.begin_training()
    if not use_default_model:
        textcat.model = build_text_classifier(3, width=64, pretrained_dims=300)
    best = None
    for i in range(6):
        losses = {}
        augmented = augment_data(train_data)
        for j, batch in enumerate(minibatch(augmented, size=128)):
            texts, annot = zip(*batch)
            nlp.update(texts, annot, sgd=optimizer, losses=losses, drop=0.3)
            if j % 10 == 0: # Pretty basic progress reporting
                if j:
                    print(j, len(batch), 'loss=', losses['textcat'])
                losses = {}
        with nlp.use_params(optimizer.averages):
            acc, loss = evaluate_model(nlp, eval_data)
            if not best or loss < best[0]:
                best = (loss, acc, nlp.to_bytes())
                print('Dev acc', loss, acc, '(new best)')
            else:
                print('Dev acc', loss, acc, '(best: %.2f)' % best[0])
    nlp.from_bytes(best[-1]) # Load our best weights back in
    return nlp
    
def augment_data(batch):
    out_texts = []
    out_authors = []
    for text, author in batch:
        out_texts.append(text)
        out_authors.append(author)
        for i in range(1, 3):
            tokens = text.split()
            n = len(tokens)
            if n < i:
                continue
            split = min(n-(i-1), max(i, int(random.random() * n)))
            out_texts.append(' '.join(tokens[:split]))
            out_texts.append(' '.join(tokens[split:]))
            out_authors.append(author)
            out_authors.append(author)
    return zip(out_texts, out_authors)

def evaluate_model(nlp, eval_data):
    right = 0.
    wrong = 0.
    loss = 0.
    for doc, gold in nlp.pipe(eval_data, as_tuples=True):
        score, guess = max((score, author) for author, score in doc.cats.items())
        if gold['cats'].get(guess):
            right += 1
        else:
            wrong += 1
        truth = [a for a, true in gold['cats'].items() if true][0]
        loss += log(doc.cats[truth])
    loss /= (right+wrong)
    print(right, wrong)
    return right / (right+wrong), -loss


def write_test(nlp, test_loc, output_loc):
    with test_loc.open('r') as file_:
        reader = csv.DictReader(file_, delimiter=',')
        texts = ((row['text'], row['id']) for row in reader)
        authors = sorted(nlp.get_pipe('textcat').labels)
        output = [['id'] + authors]
        for doc, id_ in nlp.pipe(texts, as_tuples=True):
            scores = [str(doc.cats.get(author, 0.0)) for author in authors]
            output.append([id_] + scores)
        print(output[0], output[1])
    with output_loc.open('w') as file_:
        lines = '\n'.join(','.join(row) for row in output)
        file_.write(lines)
            

def main(data_dir='../input'):
    data_dir = Path(data_dir)
    spacy_model = data_dir / 'spacyen-vectors-web-lg' / 'spacy-en_vectors_web_lg'/'en_vectors_web_lg'
    texts, authors = read_data(data_dir / 'spooky-author-identification' / 'train.csv')
    nlp = spacy.load(spacy_model)
    best = train(nlp, texts, authors)
    write_test(best, data_dir / 'spooky-author-identification' / 'test.csv', Path('run1.csv'))
    

'''Model code, from spacy/_ml.py
See here for explanation: https://support.prodi.gy/t/text-classifier-model-architecture/114/3

Set the use_default_model flag to False to use the model here, which you can then edit


Note that the architecture here doesn't assume labels are mutually exclusive. It may
be better to replace the last Affine() layer with a Softmax(), and remove the 
elementwise logistic() transform. Btw, "Affine()" is misnamed :(. It does
use a bias, so should be called "Linear". I haven't taken a maths class
since I was 16, and it shows sometimes...

The surest way to improve performance will be to use pre-trained vectors, as these
are added as extra features. You can use the GloVe vectors that come with spaCy:

https://spacy.io/models/en#en_vectors_web_lg . You can also add your own. Read them
in however you like, and make calls to nlp.vocab.set_vector(word, vector) . See
https://spacy.io/usage/vectors-similarity#custom-vectors-add for more info.

If you want to get the model after training you can either serialize the bytes
produced by nlp.to_bytes(), or use nlp.to_disk(some_path). Pickle should also
work. Using nlp.to_disk() is nice because you can then call "spacy package"
on the directory, which will wrap it with a setup.py etc, so you can call
"python setup.py bdist_wheel" to build a wheel file you can install on
your server or wherever. It'll have the model data within it, so you
can write something like:

import spacy_spooky_model

nlp = spacy_spooky_model.load()

Anyway. Here's the model code. Happy hacking!
'''

from spacy._ml import *
from thinc.v2v import Softmax, SELU, Maxout
from thinc.t2v import max_pool
from thinc.api import layerize


def build_text_classifier(nr_class, width=64, **cfg):
    nr_vector = cfg.get('nr_vector', 5000)
    pretrained_dims = cfg.get('pretrained_dims', 0)
    with Model.define_operators({'>>': chain, '+': add, '|': concatenate,
                                 '**': clone}):
        lower = HashEmbed(width, nr_vector, column=1)
        prefix = HashEmbed(width//2, nr_vector, column=2)
        suffix = HashEmbed(width//2, nr_vector, column=3)
        shape = HashEmbed(width//2, nr_vector, column=4)

        trained_vectors = (
            FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE, ID])
            >> with_flatten(
                uniqued(
                    (lower | prefix | suffix | shape)
                    >> LN(Maxout(width, width+(width//2)*3)),
                    column=0
                )
            )
        )

        if pretrained_dims:
            static_vectors = (
                SpacyVectors
                >> with_flatten(Affine(width, pretrained_dims))
            )
            # TODO Make concatenate support lists
            vectors = concatenate_lists(trained_vectors, static_vectors)
            vectors_width = width*2
        else:
            vectors = trained_vectors
            vectors_width = width
            static_vectors = None
        cnn_model = (
            vectors
            >> flatten_add_lengths
            >> with_getitem(0,
                LN(Maxout(width, vectors_width))
                >> Residual(
                    (ExtractWindow(nW=1) >> LN(Maxout(width, width*3)))
                ) ** 3
            )
            >> Pooling(max_pool)
        )

        linear_model = (
            ngrams(Model.ops, attr=ORTH, bigrams=True, trigrams=False, tetragrams=False)
            >> LinearModel(nr_class, size=2**16)
        )

        model = (
            (linear_model | cnn_model)
            >> zero_init(Affine(nr_class, width+nr_class))
            >> logistic
        )
    model.nO = nr_class
    model.lsuv = False
    return model


def ngrams(ops, attr=ORTH, bigrams=True, trigrams=False, tetragrams=False):
    def preprocess_fwd(docs, drop=0.):
        unigrams = [doc.to_array(attr) for doc in docs]
        keys = []
        for doc_unis in unigrams:
            doc_keys = [doc_unis]
            if bigrams and doc_unis.shape[0] >= 2:
                doc_keys.append(ops.ngrams(2, doc_unis))
            if trigrams and doc_unis.shape[0] >= 3:
                doc_keys.append(ops.ngrams(3, doc_unis))
            if tetragrams and doc_unis.shape[0] >= 4:
                doc_keys.append(ops.ngrams(4, doc_unis))
            keys.append(ops.xp.concatenate(doc_keys))
        keys, vals = zip(*[ops.xp.unique(k, return_counts=True) for k in keys])
        lengths = ops.asarray([arr.shape[0] for arr in keys], dtype=numpy.int_)
        keys = ops.xp.concatenate(keys)
        vals = ops.asarray(ops.xp.concatenate(vals), dtype='f')
        if drop:
            mask = ops.get_dropout_mask(vals.shape, drop)
            vals *= mask
        return (keys, vals, lengths), None
    return layerize(preprocess_fwd)

    
if __name__ == '__main__':
    main()