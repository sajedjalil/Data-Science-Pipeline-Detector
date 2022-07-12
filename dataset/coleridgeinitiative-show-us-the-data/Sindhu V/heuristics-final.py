import pandas as pd, numpy as np, re, json, spacy
print(spacy.__version__)
search_words = ['survey', 'study', 'data', 'database', 'data base', 'databases', 'sample']

nlp = spacy.load('en_core_web_lg')
DATA_DIR = '../input/coleridgeinitiative-show-us-the-data/test'
test = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/sample_submission.csv')
test_ids = test['Id'].values.tolist()

hardcoded = ['adni', "alzheimer's disease neuroimaging initiative (adni)"]

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

def post_process_results(entities):
    pp_entities = []
    for ent in entities:
        ent = ent.lower()
        if ent in hardcoded:
            pp_entities.append(ent)
            continue
        #TODO: Remove num, punct.
        doc = nlp(ent)
        #print([x.pos_ for x in doc])
        ftoks = list(filter(lambda token: token.pos_ !='DET' and token.pos_ !='NUM' and token.pos_ !='PUNCT',  doc))
        if len(ftoks)==0:
            continue
        if ftoks[-1].pos_ == 'ADP':
            ftoks = ftoks[:-1]
        ftoks = list(map(lambda token: token.text, ftoks))
        #print(ftoks)
        ent = ' '.join(ftoks)
        if (len(ent.split())==1 and ent != 'covid') or 'et al' in ent:
            continue
        pp_entities.append(ent)
    pp_entities = list(set(pp_entities))
    pp_entities = list(set(map(clean_text, pp_entities)))
    pp_entities = '|'.join(pp_entities)
    return pp_entities

#for rix, row in tqdm(test.iterrows(), total=test.shape[0]):
def generate_predictions(row_id):
    with open(f'{DATA_DIR}/{row_id}.json') as f:
        pub = json.load(f)
        pub_texts = list(map(lambda x: x['text'], pub))

    entities = []
    for excerpt in pub_texts:
        excerpt = re.sub('\s+', ' ', excerpt)
        for hc in hardcoded:
            if hc in excerpt.lower():
                entities.append(hc)
        if len(excerpt)>1000000:
            continue #TODO: Chunk here.
        doc = nlp(excerpt)
        ents = list(doc.ents)
        ents = list(filter(lambda x: x.label_ == 'ORG', ents))
        filtered_ents = []
        for ent in ents:
            txt = ent.sent.text.lower()
            kws = []
            for tok in ent.sent:
                if tok.text.lower().strip() in search_words:
                    kws.append(tok)
            if len(kws)>0:
                is_candidate = True in [kw.is_ancestor(etok) for kw in kws for etok in ent]
                if is_candidate:
                    filtered_ents.append(ent.text)
        entities.extend(filtered_ents)
    entities = list(set(entities))
    entities = post_process_results(entities)
    return entities

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

if __name__ == '__main__':
    test['PredictionString'] = process_map(generate_predictions, test_ids, chunksize=10)
    test.to_csv('submission.csv', index=False)