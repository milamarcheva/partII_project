import spacy
import pandas as pd

nlp = spacy.load('en_core_web_lg')
hc_trunc = pd.read_csv(r'../data/hc_trunc.csv')

hc = hc_trunc.copy()
hc = hc.drop(columns=['summary', 'event'])


def preprocess_token(t):
    tok = (t.lemma_).lower()
    if tok == '-pron-':
        tok = t.lower_
    return tok


def preprocess_doc(doc):
    tokens = [t for t in doc if t.is_alpha]
    preprocessed_tokens = [preprocess_token(t) for t in tokens]
    return preprocessed_tokens


hc['docs'] = [nlp(story) for story in hc['story']]
hc['tokens'] = [preprocess_doc(doc) for doc in hc['docs']]

hc.to_csv(r'../data/hc_concreteness.csv', index=False)
