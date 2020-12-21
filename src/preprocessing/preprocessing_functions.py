import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg')

hc = pd.read_csv(r'..\..\data\hc_analysis.csv')

docs = [nlp(story) for story in hc['story']]

tokens = [[t for t in doc if t.is_alpha] for doc in docs]

def get_docs(hc):
    docs = [nlp(story) for story in hc['story']]
    return docs

def get_tokens(docs):
    tokens = [[t for t in doc ] for doc in docs]
    return tokens

def _preprocess_token(t):
    tok = (t.lemma_).lower()
    if tok == '-pron-':
        tok = t.lower_
    return tok

def _preprocess_doc(doc):
    tokens = [t for t in doc if t.is_alpha]
    preprocessed_tokens = [_preprocess_token(t) for t in tokens]
    return preprocessed_tokens

def get_tokens_concreteness(docs):
    tokens = [_preprocess_doc(doc) for doc in docs]
    return tokens

# def get_vocab_stories(stories):
#     vocab  = dict()
#     ind = 1
#     for story in stories:
#         for t in story:
#             if t not in vocab:
#                 vocab[t] = ind
#                 ind+=1
#     return vocab
#
# def encode_stories(vocab, stories):
#     encoded_stories = list()
#     for story in stories:
#         encoded_story = list()
#         for t in story:
#             encoded_story.append(vocab[t])
#         encoded_stories.append(encoded_story)
#     return encoded_stories
