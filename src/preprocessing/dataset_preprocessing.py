import pandas as pd
from src.preprocessing.preprocessing_utils import get_docs, get_tokens_concreteness, get_sentences, \
    get_encoded_sentences, get_encoded_contexts, get_encoded_sentences_len

hc = pd.read_csv(r'../../data/hc_analysis.csv')

summaries = hc['summary']
events = hc['event']

docs = get_docs(hc)

#concreteness
#hc['tokens'] = get_tokens(docs)
hc['tokens_concreteness'] = get_tokens_concreteness(docs)

#narrative flow
sentences = get_sentences(docs)
hc['sentences'] = sentences
hc['encoded_sentences'] = get_encoded_sentences(sentences)
hc['encoded_sentences_len'] = get_encoded_sentences_len(hc['encoded_sentences'])
hc['encoded_summaries'] = get_encoded_contexts(summaries)
hc['encoded_events'] = get_encoded_contexts(events)

hc.to_csv(r'../../data/hc_analysis.csv', index=False)
