import pandas as pd
from ast import literal_eval

hc = pd.read_csv(r'../data/hc_analysis.csv')

encoded_summaries = [literal_eval(s) for s in hc['encoded_summaries']]
encoded_events = hc['encoded_events']
encoded_sentences = hc['encoded_sentences']
encoded_sentences_len = hc['encoded_sentences_len']

print(type(encoded_summaries[0]))
print(type(encoded_sentences_len[0]))
print(type(encoded_sentences[0]))


encoded_sentences_len = hc['encoded_sentences_len']