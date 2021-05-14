import pandas as pd
from ast import literal_eval
from logprob_utils import get_inputs, get_stories_logprobs_chain

hc = pd.read_csv(r'partII_project/data/hc_analysis.csv')

encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
encoded_events = [literal_eval(l) for l in hc['encoded_events']]
encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]

print('start analysis')

chain_inputs_s = get_inputs('chain_all_prev_sents', encoded_summaries, encoded_sentences)
chain_inputs_e = get_inputs('chain_all_prev_sents', encoded_events, encoded_sentences)

print('chain inputs done')

logprobs_chain_s = get_stories_logprobs_chain('all_prev_sents', encoded_summaries, encoded_sentences, chain_inputs_s)
print('chain probs done')

hc['logprobs_chain_all_prev_sent_s'] = logprobs_chain_s
print('chain probs written to hc')

hc.to_csv(r'partII_project/data/hc_analysis.csv', index=False)
print('hc updated')