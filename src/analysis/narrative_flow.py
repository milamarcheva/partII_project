import pandas as pd
import numpy as np
from ast import literal_eval
from src.analysis.narrative_flow_utils import get_narrative_flow

hc = pd.read_csv(r'../../data/hc_analysis.csv')

encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]
logprobs_bag_s = [literal_eval(l) for l in hc['logprobs_bag_s']]

#merge logprobs_chain_s
logprobs_chain_s_as_strings = []
for i in range(1,7):
    print(i)
    number = str(i*1000)
    lpchain_i_df = pd.read_csv(r'../../data/lpchain/lpchain_'+ number +'.csv')
    lpchain_i_list = list(lpchain_i_df['logprobs'])
    logprobs_chain_s_as_strings+=lpchain_i_list

logprobs_chain_s = [literal_eval(l) for l in logprobs_chain_s_as_strings]
hc['logprobs_chain_s'] = logprobs_chain_s
print('added chain to hc')
hc['narrative_flow_s'] = get_narrative_flow(logprobs_bag_s, logprobs_chain_s, encoded_sentences_len)
print('calculated nf')
hc['avg_narrative_flow_s'] = [np.average(nfs) for nfs in hc['narrative_flow_s']]
print('calculated avg nf')

hc.to_csv(r'../../data/hc_analysis.csv', index=False)
print('wrote to hc')

# HPC code:
# encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
# encoded_events = [literal_eval(l) for l in hc['encoded_events']]
# encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
# bag_inputs_s = get_inputs('bag', encoded_summaries, encoded_sentences)
# chain_inputs_s = get_inputs('chain', encoded_summaries, encoded_sentences)
# logprobs_bag_s = get_stories_logprobs_bag(encoded_summaries, bag_inputs_s)
# hc['logprobs_bag_s'] = logprobs_bag_s
# logprobs_chain_s = get_stories_logprobs_chain(encoded_summaries, encoded_sentences, chain_inputs_s)
# hc['logprobs_chain_s'] = logprobs_chain_s
# bag_inputs_e = get_inputs('bag', encoded_events, encoded_sentences)
# chain_inputs_e = get_inputs('chain', encoded_events, encoded_sentences)
# logprobs_bag_e = get_stories_logprobs_bag(encoded_events, bag_inputs_e)
# logprobs_chain_e = get_stories_logprobs_chain(encoded_events, chain_inputs_e)
# hc['narrative_flow_events'] = get_narrative_flow(logprobs_bag_e, logprobs_chain_e, encoded_sentences_len)