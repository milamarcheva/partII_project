import pandas as pd
import numpy as np
from ast import literal_eval
from src.analysis.narrative_flow_utils import get_narrative_flow

hc = pd.read_csv(r'../../data/hc_analysis.csv')

encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]

logprobs_bag_summaries = [literal_eval(l) for l in hc['logprobs_bag_summaries']]
logprobs_chain_summaries = [literal_eval(l) for l in  hc['logprobs_chain_summaries']]

logprobs_bag_events =  [literal_eval(l) for l in hc['logprobs_bag_events']]
logprobs_chain_events =  [literal_eval(l) for l in hc['logprobs_chain_events']]

logprobs_bag_empty =  [literal_eval(l) for l in hc['logprobs_bag_empty']]
logprobs_chain_empty =  [literal_eval(l) for l in hc['logprobs_chain_empty']]

#Deprecated routine when the chain probabilities were calculated by 6 separate files
#merge logprobs_chain_s
# logprobs_chain_s_as_strings = []
# for i in range(1,7):
#     print(i)
#     number = str(i*1000)
#     lpchain_i_df = pd.read_csv(r'../../data/lpchain/lpchain_'+ number +'.csv')
#     lpchain_i_list = list(lpchain_i_df['logprobs'])
#     logprobs_chain_s_as_strings+=lpchain_i_list
#logprobs_chain_s = [literal_eval(l) for l in logprobs_chain_s_as_strings]
# hc['logprobs_chain_s'] = logprobs_chain_s
# print('added chain to hc')

hc['narrative_flow_summaries'] = get_narrative_flow(logprobs_bag_summaries, logprobs_chain_summaries, encoded_sentences_len)
hc['narrative_flow_events'] = get_narrative_flow(logprobs_bag_events, logprobs_chain_events, encoded_sentences_len)
hc['narrative_flow_empty'] = get_narrative_flow(logprobs_bag_empty, logprobs_chain_empty, encoded_sentences_len)
print('calculated nf')

hc['avg_narrative_flow_summaries'] = [np.average(nfs) for nfs in hc['narrative_flow_summaries']]
hc['avg_narrative_flow_events'] = [np.average(nfs) for nfs in hc['narrative_flow_events']]
hc['avg_narrative_flow_empty'] = [np.average(nfs) for nfs in hc['narrative_flow_empty']]
print('calculated avg nf')

hc.to_csv(r'../../data/hc_analysis.csv', index=False)
print('wrote to hc')