import pandas as pd
import numpy as np
from ast import literal_eval
from narrative_flow_utils import get_inputs, get_stories_logprobs_bag, get_stories_logprobs_chain, get_narrative_flow
hc = pd.read_csv(r'partII_project/data/hc_analysis.csv')

# encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
# encoded_events = [literal_eval(l) for l in hc['encoded_events']]
# encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]
print('sent done')
logprobs_bag_s = [literal_eval(l) for l in hc['logprobs_bag_s']]
print('bag done')
logprobs_chain_s = [literal_eval(l) for l in hc['logprobs_chain_s']]
print('chain done')

# bag_inputs_s = get_inputs('bag', encoded_summaries, encoded_sentences)
# chain_inputs_s = get_inputs('chain', encoded_summaries, encoded_sentences)
# logprobs_bag_s = get_stories_logprobs_bag(encoded_summaries, bag_inputs_s)
# hc['logprobs_bag_s'] = logprobs_bag_s
# logprobs_chain_s = get_stories_logprobs_chain(encoded_summaries, encoded_sentences, chain_inputs_s)
# hc['logprobs_chain_s'] = logprobs_chain_s

hc['narrative_flow_s'] = get_narrative_flow(logprobs_bag_s, logprobs_chain_s, encoded_sentences_len)
hc['avg_narrative_flow_s'] = [np.average(nfs) for nfs in hc['narrative_flow_s']]


hc.to_csv(r'partII_project/data/hc_analysis.csv', index=False)
