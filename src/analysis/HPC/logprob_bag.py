import pandas as pd
import numpy as np
from ast import literal_eval
from narrative_flow_utils import get_inputs, get_stories_logprobs_bag, get_stories_logprobs_chain, get_narrative_flow
hc = pd.read_csv(r'partII_project/data/hc_analysis.csv')

encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]
#logprobs_bag_s = [literal_eval(l) for l in hc['logprobs_bag_s']]

bag_inputs_s = get_inputs('bag', encoded_summaries, encoded_sentences)
print('inputs done')

logprobs_bag_s = get_stories_logprobs_bag(encoded_summaries, bag_inputs_s)
print('probs done')

hc['logprobs_bag_s'] = logprobs_bag_s

hc.to_csv(r'partII_project/data/hc_analysis.csv', index=False)

print('hc updated')