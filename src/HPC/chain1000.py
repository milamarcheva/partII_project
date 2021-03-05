import pandas as pd
from ast import literal_eval
from narrative_flow_utils import get_inputs, get_stories_logprobs_chain_start
hc = pd.read_csv(r'partII_project/data/hc_analysis.csv')

encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]

print('start anlayis')

chain_inputs_s = get_inputs('chain_all_prev_sents', encoded_summaries, encoded_sentences)
print('chain inputs done')

logprobs_chain_s = get_stories_logprobs_chain_start('all_prev_sent', encoded_summaries, encoded_sentences, chain_inputs_s, 0, 1000)
print('chain probs done')

df = pd.DataFrame({'logprobs':logprobs_chain_s})

df.to_csv(r'partII_project/data/lpchain_1000.csv', index=False)
print('probs written to csv')