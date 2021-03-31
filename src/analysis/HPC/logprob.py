import pandas as pd
from ast import literal_eval
from logprob_utils import get_inputs, get_stories_logprobs_bag, get_stories_logprobs_chain

hc = pd.read_csv(r'partII_project/data/hc_analysis.csv')

encoded_summaries = [literal_eval(l) for l in hc['encoded_summaries']]
encoded_events = [literal_eval(l) for l in hc['encoded_events']]
empty_context = []*len(encoded_events)
encoded_sentences = [literal_eval(l) for l in hc['encoded_sentences']]
encoded_sentences_len = [literal_eval(l) for l in hc['encoded_sentences_len']]

bag_inputs_summaries = get_inputs('bag', encoded_summaries, encoded_sentences)
bag_inputs_events = get_inputs('bag', encoded_events, encoded_sentences)
bag_inputs_empty = get_inputs('bag', empty_context, encoded_sentences)
print('bag inputs done')

chain_inputs_summaries = get_inputs('chain_all_prev_sents', encoded_summaries, encoded_sentences)
chain_inputs_events = get_inputs('chain_all_prev_sents', encoded_events, encoded_sentences)
chain_inputs_empty = get_inputs('chain_all_prev_sents', empty_context, encoded_sentences)
print('chain inputs done')


print('start analysis')

logprobs_bag_summaries = get_stories_logprobs_bag(encoded_summaries, bag_inputs_summaries)
logprobs_bag_events = get_stories_logprobs_bag(encoded_events, bag_inputs_events)
logprobs_bag_empty = get_stories_logprobs_bag(empty_context, bag_inputs_empty)
print('bag probs done')

logprobs_chain_summaries = get_stories_logprobs_chain('all_prev_sents', encoded_summaries, encoded_sentences, chain_inputs_summaries)
logprobs_chain_events = get_stories_logprobs_chain('all_prev_sents', encoded_events, encoded_sentences, chain_inputs_events)
logprobs_chain_empty= get_stories_logprobs_chain('all_prev_sents', empty_context, encoded_sentences, chain_inputs_empty)
print('chain probs done')

hc['logprobs_bag_summaries'] = logprobs_bag_summaries
hc['logprobs_bag_events'] = logprobs_bag_events
hc['logprobs_bag_empty'] = logprobs_bag_empty

hc['logprobs_chain_summaries'] = logprobs_chain_summaries
hc['logprobs_chain_events'] = logprobs_chain_events
hc['logprobs_chain_empty'] = logprobs_chain_empty


hc.to_csv(r'partII_project/data/hc_analysis.csv', index=False)
print('hc updated')