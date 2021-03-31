import numpy as np

def get_narrative_flow(logprobs_bag, logprobs_chain, encoded_sentences_len):
    number_of_sentences = len(logprobs_bag)
    narrative_flow = [0] * number_of_sentences
    for i in range(number_of_sentences):
        sentence_len = np.array(encoded_sentences_len[i])
        diff = np.array(logprobs_bag[i]) - np.array(logprobs_chain[i])
        narrative_flow[i] = -(diff/sentence_len)
    return narrative_flow
