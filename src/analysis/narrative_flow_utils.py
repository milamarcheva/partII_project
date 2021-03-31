import numpy as np

def get_narrative_flow(logprobs_bag, logprobs_chain, encoded_sentences_len):
    n = len(logprobs_bag)
    narrative_flow = [0] * n
    for i in range(n):
        sentences_len = np.array(encoded_sentences_len[i])
        diff = np.array(logprobs_bag[i]) - np.array(logprobs_chain[i])
        narrative_flow[i] = -(diff / sentences_len)
    return narrative_flow
