import argparse
import pandas as pd
import numpy as np
from ast import literal_eval

def get_narrative_flow(logprobs_bag, logprobs_chain, encoded_sentences_len):
    number_of_sentences = len(logprobs_bag)
    narrative_flow = [0] * number_of_sentences
    for i in range(number_of_sentences):
        sentence_len = np.array(encoded_sentences_len[i])
        diff = np.array(logprobs_bag[i]) - np.array(logprobs_chain[i])
        narrative_flow[i] = -(diff/sentence_len)
    return narrative_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode = args['mode']

    if mode == 'hc':

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

    elif mode == 'news':
        df = pd.read_csv(r'../../data/news_analysis.csv')
        print(df.columns)
        encoded_sentences_len = [literal_eval(l) for l in df['encoded_sentences_len']]

        logprobs_bag_summaries = [literal_eval(l) for l in df['logprobs_bag_summaries']]
        logprobs_chain_summaries = [literal_eval(l) for l in df['logprobs_chain_summaries']]


        df['narrative_flow_summaries'] = get_narrative_flow(logprobs_bag_summaries, logprobs_chain_summaries,
                                                            encoded_sentences_len)
        print('calculated nf')

        df['avg_narrative_flow_summaries'] = [np.average(nfs) for nfs in df['narrative_flow_summaries']]
        print('calculated avg nf')

        df.to_csv(r'../../data/news_analysis.csv', index=False)
        print('wrote to hc')

