import argparse
import pandas as pd
from src.preprocessing.preprocessing_utils import get_docs, get_tokens_concreteness, get_sentences, \
    get_encoded_sentences, get_encoded_topics, get_encoded_sentences_len


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode=args['mode']

    if mode == 'hc':
        hc = pd.read_csv(r'../../data/hc_analysis.csv')

        summaries = hc['summary']
        events = hc['event']

        docs = get_docs(hc)

        #concreteness
        hc['tokens_concreteness'] = get_tokens_concreteness(docs)

        #narrative flow
        sentences = get_sentences(docs)
        hc['encoded_sentences'] = get_encoded_sentences(sentences)
        hc['encoded_sentences_len'] = get_encoded_sentences_len(hc['encoded_sentences'])
        hc['encoded_summaries'] = get_encoded_topics(summaries)
        hc['encoded_events'] = get_encoded_topics(events)

        hc.to_csv(r'../../data/hc_analysis.csv', index=False)

    elif mode == 'news':

        print('news')
        news_df = pd.read_csv(r'../../data/news_analysis.csv')

        descriptions = news_df['meta_description']

        #spaCy
        docs = get_docs(news_df)

        # concreteness
        news_df['tokens_concreteness'] = get_tokens_concreteness(docs)

        # narrative flow
        sentences = get_sentences(docs)
        news_df['encoded_sentences'] = get_encoded_sentences(sentences)
        news_df['encoded_sentences_len'] = get_encoded_sentences_len(news_df['encoded_sentences'])
        news_df['encoded_summaries'] = get_encoded_topics(descriptions)

        news_df.to_csv(r'../../data/news_analysis.csv', index=False)


    elif mode == "test":
        import spacy

        nlp = spacy.load('en_core_web_lg')
        story = 'The day started nicely with a slice of freshly baked banana bread. Then I moved on to reading my emails and making a to-do list for the day. I wrote down a schedule including all of the tasks and also times allotted for breaks and sat down to work.'
        docs = [nlp(story)]

        correct_tokens = [['the', 'day', 'start', 'nicely', 'with', 'a', 'slice', 'of', 'freshly', 'bake', 'banana', 'bread']]
        correct_sentences = [['The day started nicely with a slice of freshly baked banana bread.',
                              'Then I moved on to reading my emails and making a to-do list for the day.',
                              'I wrote down a schedule including all of the tasks and also times allotted for breaks and sat down to work.']]
        # Test for get_tokens_concreteness
        assert (correct_tokens[0][:10]) == (get_tokens_concreteness(docs)[0][:10])

        # Test for get_sentences
        assert (correct_sentences) == (get_sentences(docs))
