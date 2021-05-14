import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from ast import literal_eval

#read Brysbaert's lexiocn
concreteness_lexicon = pd.read_excel(r'../../resources/concreteness.xlsx', engine='openpyxl')

# Create vocab from lexicon (set)
lex = dict(zip(concreteness_lexicon.Word, concreteness_lexicon.Conc_M))

# how many words per story can be found in the lexicon (a bigram is counted as a single word in count but in n it is counted as 2 words)
percentage_in_lex = []
percentage_bigrams = []

# Look for token encoding in vocab
# deals with the bigrams
def get_concreteness_score(story):
    n = len(story)  # 'lexiographically defined' length
    count = 0  # 'semantic' count
    sum = 0
    i = 0
    bigram_count = 0

    while i < n:
        t1 = story[i]
        if i != n - 1:
            t1t2 = t1 + ' ' + story[i + 1]
            if t1t2 in lex:
                bigram_count += 1
                sum += lex[t1t2]
                i += 2
                count += 1
                continue
        if t1 in lex:
            sum += lex[t1]
            count += 1
        i += 1

    # percentage of words in lexicon (semantic count)
    percentage_in_lex.append(count / (n - bigram_count))
    # percentage of bigrams over all semantic words in a story
    percentage_bigrams.append(bigram_count / (n - bigram_count))
    avg = sum / count
    return avg

def print_number_of_bigrams():
    counter = 0
    for p in percentage_bigrams:
        if p != 0:
            counter += 1

    print('Number of stories where at least one bigram was detected: ', counter, ' out of ', len(percentage_bigrams), ' total stories.')

def get_plots():
    #Percentages histogram
    mu = np.mean(percentage_in_lex)
    sigma = np.std(percentage_in_lex)
    mx = max(percentage_in_lex)
    mn = min(percentage_in_lex)

    print('mean = ',mu,'std = ',sigma,'max = ',mx,'min = ',mn)

    x=np.linspace(mn,mx,100)
    bins = 50
    #plt.plot(x,len(percentage_in_lex)*((mx-mn)/bins)*stats.norm.pdf(x,mu,sigma), color='midnightblue')
    plt.hist(percentage_in_lex,bins, color = 'thistle' )
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.show()

    # #Bigram count
    mu = np.mean(percentage_bigrams)
    sigma = np.std(percentage_bigrams)
    mx = max(percentage_bigrams)
    mn = min(percentage_bigrams)

    print('mean = ',mu,'std = ',sigma,'max = ',mx,'min = ',mn)

    x=np.linspace(mn,mx,100)
    bins = 5
    #plt.plot(x,len(percentage_bigrams)*((mx-mn)/bins)*stats.norm.pdf(x,mu,sigma), color='midnightblue')
    plt.hist(percentage_bigrams,bins, color = 'thistle' )
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode=args['mode']

    if mode == 'hc':
        hc = pd.read_csv(r'../../data/hc_analysis.csv')
        hc['concreteness'] = [get_concreteness_score(literal_eval(story)) for story in hc['tokens_concreteness']]
        get_plots()
        # hc.to_csv(r'../../data/hc_analysis.csv', index=False)
        print_number_of_bigrams()

    elif mode == 'news':
        news_df = pd.read_csv(r'../../data/news_analysis.csv')
        news_df['concreteness'] = [get_concreteness_score(literal_eval(story)) for story in news_df['tokens_concreteness']]
        news_df.to_csv(r'../../data/news_analysis.csv', index=False)
        print_number_of_bigrams()

    elif mode == 'test':
        story_tokens = ['the','day', 'start', 'nicely', 'with', 'a', 'slice','of','freshly','baked', 'banana','bread']
        score_function = get_concreteness_score(story_tokens)
        score_correct =35.47/12
        assert score_function == score_correct