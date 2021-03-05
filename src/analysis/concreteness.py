import pandas as pd
hc = pd.read_csv(r'../../data/hc_analysis.csv')
concreteness_lexicon = pd.read_excel(r'../../data/concreteness.xlsx', engine='openpyxl')

# Create vocab from lexicon (set)
#lex = dict(zip(concreteness_lexicon.Word, zip(concreteness_lexicon.Bigram, concreteness_lexicon.Conc_M)))
lex = dict(zip(concreteness_lexicon.Word,  concreteness_lexicon.Conc_M))

#TODO Encode tokens; OOV token

# Look for token encoding in vocab
#deals with the bigrams
def get_concreteness_score(story):
    n = len(story)
    count = 0
    sum = 0
    i = 0

    while i < n:
        t1 = story[i]
        if i != n - 1:
            t1t2 = t1 + ' ' + story[i + 1]
            if t1t2 in lex:
                sum += lex[t1t2]
                i += 2
                count += 2
                continue
        if t1 in lex:
            sum += lex[t1]
            count += 1
        i += 1

    avg = sum / count
    return avg


story_tokens = ['the','day', 'start', 'nicely', 'with', 'a', 'slice','of','freshly','baked', 'banana','bread']
score_function = get_concreteness_score(story_tokens)
# print(score_function)
score_correct =35.47/12
#print(score_correct)
assert score_function == score_correct

# hc['concreteness'] = [get_concreteness_score(story) for story in hc['tokens_concreteness']]
#
# hc.to_csv(r'../../data/hc_analysis.csv', index=False)
