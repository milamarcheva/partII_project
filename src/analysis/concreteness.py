import pandas as pd

hc = pd.read_csv(r'../../data/hc_analysis.csv')
concreteness_lexicon = pd.read_excel(r'../../data/concreteness.xlsx', engine='openpyxl')

# Create vocab from lexicon (set)
lex = dict(zip(concreteness_lexicon.Word, concreteness_lexicon.Conc_M))
# Encode tokens; OOV token

# Look for token encoding in vocab
#TODO deal with the bigrams
def get_concreteness_score(story):
    count = 0
    sum = 0
    for t in story:
        if t in lex:
            sum+=lex[t]
            count+=1
    avg = sum/count
    return avg

hc['concreteness'] = [get_concreteness_score(story) for story in hc['tokens_concreteness']]

hc.to_csv(r'../../data/hc_analysis.csv', index=False)
