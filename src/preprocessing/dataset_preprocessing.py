import pandas as pd
from src.preprocessing.preprocessing_functions import get_docs, get_tokens, get_tokens_concreteness


hc = pd.read_csv(r'../../data/hc_analysis.csv')

docs = get_docs(hc)

#tokens = get_tokens(docs)
#tokens_concreteness = get_tokens_concreteness(docs)

hc['tokens'] = get_tokens(docs)
hc['tokens_concreteness'] = get_tokens_concreteness(docs)


#TODO hc['tokens_nflow'] = get_tokens_nflow(docs)

hc.to_csv(r'../../data/hc_analysis.csv', index=False)
