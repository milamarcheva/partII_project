import pandas as pd

hc = pd.read_csv(r'../data/hippoCorpusV2.csv')

hc_trunc = hc[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
hc_trunc.columns = ['id', 'story', 'label', 'event', 'summary']

hc_Brysbaert = hc_trunc[['id', 'story', 'label']]
hc_LIWC = hc_trunc[['id', 'story', 'label']]
hc_GPT2 = hc_trunc.copy()

hc_Brysbaert.to_csv(r'../data/hc_Brysbaert.csv', index=False)
hc_LIWC.to_csv(r'../data/hc_LIWC.csv', index=False)
hc_GPT2.to_csv(r'../data/hc_GPT2.csv', index=False)
