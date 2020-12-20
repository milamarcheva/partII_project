import pandas as pd

hc = pd.read_csv(r'../data/hippoCorpusV2.csv')

hc_trunc = hc[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
hc_trunc.columns = ['id', 'story', 'label', 'event', 'summary']
hc_trunc = hc_trunc[hc_trunc.label!='retold']
hc_trunc.to_csv(r'../data/hc_trunc.csv', index=False)

