import pandas as pd

hc = pd.read_csv(r'../../data/hippoCorpusV2.csv')
hc_copy = hc.copy()

hc_analysis = hc_copy[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
hc_analysis.columns = ['id', 'story', 'label', 'event', 'summary']
hc_analysis = hc_analysis[hc_analysis.label!='retold']

hc_analysis.to_csv(r'../../data/hc_analysis.csv', index=False)