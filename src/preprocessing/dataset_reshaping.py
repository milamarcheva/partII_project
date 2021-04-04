import pandas as pd
from collections import Counter

hc = pd.read_csv(r'../../data/hippoCorpusV2.csv')
hc_copy = hc.copy()

#leaving only the necessary columns and renaming them for ease of use
hc_analysis = hc_copy[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
hc_analysis.columns = ['id', 'story', 'label', 'event', 'summary']

#removing the retold stories and resetting the index
hc_analysis = hc_analysis[hc_analysis.label!='retold']
hc_analysis.reset_index(drop='True', inplace=True)

#cleaning non-paired stories

recalled_summaries = set(hc_analysis[hc_analysis.label=='recalled']['summary'])
imagined_summaries = set(hc_analysis[hc_analysis.label=='imagined']['summary'])
intersection= recalled_summaries.intersection(imagined_summaries)

#taking all summaries currently in hc_analysis
summaries = hc_analysis['summary']

#dropping the rows where the summary is not in the intersection
for i in range(len(hc_analysis)):
    summary = summaries[i]
    if summary not in intersection:
        hc_analysis.drop(hc_analysis.index[hc_analysis['summary']==summary], inplace=True)

hc_analysis.reset_index(drop='True', inplace=True)

#looking at various types of recalled-imagined pairings
prompts = dict()
remaining_summaries = hc_analysis['summary']

for summary in remaining_summaries:
    if summary in prompts:
        prompts[summary]+=1
    else:
        prompts[summary] = 1

print(Counter(prompts.values()))

hc_analysis.to_csv(r'../../data/hc_analysis.csv', index=False)

