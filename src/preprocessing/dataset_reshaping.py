import pandas as pd
from collections import Counter

hc = pd.read_csv(r'../../data/hippoCorpusV2.csv')
hc_copy = hc.copy()

hc_analysis = hc_copy[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
hc_analysis.columns = ['id', 'story', 'label', 'event', 'summary']

hc_analysis = hc_analysis[hc_analysis.label!='retold']
hc_analysis.reset_index(drop='True', inplace=True)
#clean non-paired stories
recalled_summaries = set(hc_analysis[hc_analysis.label=='recalled']['summary'])
imagined_summaries = set(hc_analysis[hc_analysis.label=='imagined']['summary'])
intersection= recalled_summaries.intersection(imagined_summaries)


print(len(hc_analysis))
print(hc_analysis)
print(hc_analysis.index)

summaries = hc_analysis['summary']
print(summaries)

#for i in hc_analysis.index[::-1]:
for i in range(len(hc_analysis)):
    #summary = hc_analysis['summary']
    summary = summaries[i]
    #label = hc_analysis['label'].index[i]
    #if (label == 'imagined' and summary not in recalled_summaries) or (label == 'recalled' and summary not in imagined_summaries):
    if summary not in intersection:
        #print(summary)
        hc_analysis.drop(hc_analysis.index[hc_analysis['summary']==summary], inplace=True)

hc_analysis.reset_index(drop='True', inplace=True)

print(hc_analysis)


d = dict()
summaries_new = hc_analysis['summary']

for s in summaries_new:
    if s in d:
        # = d[s]
        d[s]+=1
    else:
        d[s] = 1

print(Counter(d.values()))
for i in d:
    if d[i] == 10:
        print(i)


hc_analysis.to_csv(r'../../data/hc_analysis.csv', index=False)

