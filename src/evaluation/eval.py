import pandas as pd
import scipy.stats
import numpy as np
from ast import literal_eval
from collections import Counter

hc_analysis = pd.read_csv(r'../../data/hc_analysis.csv')
hc_copy = hc_analysis.copy()

hc_eval = hc_copy[['id', 'label', 'summary', 'concreteness', 'analytic', 'tone', 'i', 'posemo', 'negemo','cogproc']]
#print(len(hc_eval))


hc_recalled = hc_eval[hc_eval.label=='recalled']
hc_imagined = hc_eval[hc_eval.label=='imagined']


# summaries = set(hc_eval['summary'])
#
# summaries_count = dict()
# summaries_new = hc_analysis['summary']
#
# for s in hc_eval['summary']:
#     if s in summaries_count :
#         summaries_count[s]+=1
#     else:
#         summaries_count[s] = 1
#
# print(Counter(summaries_count.values()))

# for i in d:
#     if d[i] == 10:
#         print(i)

# for s in hc_eval_recalled:
#     sum_r = hc_eval_recalled.summa

# hc_eval_recalled_paired = hc_eval_recalled[hc_eval_recalled.summary==hc_eval_imagined.summary]

#TODO improve efficiency by using id as an index and eliminating while loops

recalled_imagined_ids = dict()
imagined_recalled_ids = dict()


for id_r in hc_recalled['id']:
    s = hc_recalled[hc_recalled.id==id_r].summary.item()
    ids_i = hc_imagined[hc_imagined.summary==s].id
    recalled_imagined_ids[id_r] = list(ids_i)
    for id_i in ids_i:
        imagined_recalled_ids[id_i] = id_r


def paired_t_test(duplicates, metric):
    if not duplicates:
        #imagined_ids_first = [imagined_ids[0] for imagined_ids in list(recalled_imagined_ids.values())]
        # recalled = hc_recalled[metric]
        # imagined = hc_imagined[hc_imagined.id in imagined_ids_first][metric]
        n = len(recalled_imagined_ids.keys())
        recalled = []
        imagined = []
        for id_r in recalled_imagined_ids:
            id_i = recalled_imagined_ids[id_r][0]
            recalled.append(hc_recalled[hc_recalled.id==id_r][metric].item())
            imagined.append(hc_imagined[hc_imagined.id==id_i][metric].item())
    else:
        imagined_ids = list(imagined_recalled_ids.keys())
        recalled = [0]*len(imagined_ids)
        imagined = [0]*len(imagined_ids)
        for i in range(len(imagined_ids)):
            id_i = imagined_ids[i]
            id_r = imagined_recalled_ids[id_i]
            imagined[i] = hc_imagined[hc_imagined.id==id_i][metric].item()
            recalled[i] = hc_recalled[hc_recalled.id==id_r][metric].item()

    effect_size = np.mean(recalled) - np.mean(imagined)
    t_metric = scipy.stats.ttest_rel(imagined, recalled)
    return t_metric, effect_size

metrics = {'concreteness', 'analytic', 'tone', 'i',	'posemo', 'negemo',	'cogproc'}

for m in metrics:
    ttest1,  es1 = paired_t_test(True, m)
    tstat1, pvalue1 = ttest1
    ttest2, es2 = paired_t_test(False, m)
    tstat2, pvalue2 = ttest2

    #print(m+' with duplicates: ', paired_t_test(True, m))
    print(m+' with duplicates:  t-statistic: ', tstat1, ', p-value: ', pvalue1, ' effect size: ', es1)
    print(m+' without duplicates: t-statistic: ', tstat2, ', p-value: ', pvalue2, ' effect size: ', es2)



# t_LIWC_analytic = scipy.stats.ttest_rel(hc_eval_imagined.analytic, hc_eval_recalled.analytic)
# hc_eval['analytic'] = t_LIWC_analytic
hc_eval.to_csv(r'../../data/hc_eval.csv', index=False)