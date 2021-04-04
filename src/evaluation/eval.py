import pandas as pd
import scipy.stats
import numpy as np
from math import sqrt

hc_analysis = pd.read_csv(r'../../data/hc_analysis.csv')
hc_copy = hc_analysis.copy()

hc_eval = hc_copy[['id', 'label', 'summary', 'concreteness', 'analytic', 'tone', 'i', 'posemo', 'negemo','cogproc', 'avg_narrative_flow_summaries', 'avg_narrative_flow_events', 'avg_narrative_flow_empty']]

hc_recalled = hc_eval[hc_eval.label=='recalled']
hc_imagined = hc_eval[hc_eval.label=='imagined']

recalled_imagined_ids = dict()
imagined_recalled_ids = dict()

for id_r in hc_recalled['id']:
    s = hc_recalled[hc_recalled.id==id_r].summary.item()
    ids_i = hc_imagined[hc_imagined.summary==s].id
    recalled_imagined_ids[id_r] = list(ids_i)
    for id_i in ids_i:
        imagined_recalled_ids[id_i] = id_r


def paired_t_test(duplicates, metric):
    recalled = []
    imagined = []
    if not duplicates:
        for id_r in recalled_imagined_ids:
            id_i = recalled_imagined_ids[id_r][0]
            recalled.append(hc_recalled[hc_recalled.id==id_r][metric].item())
            imagined.append(hc_imagined[hc_imagined.id==id_i][metric].item())
    else:
        for id_i in imagined_recalled_ids:
            id_r = imagined_recalled_ids[id_i]
            imagined.append(hc_imagined[hc_imagined.id==id_i][metric].item())
            recalled.append(hc_recalled[hc_recalled.id==id_r][metric].item())

    size = len(imagined)
    pooled_std = sqrt((size - 1)*(np.var(imagined)+np.var(recalled))/(2*size - 2))
    effect_size = (np.mean(imagined) - np.mean(recalled))/pooled_std
    t_stat, p_val = scipy.stats.ttest_rel(imagined, recalled)
    return t_stat, p_val, effect_size

metrics = ['concreteness', 'analytic', 'tone', 'i',	'posemo', 'negemo',	'cogproc', 'avg_narrative_flow_summaries', 'avg_narrative_flow_events', 'avg_narrative_flow_empty']
metric_scores = dict()

for m in metrics:
    tstat1, pvalue1,  es1 = paired_t_test(True, m)
    #tstat1, pvalue1 = ttest1
    direction1 = 'imagined'
    if tstat1<0:
        direction1 = 'recalled'
    tstat2, pvalue2, es2 = paired_t_test(False, m)
    #tstat2, pvalue2 = ttest2
    direction2 = 'imagined'
    if tstat2 < 0:
        direction2 = 'recalled'
    scores1 = [tstat1, pvalue1, es1, direction1]
    scores2 = [tstat2, pvalue2, es2, direction2]
    #print(m+' with duplicates: ', paired_t_test(True, m))
    print(m+' with duplicates:  t-statistic: ', tstat1, ', p-value: ', pvalue1, ' effect size: ', es1, ' direction: ', direction1)
    print(m+' without duplicates: t-statistic: ', tstat2, ', p-value: ', pvalue2,' effect size: ', es2, ' direction: ', direction2)
    #print(m+' with duplicates: ', ', p-value: ', pvalue1, ' effect size: ', es1, ' direction: ', direction1)
    metric_scores[m+' with duplicates']= scores1
    metric_scores[m+' without duplicates'] = scores2


hc_metrics = pd.DataFrame(metric_scores,  index=['t-statistic', 'p-value', 'effect size', 'direction'])

hc_eval.to_csv(r'../../data/hc_eval.csv', index=False)
hc_metrics.to_csv(r'../../data/hc_metrics.csv')