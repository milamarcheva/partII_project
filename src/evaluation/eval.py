import pandas as pd
import scipy.stats
import numpy as np
from ast import literal_eval
hc_analysis = pd.read_csv(r'../../data/hc_analysis.csv')
hc_copy = hc_analysis.copy()

hc_eval = hc_copy[['id', 'story', 'label', 'summary', 'concreteness', 'analytic', 'tone', 'i',	'posemo', 'negemo',	'cogproc']]
print(len(hc_eval))


hc_eval_recalled = pd.DataFrame(hc_eval[hc_eval.label=='recalled'])
hc_eval_imagined = pd.DataFrame(hc_eval[hc_eval.label=='imagined'])

# for s in hc_eval_recalled:
#     sum_r = hc_eval_recalled.summa

# hc_eval_recalled_paired = hc_eval_recalled[hc_eval_recalled.summary==hc_eval_imagined.summary]

print(hc_eval_recalled['summary'])
print(hc_eval_imagined['summary'])
t_LIWC_analytic = scipy.stats.ttest_rel(hc_eval_imagined.analytic, hc_eval_recalled.analytic)
hc_eval['analytic'] = t_LIWC_analytic
hc_eval.to_csv(r'../../data/hc_eval.csv', index=False)