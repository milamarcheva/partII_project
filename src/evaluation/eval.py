import pandas as pd
import scipy
import numpy as np
from ast import literal_eval
hc_analysis = pd.read_csv(r'../../data/hc_analysis.csv')
hc_copy = hc_analysis.copy()

hc_eval = hc_copy[['id', 'story', 'label', 'concreteness', 'analytic', 'tone', 'i',	'posemo', 'negemo',	'cogproc']]

hc_eval_recalled = [hc_eval[hc_eval.label=='recalled']]
hc_eval_imagined = [hc_eval[hc_eval.label=='imagined']]

t_LIWC_analytic = scipy.stats.ttest_rel(hc_eval_imagined['analytic'], hc_eval_recalled['analytic'])
hc_eval['analytic'] = t_LIWC_analytic
hc_eval.to_csv(r'../../data/hc_eval.csv', index=False)