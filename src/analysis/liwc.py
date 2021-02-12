import pandas as pd

hc = pd.read_csv(r'../../data/hc_analysis.csv')
liwc_results = pd.read_csv(r'../../data/LIWC_results.csv')

hc['analytic'] = liwc_results['Analytic']
hc['tone'] = liwc_results['Tone']
hc['i'] = liwc_results['i']
hc['posemo'] = liwc_results['posemo']
hc['negemo'] = liwc_results['negemo']
hc['cogproc'] = liwc_results['cogproc']

hc.to_csv(r'../../data/hc_analysis.csv', index=False)