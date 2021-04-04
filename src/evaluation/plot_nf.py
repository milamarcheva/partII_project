import pandas as pd
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage .filters import gaussian_filter1d

hc_eval = pd.read_csv(r'../../data/hc_eval.csv')
hc = hc_eval.copy()

hc_recalled = hc[hc.label=='recalled']
hc_imagined = hc[hc.label=='imagined']

#recalled
avg_nf_summaries_r = list(hc_recalled['avg_narrative_flow_summaries'])
avg_nf_events_r = list(hc_recalled['avg_narrative_flow_events'])
avg_nf_empty_r = list(hc_recalled['avg_narrative_flow_empty'])

#imagined
avg_nf_summaries_i = list(hc_imagined['avg_narrative_flow_summaries'])
avg_nf_events_i = list(hc_imagined['avg_narrative_flow_events'])
avg_nf_empty_i = list(hc_imagined['avg_narrative_flow_empty'])

bins = 30
maxvalue = 1.5
smoothing = 1

avg_nf_summaries_i_srtd = [i for i in sorted(avg_nf_summaries_i) if i < maxvalue]
avg_nf_summaries_r_srtd = [i for i in sorted(avg_nf_summaries_r) if i < maxvalue]
mn = min(avg_nf_summaries_i_srtd[0],avg_nf_summaries_r_srtd[0])
mx = max(avg_nf_summaries_i_srtd[-1],avg_nf_summaries_r_srtd[-1])

ranges = np.linspace(mn,mx,bins+1)
binwidth = (mx-mn)/bins
x = np.linspace(mn+binwidth/2,mx-binwidth/2,bins)

bin_heights_i = [0]*bins
bin_heights_r = [0]*bins

index=0
for i in range(len(avg_nf_summaries_i_srtd)):
    if avg_nf_summaries_i_srtd[i] > ranges[index+1]:
        index+=1
    bin_heights_i[index] += 1
density_heights_i = np.array(bin_heights_i)/(len(avg_nf_summaries_i_srtd)*binwidth)
density_heights_i_smoothed = gaussian_filter1d(density_heights_i,sigma=smoothing)

index=0
for i in range(len(avg_nf_summaries_r_srtd)):
    if avg_nf_summaries_r_srtd[i] > ranges[index+1]:
        index+=1
    bin_heights_r[index] += 1
density_heights_r = np.array(bin_heights_r)/(len(avg_nf_summaries_i_srtd)*binwidth)
density_heights_r_smoothed = gaussian_filter1d(density_heights_r,sigma=smoothing)


# plt.hist(avg_nf_summaries_i,bins,range=[mn,mx])
#plt.plot(x,bin_heights_i)
plt.plot(x,density_heights_i_smoothed, color='purple')
#plt.plot(x,bin_heights_r)
plt.plot(x,density_heights_r_smoothed, color='cyan')

plt.show()