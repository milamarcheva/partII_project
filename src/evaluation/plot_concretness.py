import pandas as pd
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

hc_eval = pd.read_csv(r'../../data/hc_eval.csv')
hc = hc_eval.copy()

hc_recalled = hc[hc.label=='recalled']

# print(type(hc_recalled))
hc_imagined = hc[hc.label=='imagined']

ar_hc_recalled = list(hc_recalled['concreteness'])
yr = np.random.normal(0.0,0.1,len(ar_hc_recalled))
ar_hc_imagined = list(hc_imagined['concreteness'])
yi = np.random.normal(0.0,0.1,len(ar_hc_imagined))


fig,ax=plt.subplots(4)
fig.tight_layout(pad=1)

ax[0].scatter(ar_hc_recalled,yr,color='g',alpha=1.0,s=0.2)
ax[0].scatter(ar_hc_imagined,yi,color='r',alpha=1.0,s=0.2)
ax[0].set_ylim([-1,1])
ax[0].title.set_text('Concreteness of each story, recalled in green and imagined in red')

minval = min(ar_hc_recalled+ar_hc_imagined)
maxval = max(ar_hc_recalled+ar_hc_imagined)
ax[1].hist(ar_hc_recalled,bins=100,range=[minval,maxval],color='g', histtype='step' )
ax[1].hist(ar_hc_imagined,bins=100,range=[minval,maxval],color='r', histtype='step' )
ax[1].title.set_text('Concreteness of recalled in green and imagined in red')



recalled_imagined_ids = dict()
imagined_recalled_ids = dict()


for id_r in hc_recalled['id']:
    s = hc_recalled[hc_recalled.id==id_r].summary.item()
    ids_i = hc_imagined[hc_imagined.summary==s].id
    recalled_imagined_ids[id_r] = list(ids_i)
    for id_i in ids_i:
        imagined_recalled_ids[id_i] = id_r


def diff(duplicates):

    if not duplicates:
        n = len(recalled_imagined_ids.keys())
        recalled = []
        imagined = []
        for id_r in recalled_imagined_ids:
            id_i = recalled_imagined_ids[id_r][0]
            recalled.append(hc_recalled[hc_recalled.id==id_r]['concreteness'].item())
            imagined.append(hc_imagined[hc_imagined.id==id_i]['concreteness'].item())
    else:
        imagined_ids = list(imagined_recalled_ids.keys())
        recalled = [0]*len(imagined_ids)
        imagined = [0]*len(imagined_ids)
        for i in range(len(imagined_ids)):
            id_i = imagined_ids[i]
            id_r = imagined_recalled_ids[id_i]
            imagined[i] = hc_imagined[hc_imagined.id==id_i]['concreteness'].item()
            recalled[i] = hc_recalled[hc_recalled.id==id_r]['concreteness'].item()

    np_r = np.array(recalled)
    np_i = np.array(imagined)

    return np.subtract(np_r, np_i)

diff_dupl = diff(True)

diff_nodupl = diff(False)

ax[2].hist(diff_dupl,bins=100,range=[min(diff_dupl),max(diff_dupl)],color='r')
ax[2].title.set_text('Difference between recalled and imagined, with duplicates')

ax[3].hist(diff_nodupl,bins=100,range=[min(diff_dupl),max(diff_dupl)],color='r')
ax[3].title.set_text('Difference between recalled and imagined, no duplicates')


plt.show()


