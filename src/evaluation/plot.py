import argparse
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode = args['mode']

    if mode == 'concreteness':
        hc_eval = pd.read_csv(r'../../data/hc_analysis.csv')
        hc = hc_eval.copy()

        hc_recalled = hc[hc.label == 'recalled']

        hc_imagined = hc[hc.label == 'imagined']

        ar_hc_recalled = list(hc_recalled['concreteness'])
        yr = np.random.normal(0.0, 0.1, len(ar_hc_recalled))
        ar_hc_imagined = list(hc_imagined['concreteness'])
        yi = np.random.normal(0.0, 0.1, len(ar_hc_imagined))

        fig, ax = plt.subplots(2)
        fig.tight_layout(pad=1)

        ax[0].scatter(ar_hc_imagined, yi, color='purple', alpha=1.0, s=0.2, label='imagined')
        ax[0].scatter(ar_hc_recalled, yr, color='cyan', alpha=1.0, s=0.2, label='recalled')
        ax[0].set_ylim([-1, 1])
        # ax[0].title.settext('Concreteness of each story, recalled in green and imagined in red')

        minval = min(ar_hc_recalled + ar_hc_imagined)
        maxval = max(ar_hc_recalled + ar_hc_imagined)
        ax[1].hist(ar_hc_recalled, bins=100, range=[minval, maxval], color='cyan', histtype='step')
        ax[1].hist(ar_hc_imagined, bins=100, range=[minval, maxval], color='purple', histtype='step')
        # ax[1].title.set_text('Concreteness of recalled in green and imagined in red')

        # recalled_imagined_ids = dict()
        # imagined_recalled_ids = dict()
        #
        #
        # for id_r in hc_recalled['id']:
        #     s = hc_recalled[hc_recalled.id==id_r].summary.item()
        #     ids_i = hc_imagined[hc_imagined.summary==s].id
        #     recalled_imagined_ids[id_r] = list(ids_i)
        #     for id_i in ids_i:
        #         imagined_recalled_ids[id_i] = id_r
        #
        #
        # def diff(duplicates):
        #
        #     if not duplicates:
        #         n = len(recalled_imagined_ids.keys())
        #         recalled = []
        #         imagined = []
        #         for id_r in recalled_imagined_ids:
        #             id_i = recalled_imagined_ids[id_r][0]
        #             recalled.append(hc_recalled[hc_recalled.id==id_r]['concreteness'].item())
        #             imagined.append(hc_imagined[hc_imagined.id==id_i]['concreteness'].item())
        #     else:
        #         imagined_ids = list(imagined_recalled_ids.keys())
        #         recalled = [0]*len(imagined_ids)
        #         imagined = [0]*len(imagined_ids)
        #         for i in range(len(imagined_ids)):
        #             id_i = imagined_ids[i]
        #             id_r = imagined_recalled_ids[id_i]
        #             imagined[i] = hc_imagined[hc_imagined.id==id_i]['concreteness'].item()
        #             recalled[i] = hc_recalled[hc_recalled.id==id_r]['concreteness'].item()
        #
        #     np_r = np.array(recalled)
        #     np_i = np.array(imagined)
        #
        #     return np.subtract(np_r, np_i)
        #
        # diff_dupl = diff(True)
        #
        # diff_nodupl = diff(False)
        #
        # ax[2].hist(diff_dupl,bins=100,range=[min(diff_dupl),max(diff_dupl)],color='r')
        # ax[2].title.set_text('Difference between recalled and imagined, with duplicates')
        #
        # ax[3].hist(diff_nodupl,bins=100,range=[min(diff_dupl),max(diff_dupl)],color='r')
        # ax[3].title.set_text('Difference between recalled and imagined, no duplicates')
        #ax[0].legend(prop={"size": 20})

        lgnd = ax[0].legend(scatterpoints=1, fontsize=20)
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]

        plt.show()

    if mode == 'nf':
        #Density plots of different topics used for narrative_flow
        hc_eval = pd.read_csv(r'../../data/hc_analysis.csv')
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

        hc_eval = pd.read_csv(r'../../data/hc_analysis.csv')
        hc = hc_eval.copy()
        hc_recalled = hc[hc.label == 'recalled']
        hc_imagined = hc[hc.label == 'imagined']

        bins = 30
        maxvalue = 1.5
        # maxvalue=10
        smoothing = 2

        avg_nf_summaries_i_srtd = [i for i in sorted(avg_nf_summaries_i) if i < maxvalue]
        print(avg_nf_summaries_i_srtd)
        avg_nf_summaries_r_srtd = [i for i in sorted(avg_nf_summaries_r) if i < maxvalue]

        mn = min(avg_nf_summaries_i_srtd[0], avg_nf_summaries_r_srtd[0])
        mx = max(avg_nf_summaries_i_srtd[-1], avg_nf_summaries_r_srtd[-1])

        ranges = np.linspace(mn, mx, bins + 1)
        binwidth = (mx - mn) / bins
        x = np.linspace(mn + binwidth / 2, mx - binwidth / 2, bins)

        bin_heights_i = [0] * bins
        bin_heights_r = [0] * bins
        bin_heights_comb = [0] * bins

        index = 0
        for i in range(len(avg_nf_summaries_i_srtd)):
            if avg_nf_summaries_i_srtd[i] > ranges[index + 1]:
                index += 1
            bin_heights_i[index] += 1
        density_heights_i = np.array(bin_heights_i) / (len(avg_nf_summaries_i_srtd) * binwidth)
        density_heights_i_smoothed = gaussian_filter1d(density_heights_i, sigma=smoothing)

        index = 0
        for i in range(len(avg_nf_summaries_r_srtd)):
            if avg_nf_summaries_r_srtd[i] > ranges[index + 1]:
                index += 1
            bin_heights_r[index] += 1
        density_heights_r = np.array(bin_heights_r) / (len(avg_nf_summaries_i_srtd) * binwidth)
        density_heights_r_smoothed = gaussian_filter1d(density_heights_r, sigma=smoothing)

        index = 0
        for i in range(len(avg_nf_summaries_r_srtd)):
            if avg_nf_summaries_r_srtd[i] > ranges[index + 1]:
                index += 1
            bin_heights_comb[index] += 1
        index = 0
        for i in range(len(avg_nf_summaries_i_srtd)):
            if avg_nf_summaries_i_srtd[i] > ranges[index + 1]:
                index += 1
            bin_heights_comb[index] -= 1
        density_heights_comb = np.array(bin_heights_comb) / (len(avg_nf_summaries_i_srtd) * binwidth)
        density_heights_comb_smoothed = gaussian_filter1d(density_heights_comb, sigma=smoothing)

        plt.hist(avg_nf_summaries_i,bins,range=[mn,mx])
        plt.plot(x,bin_heights_i)
        plt.plot(x,density_heights_i_smoothed, color='purple')
        plt.plot(x,bin_heights_r)
        plt.plot(x,density_heights_r_smoothed, color='cyan')

        plt.plot(x, density_heights_comb_smoothed, color='orange')
        plt.legend()

        plt.show()
########################################################################################################################
        #Density plot of recalled vs. imagined
        # hc_eval = pd.read_csv(r'../../data/hc_analysis.csv')
        #
        # hc = hc_eval.copy()
        # hc_recalled = hc[hc.label=='recalled']
        # hc_imagined = hc[hc.label=='imagined']
        #
        # # recalled
        # avg_nf_summaries_r = list(hc_recalled['avg_narrative_flow_summaries'])
        # print(len(avg_nf_summaries_r))
        # # imagined
        # avg_nf_summaries_i = list(hc_imagined['avg_narrative_flow_summaries'])
        # print(len(avg_nf_summaries_i))
        # print(avg_nf_summaries_i)
        #
        # bins = 30
        # maxvalue = 1.5
        # smoothing = 1
        #
        # avg_nf_summaries_i_srtd = [i for i in sorted(avg_nf_summaries_i) if i < maxvalue]
        # print(avg_nf_summaries_i_srtd)
        # avg_nf_summaries_r_srtd = [i for i in sorted(avg_nf_summaries_r) if i < maxvalue]
        #
        # mn = min(avg_nf_summaries_i_srtd[0], avg_nf_summaries_r_srtd[0])
        # mx = max(avg_nf_summaries_i_srtd[-1], avg_nf_summaries_r_srtd[-1])
        #
        # ranges = np.linspace(mn, mx, bins + 1)
        # binwidth = (mx - mn) / bins
        # x = np.linspace(mn + binwidth / 2, mx - binwidth / 2, bins)
        #
        # bin_heights_i = [0] * bins
        # bin_heights_r = [0] * bins
        #
        # index = 0
        # for i in range(len(avg_nf_summaries_i_srtd)):
        #     if avg_nf_summaries_i_srtd[i] > ranges[index + 1]:
        #         index += 1
        #     bin_heights_i[index] += 1
        # density_heights_i = np.array(bin_heights_i) / (len(avg_nf_summaries_i_srtd) * binwidth)
        # density_heights_i_smoothed = gaussian_filter1d(density_heights_i, sigma=smoothing)
        #
        # index = 0
        # for i in range(len(avg_nf_summaries_r_srtd)):
        #     if avg_nf_summaries_r_srtd[i] > ranges[index + 1]:
        #         index += 1
        #     bin_heights_r[index] += 1
        # density_heights_r = np.array(bin_heights_r) / (len(avg_nf_summaries_i_srtd) * binwidth)
        # density_heights_r_smoothed = gaussian_filter1d(density_heights_r, sigma=smoothing)
        #
        # # plt.hist(avg_nf_summaries_i,bins,range=[mn,mx])
        # # plt.plot(x,bin_heights_i)
        # plt.plot(x, density_heights_i_smoothed, color='purple', label = 'imagined')
        # # plt.plot(x,bin_heights_r)
        # plt.plot(x, density_heights_r_smoothed, color='cyan', label = 'recalled')
        # plt.legend(prop={"size": 30})
        # plt.tick_params(axis='x', labelsize=30)
        # plt.tick_params(axis='y', labelsize=30)
        # plt.show()

