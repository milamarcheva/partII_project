import argparse

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from scipy.ndimage import gaussian_filter1d


def plot_topics(hc_eval, recalled_imagined_ids):
    imagined_minus_recalled_summaries = []
    imagined_minus_recalled_events = []
    imagined_minus_recalled_empty= []

    for id_r in recalled_imagined_ids:
        id_i= recalled_imagined_ids[id_r][0]
        imagined_minus_recalled_summaries.append(-hc_eval[hc_eval.id==id_r]['avg_narrative_flow_summaries'].item() + hc_eval[hc_eval.id==id_i]['avg_narrative_flow_summaries'].item())
        imagined_minus_recalled_events.append(-hc_eval[hc_eval.id==id_r]['avg_narrative_flow_events'].item() + hc_eval[hc_eval.id==id_i]['avg_narrative_flow_events'].item())
        imagined_minus_recalled_empty.append(-hc_eval[hc_eval.id==id_r]['avg_narrative_flow_empty'].item() + hc_eval[hc_eval.id==id_i]['avg_narrative_flow_empty'].item())

    avg_nf_summaries_combined = imagined_minus_recalled_summaries
    avg_nf_events_combined = imagined_minus_recalled_events
    avg_nf_empty_combined = imagined_minus_recalled_empty

    bins = 30
    minvalue = -2
    maxvalue = 2
    # maxvalue=10
    smoothing = 2

    avg_nf_summaries_combined_srtd = [i for i in sorted(avg_nf_summaries_combined) if i < maxvalue and i > minvalue]
    avg_nf_events_combined_srtd = [i for i in sorted(avg_nf_events_combined) if i < maxvalue and i > minvalue]
    avg_nf_empty_combined_srtd = [i for i in sorted(avg_nf_empty_combined) if i < maxvalue and i > minvalue]

    mn = min(avg_nf_summaries_combined_srtd[0], avg_nf_events_combined_srtd[0], avg_nf_empty_combined_srtd[0])
    mx = max(avg_nf_summaries_combined_srtd[-1], avg_nf_events_combined_srtd[-1], avg_nf_empty_combined_srtd[-1])

    ranges = np.linspace(mn, mx, bins + 1)
    binwidth = (mx - mn) / bins
    x = np.linspace(mn + binwidth / 2, mx - binwidth / 2, bins)

    bin_heights_comb_summaries_combined = [0] * bins
    bin_heights_comb_events_combined = [0] * bins
    bin_heights_comb_empty_combined = [0] * bins

    index = 0
    for i in range(len(avg_nf_summaries_combined_srtd)):
        if avg_nf_summaries_combined_srtd[i] > ranges[index + 1]:
            index += 1
        bin_heights_comb_summaries_combined[index] += 1
    density_heights_summaries_combined = np.array(bin_heights_comb_summaries_combined) / (
                len(avg_nf_summaries_combined_srtd) * binwidth)
    density_heights_summaries_combined_smoothed = gaussian_filter1d(density_heights_summaries_combined, sigma=smoothing)

    index = 0
    for i in range(len(avg_nf_events_combined_srtd)):
        if avg_nf_events_combined_srtd[i] > ranges[index + 1]:
            index += 1
        bin_heights_comb_events_combined[index] += 1
    density_heights_events_combined = np.array(bin_heights_comb_events_combined) / (
                len(avg_nf_events_combined_srtd) * binwidth)
    density_heights_events_combined_smoothed = gaussian_filter1d(density_heights_events_combined, sigma=smoothing)

    index = 0
    for i in range(len(avg_nf_empty_combined_srtd)):
        if avg_nf_empty_combined_srtd[i] > ranges[index + 1]:
            index += 1
        bin_heights_comb_empty_combined[index] += 1
    density_heights_empty_combined = np.array(bin_heights_comb_empty_combined) / (
                len(avg_nf_empty_combined_srtd) * binwidth)
    density_heights_empty_combined_smoothed = gaussian_filter1d(density_heights_empty_combined, sigma=smoothing)

    plt.plot(x, density_heights_summaries_combined_smoothed, color='purple', label='summaries')
    plt.plot(x, density_heights_events_combined_smoothed, color='cyan', label='events')
    plt.plot(x, density_heights_empty_combined_smoothed, color='orange', label='empty')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode=args['mode']

    if mode == 'hc':
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
        hc_metrics.to_csv(r'../../data/hc_metrics.csv')

        plot_topics(hc_eval, recalled_imagined_ids)

    elif mode == 'news':
        news_analysis = pd.read_csv(r'../../data/news_analysis.csv')
        news_copy = news_analysis.copy()
        news_eval = news_copy[
            ['id', 'label', 'paired_id', 'concreteness', 'analytic', 'tone', 'i', 'posemo', 'negemo', 'cogproc','avg_narrative_flow_summaries']]
            #'avg_narrative_flow_summaries', 'avg_narrative_flow_events', 'avg_narrative_flow_empty']]

        news_recalled = news_eval[news_eval.label == 'reliable']
        news_imagined = news_eval[news_eval.label == 'fake']

        recalled_imagined_ids = dict()

        for id_r in news_recalled['id']:
            recalled_imagined_ids[id_r] = news_imagined[news_imagined.paired_id == id_r].id.item()

        def paired_t_test(metric):
            recalled = []
            imagined = []
            for id_r in recalled_imagined_ids:
                id_i = recalled_imagined_ids[id_r]
                if metric == 'avg_narrative_flow_summaries':
                    if news_recalled[news_recalled.id == id_r][metric].item()==1000000 or news_imagined[news_imagined.id == id_i][metric].item()==1000000:
                        continue
                recalled.append(news_recalled[news_recalled.id == id_r][metric].item())
                imagined.append(news_imagined[news_imagined.id == id_i][metric].item())
            size = len(imagined)
            pooled_std = sqrt((size - 1) * (np.var(imagined) + np.var(recalled)) / (2 * size - 2))
            effect_size = (np.mean(imagined) - np.mean(recalled)) / pooled_std
            t_stat, p_val = scipy.stats.ttest_rel(imagined, recalled)
            return t_stat, p_val, effect_size


        metrics = ['concreteness', 'analytic', 'tone', 'i', 'posemo', 'negemo', 'cogproc', 'avg_narrative_flow_summaries']
        #, 'avg_narrative_flow_events', 'avg_narrative_flow_empty']

        metric_scores = dict()

        for m in metrics:
            tstat1, pvalue1, es1 = paired_t_test(m)
            direction1 = 'imagined'
            if tstat1 < 0:
                direction1 = 'recalled'
            scores1 = [tstat1, pvalue1, es1, direction1]
            print(m + ' :  t-statistic: ', tstat1, ', p-value: ', pvalue1, ' effect size: ', es1,
                  ' direction: ', direction1)

            metric_scores[m + ' with duplicates'] = scores1

        news_metrics = pd.DataFrame(metric_scores, index=['t-statistic', 'p-value', 'effect size', 'direction'])
        news_metrics.to_csv(r'../../data/news_metrics.csv')