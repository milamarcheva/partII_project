import argparse
import pandas as pd

def add_liwc_results(df, liwc_results):
    df['analytic'] = liwc_results['Analytic']
    df['tone'] = liwc_results['Tone']
    df['i'] = liwc_results['i']
    df['posemo'] = liwc_results['posemo']
    df['negemo'] = liwc_results['negemo']
    df['cogproc'] = liwc_results['cogproc']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode=args['mode']

    if mode == 'hc':
        hc = pd.read_csv(r'../../data/hc_analysis.csv')
        liwc_results = pd.read_csv(r'../../resources/LIWC_results.csv')
        add_liwc_results(hc, liwc_results)
        hc.to_csv(r'../../data/hc_analysis.csv', index=False)

    elif mode == 'news':
        news_df = pd.read_csv(r'../../data/news_analysis.csv')
        liwc_results = pd.read_csv(r'../../resources/LIWC_results_news.csv')
        add_liwc_results(news_df, liwc_results)
        news_df.to_csv(r'../../data/news_analysis.csv', index=False)