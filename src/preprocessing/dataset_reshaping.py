import argparse
from ast import literal_eval
import chardet
import numpy as np
import pandas as pd
from collections import Counter
from src.preprocessing.preprocessing_utils import clean_news_stories, get_docs, get_named_entities

def chunk_filtering(chunk):
    n = len(chunk)
    sorted_chunk = chunk.sort_values(by='scraped_at', ascending = False).dropna()
    filtered_chunk = sorted_chunk.head(int(n*0.025))
    return filtered_chunk

def check_encoding():
    rawdata = open(r'../../../../Downloads/news_cleaned_2018_02_13.csv',"rb").readlines(1000000)
    for line in rawdata:
        print(chardet.detect(line))

def named_entity_filtering(news_df):
    fake_df = news_df[news_df.type =='fake']
    reliable_df = news_df[news_df.type =='reliable']

    n = len(fake_df)
    already_matched = set()
    news_df['paired_id'] = [-10]*len(news_df)

    i=1
    for id_fake in fake_df['id']:
        print(i,'/',n)
        i += 1
        ne_set_fake_string = fake_df.named_entities[fake_df.id==id_fake].item()
        if ne_set_fake_string == 'set()':
            continue
        else:
            ne_set_fake=literal_eval(ne_set_fake_string)
        for id_reliable in reliable_df['id']:
            ne_set_reliable_string = reliable_df.named_entities[reliable_df.id == id_reliable].item()
            if ne_set_reliable_string == 'set()':
                continue
            else:
                ne_set_reliable = literal_eval(ne_set_reliable_string)
            intersection = ne_set_fake.intersection(ne_set_reliable)
            if len(intersection)>=5 and id_reliable not in already_matched:
                already_matched.add(id_reliable)
                news_df.loc[(news_df.id == id_fake), 'paired_id'] = id_reliable
                news_df.loc[(news_df.id == id_reliable), 'paired_id'] = id_fake
                break

    print(len(news_df[news_df.paired_id!=-10]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', help='{hc, news, test}', required=True)

    args = vars(parser.parse_args())
    print(args)

    mode=args['mode']

    if mode == 'hc':
        hc = pd.read_csv(r'../../resources/hippoCorpusV2.csv')
        hc_copy = hc.copy()

        #leaving only the necessary columns and renaming them for ease of use
        hc_analysis = hc_copy[['AssignmentId', 'story', 'memType', 'mainEvent', 'summary']]
        hc_analysis.columns = ['id', 'story', 'label', 'event', 'summary']

        #removing the retold stories and resetting the index
        hc_analysis = hc_analysis[hc_analysis.label!='retold']
        hc_analysis.reset_index(drop='True', inplace=True)

        #cleaning non-paired stories

        recalled_summaries = set(hc_analysis[hc_analysis.label=='recalled']['summary'])
        imagined_summaries = set(hc_analysis[hc_analysis.label=='imagined']['summary'])
        intersection= recalled_summaries.intersection(imagined_summaries)

        #taking all summaries currently in hc_analysis
        summaries = hc_analysis['summary']

        #dropping the rows where the summary is not in the intersection
        for i in range(len(hc_analysis)):
            summary = summaries[i]
            if summary not in intersection:
                hc_analysis.drop(hc_analysis.index[hc_analysis['summary']==summary], inplace=True)

        hc_analysis.reset_index(drop='True', inplace=True)
        hc_analysis.to_csv(r'../../data/hc_analysis.csv', index=False)

        #Uncomment next section to look at various types (tuple, triple...) of recalled-imagined pairings
        # prompts = dict()
        # remaining_summaries = hc_analysis['summary']
        #
        # for summary in remaining_summaries:
        #     if summary in prompts:
        #         prompts[summary]+=1
        #     else:
        #         prompts[summary] = 1
        #
        # print(Counter(prompts.values()))

    elif mode == 'news':
        selected_cols = ['type', 'content', 'meta_description', 'scraped_at']

        # Uncomment next line to look at the encodings
        # check_encoding()

        # Since the original news file is 27GB, we need to use chunksize to process it in chunks, otherwise crash
        news_chunks = pd.read_csv(r'../../../../Downloads/news_cleaned_2018_02_13.csv', header=0, usecols = selected_cols, chunksize=100000, lineterminator='\n', encoding='utf_8_sig')

        #Processing the chunks
        news_chunks_list = []
        i = 1
        for chunk in news_chunks:
            fake_chunk = chunk[chunk.type=='fake']
            reliable_chunk = chunk[chunk.type=='reliable']

            filtered_fake = chunk_filtering(fake_chunk)
            filtered_reliable = chunk_filtering(reliable_chunk)

            news_chunks_list.append(filtered_fake)
            news_chunks_list.append(filtered_reliable)

            print(i, '/86')
            i+=1
        news_df = pd.concat(news_chunks_list).drop(columns=['scraped_at'])
        news_df.rename(columns={'type': 'label'}, inplace=True)

        #Creating ids
        news_df['id'] = np.arange(1, len(news_df)+1)

        #Cleaning the stories from wrong characters
        clean_news_stories(news_df)

        #Getting named entities
        docs = get_docs(news_df)
        news_df['named_entities'] = get_named_entities(news_df)

        #Named entity clean
        named_entity_filtering(news_df)

        #Dropping the stories without a pairing
        news_df = news_df[news_df.paired_id != -10]

        #Writing the file
        news_df.to_csv(r'../../data/news_analysis.csv', index=False)

    elif mode == 'test':
        story = 'VIEW GALLERY  If Maisie Williams being a bridesmaid at Sophie Turnerâ€™s wedding isnâ€™t the definition of #FriendshipGoals, then I donâ€™t know what is.  While speaking to Radio Times, Williams, who plays Ayra Stark on the hit HBO fantasy drama series Game Of Thrones, revealed that she is going to be a bridesmaid in Turnerâ€™s wedding, who is getting married to DNCE singer and Jonas Brother Joe Jonas. Turner, of course, plays Aryaâ€™s older sister Sansa on Game Of Thrones.  When asked if she has gotten official word from Sophie yet about participating in the ceremonies, Williams said: â€œOh, already got it. Yeah, itâ€™s very, very exciting. Itâ€™s kind of bizarre though.â€  However, Williams says that Turner and Jonas have yet to begin planning the wedding as she wants to wrap up production on the final season of Game Of Thrones first.  â€œWeâ€™re waiting â€™til this seasonâ€™s done until we get into it. But I think sheâ€™s already letting her little heart wander and imagine,â€ Maisie said.  2018 is shaping up to be a big year for Turner as she films the final season of Game Of Thrones, appears as the main character in the new X-Men movie Dark Phoenix, and gets married to Joe Jonas.  I only have one question, though: whereâ€™s my invite, Sophie? Iâ€™m sure Maisie needs a date, Iâ€™d be happy to oblige.'
        cleaned_story = 'VIEW GALLERY  If Maisie Williams being a bridesmaid at Sophie Turner"™s wedding isn"™t the definition of #FriendshipGoals, then I don"™t know what is.  While speaking to Radio Times, Williams, who plays Ayra Stark on the hit HBO fantasy drama series Game Of Thrones, revealed that she is going to be a bridesmaid in Turner"™s wedding, who is getting married to DNCE singer and Jonas Brother Joe Jonas. Turner, of course, plays Arya"™s older sister Sansa on Game Of Thrones.  When asked if she has gotten official word from Sophie yet about participating in the ceremonies, Williams said: "Oh, already got it. Yeah, it"™s very, very exciting. It"™s kind of bizarre though."  However, Williams says that Turner and Jonas have yet to begin planning the wedding as she wants to wrap up production on the final season of Game Of Thrones first.  "We"™re waiting "™til this season"™s done until we get into it. But I think she"™s already letting her little heart wander and imagine," Maisie said.  2018 is shaping up to be a big year for Turner as she films the final season of Game Of Thrones, appears as the main character in the new X-Men movie Dark Phoenix, and gets married to Joe Jonas.  I only have one question, though: where"™s my invite, Sophie? I"™m sure Maisie needs a date, I"™d be happy to oblige.'
        named_entities = {'Williams', 'Turner', 'Game Of Thrones', 's', 'Radio Times', 'Ayra Stark', 'HBO', 'Sansa', 'Jonas', 'Sophie', 'FriendshipGoals', 'Joe Jonas'}

        assert cleaned_story == clean_news_stories
