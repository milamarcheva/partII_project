import spacy
import string
from collections import OrderedDict
from transformers import GPT2Tokenizer

nlp = spacy.load('en_core_web_lg')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
eot_id = tokenizer.eos_token_id #used in narrative_flow_utils

erroneous_correct_chars= OrderedDict([('â€š', ','), ('â€˜',"'"), ('â€™',"'"), ('â€œ','"'), ('â€','"'), ('\n', ' ')])
numerical_entities = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}

def _debug_utf8_encodings(story):
    for erroneous_encoding in erroneous_correct_chars:
        correct_encoding = erroneous_correct_chars[erroneous_encoding]
        if erroneous_encoding in story:
            story = story.replace(erroneous_encoding, correct_encoding)
    return story


def clean_news_stories(df):
    cleaned_stories = []
    for story in df['content']:
        cleaned_stories.append(_debug_utf8_encodings(story))
    # cleaned_stories = [_debug_utf8_encodings(story) for story in df['content']]
    df['story'] = cleaned_stories


def get_docs(df):
    docs = [nlp(story) for story in df['story']]
    return docs


def _preprocess_token(t):
    tok = (t.lemma_).lower()
    if tok == '-pron-':
        tok = t.lower_
    return tok


def _preprocess_doc(doc):
    tokens = [t for t in doc if t.is_alpha]
    preprocessed_tokens = [_preprocess_token(t) for t in tokens]
    return preprocessed_tokens


def get_tokens_concreteness(docs):
    tokens = [_preprocess_doc(doc) for doc in docs]
    return tokens


def get_named_entities(docs):
  named_entities = [set(ne.text for ne in doc.ents if ne.label_ not in numerical_entities) for doc in docs]
  return named_entities


def get_sentences(docs):
    sentences = [[sentence.text for sentence in doc.sents] for doc in docs]
    return sentences


def _encode_list_of_inputs(sentences_list):
    encoded_inputs = [tokenizer.encode(sentence) for sentence in sentences_list]
    return encoded_inputs


def get_encoded_sentences(sentences):
    encoded_sentences = [_encode_list_of_inputs(sentences_list) for sentences_list in sentences]
    return encoded_sentences


def get_encoded_sentences_len(encoded_sentences):
    encoded_sentences_len = [[len(s) for s in story_sentences] for story_sentences in encoded_sentences]
    return encoded_sentences_len


def get_encoded_topics(topics):
    encoded_topics = _encode_list_of_inputs(topics)
    return encoded_topics

