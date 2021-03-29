import spacy
from transformers import GPT2Tokenizer

nlp = spacy.load('en_core_web_lg')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
eot_id = tokenizer.eos_token_id #used in narrative_flow_utils


def get_docs(hc):
    docs = [nlp(story) for story in hc['story']]
    return docs


def get_tokens(docs):
    tokens = [[t for t in doc] for doc in docs]
    return tokens


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


def get_encoded_contexts(contexts):
    encoded_contexts = _encode_list_of_inputs(contexts)
    return encoded_contexts
