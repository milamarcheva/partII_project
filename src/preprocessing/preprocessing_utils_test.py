import spacy
from src.preprocessing.preprocessing_utils import get_tokens_concreteness, get_sentences

nlp = spacy.load('en_core_web_lg')
story = 'The day started nicely with a slice of freshly baked banana bread. Then I moved on to reading my emails and making a to-do list for the day. I wrote down a schedule including all of the tasks and also times allotted for breaks and sat down to work.'
docs = [nlp(story)]
correct_tokens = [['the','day', 'start', 'nicely', 'with', 'a', 'slice','of','freshly','bake', 'banana','bread']]
correct_sentences = [['The day started nicely with a slice of freshly baked banana bread.', 'Then I moved on to reading my emails and making a to-do list for the day.',  'I wrote down a schedule including all of the tasks and also times allotted for breaks and sat down to work.']]
#Test for get_tokens_concreteness
assert (correct_tokens[0][:10]) == (get_tokens_concreteness(docs)[0][:10])

#Test for get_sentences
assert (correct_sentences) == (get_sentences(docs))

#Test for get_encoded_sentences

#Test for get_encoded_sentences_len

#Test for get_encoded_contexts


