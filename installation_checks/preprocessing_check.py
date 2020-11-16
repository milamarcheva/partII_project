import spacy
import sys

from transformers import GPT2Model

nlp = spacy.load('en_core_web_lg')

doc = nlp('Hello world. I am Mila.')

for t in doc:
    print(t.lemma_)

modulename = 'spacy'
if modulename not in sys.modules:
    print ('You have not imported the {} module'.format(modulename))
if modulename in sys.modules:
    print ('You have imported the {} module'.format(modulename))
print('Mila')

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
seq = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
print(seq)
