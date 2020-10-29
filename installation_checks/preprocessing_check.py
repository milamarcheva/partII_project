import spacy

nlp = spacy.load('en_core_web_lg')

doc = nlp('Hello world. I am Mila.')

for t in doc:
    print(t.lemma_)