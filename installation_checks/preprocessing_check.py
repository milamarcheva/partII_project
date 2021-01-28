import spacy
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#
# # nlp = spacy.load('en_core_web_lg')
# #
# # doc = nlp('Hello world. I am Mila.')
# #
# # for t in doc:
# #     print(t.lemma_)
# #
# # modulename = 'spacy'
# # if modulename not in sys.modules:
# #     print ('You have not imported the {} module'.format(modulename))
# # if modulename in sys.modules:
# #     print ('You have imported the {} module'.format(modulename))
# # print('Mila')
#
# # from transformers import pipeline, set_seed
# # generator = pipeline('text-generation', model='gpt2')
# # set_seed(42)
# # seq = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
# # print(seq)
#
# # from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# # import tensorflow as tf
# #
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = TFGPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
# #
# # inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
# # outputs = model(inputs)
# # logits = outputs.logits
# # # logits = outputs[0][0]
# # probs  =tf.nn.softmax(logits)
# # # probs = torch.softmax(logits, 1)
# # for index in range(1,6):
# #     token_id = input_ids[0][index]
# #     probability = probs[index - 1][token_id].item()
# #     print(f"Probability for the token \"{tokenizer.decode(token_id.item())}\" is {probability}")
# #     print("\n")
# # #
# #     #

pretrained_weights = 'gpt2'

model = GPT2LMHeadModel.from_pretrained(pretrained_weights)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)

def show_probabilities(INPUT_TEXT):
    input_ids = torch.tensor([tokenizer.encode(INPUT_TEXT)])

    with torch.no_grad():
        index = 0
        outputs = model(input_ids=input_ids)
        logits = outputs[0][0]
        probs = torch.softmax(logits, 1)
        for index in range(0, len(input_ids[0])):
            token_id = input_ids[0][index]
            probability = probs[index - 1][token_id].item()
            print(f"Probability for the token \"{tokenizer.decode(token_id.item())}\" is {probability}")
    print("\n")


show_probabilities('To be or not to be <|endoftext|>')
show_probabilities('<|startoftext|> To be or not to be <|endoftext|>')
show_probabilities('<|endoftext|> To be or not to be <|endoftext|>')
# show_probabilities('Hello world is so wierd?')