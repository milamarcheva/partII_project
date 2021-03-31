import torch
import numpy as np
from ast import literal_eval
from joblib import Parallel, delayed
from transformers import GPT2LMHeadModel

eot_id = 50256

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)
#model.eval()


def _wrap_with_eot(ids):
    wrapped_list = [eot_id] + ids + [eot_id]
    return wrapped_list


def _get_bag_input(encoded_context, encoded_sentences_of_story):
    bm_input = [_wrap_with_eot((encoded_context + encoded_s)) for encoded_s in encoded_sentences_of_story]
    return bm_input


def _get_chain_input(encoded_context, encoded_sentences_of_story):
    cm_input = [_wrap_with_eot(encoded_context + encoded_sentences_of_story[0])]
    for i in range(len(encoded_sentences_of_story) - 1):
        cm_input.append(_wrap_with_eot(encoded_sentences_of_story[i] + encoded_sentences_of_story[i + 1]))
    return cm_input

def _get_chain_input_all_prev_sents(encoded_context, encoded_sentences_of_story):
    n = len(encoded_sentences_of_story)
    #print(n)
    sentence_chain = encoded_context.copy()
    cm_input = [0]*n
    for i in range(n):
        #print(i)
        sentence_chain.extend(encoded_sentences_of_story[i])
        cm_input[i] = (_wrap_with_eot(sentence_chain))
    return cm_input

def get_inputs(model_name, encoded_contexts, encoded_sentences):
    n = len(encoded_contexts)
    if model_name == 'bag':
        inputs = [_get_bag_input(encoded_contexts[i], encoded_sentences[i]) for i in range(n)]
    elif model_name == 'chain':
        inputs = [_get_chain_input(encoded_contexts[i], encoded_sentences[i]) for i in range(n)]
    elif model_name == 'chain_all_prev_sents':
        inputs = [_get_chain_input_all_prev_sents(encoded_contexts[i], encoded_sentences[i]) for i in range(n)]
    return inputs

def _get_sentence_log_probability(encoded_input, context_len):
    sentence_log_probability = 0
    input_ids = torch.tensor([encoded_input])
    input_ids = input_ids.to(device)

    offset = context_len + 1

    with torch.no_grad():
        #outputs = model(input_ids).to(device)
        outputs = model(input_ids)
        logits = outputs[0][0]
        log_probs = torch.nn.functional.log_softmax(logits, 1)

        for i in range(offset, len(input_ids[0]) - 1):
            token_id = input_ids[0][i]
            log_prob = log_probs[i - 1][token_id].item()
            sentence_log_probability += log_prob

        return sentence_log_probability


def _get_story_sentences_logprobs_bag(encoded_context, inputs_for_story):
    #print('getting story prob')
    context_len = len(encoded_context)
    #print('context_len: ', context_len)
    #print('inputs_len: ', len(inputs_for_story))
    story_sentences_logprobs = [_get_sentence_log_probability(encoded_input, context_len) for encoded_input in
                                inputs_for_story]
    #print(story_sentences_logprobs)
    return story_sentences_logprobs


def _get_story_sentences_logprobs_chain(context_type, encoded_context, encoded_sentences_of_story, inputs_for_story):
    n = len(inputs_for_story)
    context_len = len(encoded_context)
    #print(encoded_context)
    #print('context_len initial: ', context_len)
    story_sentences_logprobs = [0]*n
    story_sentences_logprobs[0] = _get_sentence_log_probability(inputs_for_story[0], context_len)
    if context_type == 'one_prev_sent':
        for i in range(1, n):
            context_len = len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_log_probability(inputs_for_story[i], context_len)
    elif context_type == 'all_prev_sent':
        for i in range(1,n):
            #print(inputs_for_story[i])
            #print(context_len)
            context_len += len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_log_probability(inputs_for_story[i], context_len)
    #print(story_sentences_logprobs)
    return story_sentences_logprobs

def get_stories_logprobs_bag(encoded_contexts, inputs):
    n = len(encoded_contexts)
    #logprobs_bag = [_get_story_sentences_logprobs_bag(encoded_contexts[i], inputs[i]) for i in range(n)]
    logprobs_bag = [0]*n
    #logprobs_bag = Parallel(n_jobs=2)(delayed(_get_story_sentences_logprobs_bag)(encoded_contexts[i], inputs[i]) for i in range(n))
    for i in range(n):
        print('story #',i)
        logprobs_bag[i] = _get_story_sentences_logprobs_bag(encoded_contexts[i], inputs[i])

    return logprobs_bag


def get_stories_logprobs_chain(context_type, encoded_contexts, encoded_sentences, inputs):
    n = len(encoded_contexts)
    #logprobs_chain = [_get_story_sentences_logprobs_chain(context_type, encoded_contexts[i], encoded_sentences[i], inputs[i]) for i in range(n)]
    logprobs_chain = [0]*n
    #logprobs_chain = Parallel(n_jobs=4)(delayed(_get_story_sentences_logprobs_chain)(context_type,encoded_contexts[i], encoded_sentences[i], inputs[i]) for i in range(n))
    for i in range(n):
        print('story #',i)
        #print(encoded_contexts[i])
        logprobs_chain[i] = _get_story_sentences_logprobs_chain(context_type,encoded_contexts[i], encoded_sentences[i], inputs[i])
    return logprobs_chain

def get_stories_logprobs_chain_start(context_type, encoded_contexts, encoded_sentences, inputs, start, end):
    logprobs_chain = []
    for i in range(start, end):
        print('story #',i)
        logprobs_chain.append(_get_story_sentences_logprobs_chain(context_type,encoded_contexts[i], encoded_sentences[i], inputs[i]))
    return logprobs_chain