import torch
import numpy as np
from transformers import GPT2LMHeadModel
from src.preprocessing.preprocessing_utils import eot_id

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


def _wrap_with_eot(ids):
    wrapped_list = [eot_id] + ids + [eot_id]
    return wrapped_list


def _get_bag_input(encoded_context, encoded_sentences_of_story):
    bm_input = [_wrap_with_eot(encoded_context + encoded_s) for encoded_s in encoded_sentences_of_story]
    return bm_input


def _get_chain_input(encoded_context, encoded_sentences_of_story):
    cm_input = [_wrap_with_eot(encoded_context + encoded_sentences_of_story[0])]
    for i in range(len(encoded_sentences_of_story) - 1):
        cm_input.append(_wrap_with_eot(encoded_sentences_of_story[i] + encoded_sentences_of_story[i + 1]))
    return cm_input


def _get_chain_input_all_prev_sents(encoded_context, encoded_sentences_of_story):
    n = len(encoded_sentences_of_story)
    sentence_chain = encoded_context
    cm_input = []*n
    for i in range(0,n):
        sentence_chain.extend(encoded_sentences_of_story[i])
        cm_input[i](_wrap_with_eot(sentence_chain))
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
    offset = context_len + 1

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0][0]
        log_probs = torch.nn.functional.log_softmax(logits, 1)

        for i in range(offset, len(input_ids[0]) - 1):
            token_id = input_ids[0][i]
            log_prob = log_probs[i - 1][token_id].item()
            sentence_log_probability += log_prob

        return sentence_log_probability


def _get_story_sentences_logprobs_bag(encoded_context, inputs_for_story):
    context_len = len(encoded_context)
    story_sentences_logprobs = [_get_sentence_log_probability(encoded_input, context_len) for encoded_input in
                                inputs_for_story]
    return story_sentences_logprobs


def _get_story_sentences_logprobs_chain(context_type, encoded_context, encoded_sentences_of_story, inputs_for_story):
    n = len(inputs_for_story)
    context_len = len(encoded_context)
    story_sentences_logprobs = [0]*n
    story_sentences_logprobs[0] = [_get_sentence_log_probability(inputs_for_story[0], context_len)]
    if context_type == 'one_prev_sent':
        for i in range(1, n):
            context_len = len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_log_probability(inputs_for_story[i], context_len)
    elif context_type == 'all_prev_sent':
        for i in range(1,n):
            context_len += len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_log_probability(inputs_for_story[i], context_len)
    return story_sentences_logprobs


def get_stories_logprobs_bag(encoded_contexts, inputs):
    n = len(encoded_contexts)
    #logprobs_bag = [_get_story_sentences_logprobs_bag(encoded_contexts[i], inputs[i]) for i in range(n)]
    logprobs_bag = [0]*n
    for i in range(n):
        print('story #',i)
        logprobs_bag[i] = _get_story_sentences_logprobs_bag(encoded_contexts[i], inputs[i])
    return logprobs_bag


def get_stories_logprobs_chain(context_type, encoded_contexts, encoded_sentences, inputs):
    n = len(encoded_contexts)
    logprobs_chain = [_get_story_sentences_logprobs_chain(context_type, encoded_contexts[i], encoded_sentences[i], inputs[i]) for i in
                      range(n)]
    return logprobs_chain


def get_narrative_flow(logprobs_bag, logprobs_chain, encoded_sentences_len):
    n = len(logprobs_bag)
    narrative_flow = [0] * n
    for i in range(n):
        sentences_len = np.array(encoded_sentences_len[i])
        diff = np.array(logprobs_bag[i]) - np.array(logprobs_chain[i])
        narrative_flow[i] = -(diff / sentences_len)
    return narrative_flow
