import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#eot_id = 50256
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
eot_id = tokenizer.eos_token_id #used in narrative_flow_utils

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)

def _wrap_with_eot(ids):
    wrapped_list = [eot_id] + ids + [eot_id]
    return wrapped_list


def _get_bag_input(encoded_topic, encoded_sentences_of_story):
    bm_input = [_wrap_with_eot((encoded_topic + encoded_s)) for encoded_s in encoded_sentences_of_story]
    return bm_input


def _get_chain_input(encoded_topic, encoded_sentences_of_story):
    cm_input = [_wrap_with_eot(encoded_topic + encoded_sentences_of_story[0])]
    for i in range(len(encoded_sentences_of_story) - 1):
        cm_input.append(_wrap_with_eot(encoded_sentences_of_story[i] + encoded_sentences_of_story[i + 1]))
    return cm_input

def _get_chain_input_all_prev_sents(encoded_topic, encoded_sentences_of_story):
    n = len(encoded_sentences_of_story)
    sentence_chain = encoded_topic.copy()
    cm_input = [0]*n
    for i in range(n):
        sentence_chain.extend(encoded_sentences_of_story[i])
        cm_input[i] = (_wrap_with_eot(sentence_chain))
    return cm_input

def get_inputs(model_name, encoded_topics, encoded_sentences):
    n = len(encoded_topics)
    if model_name == 'bag':
        inputs = [_get_bag_input(encoded_topics[i], encoded_sentences[i]) for i in range(n)]
    elif model_name == 'chain':
        inputs = [_get_chain_input(encoded_topics[i], encoded_sentences[i]) for i in range(n)]
    elif model_name == 'chain_all_prev_sents':
        inputs = [_get_chain_input_all_prev_sents(encoded_topics[i], encoded_sentences[i]) for i in range(n)]
    return inputs

def _get_sentence_logprobability(encoded_input, context_len):
    sentence_logprobability = 0
    offset = context_len + 1

    input_ids = torch.tensor([encoded_input])
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0][0]
        logprobs = torch.nn.functional.log_softmax(logits, 1)

    for i in range(offset, len(input_ids[0]) - 1):
        token_id = input_ids[0][i]
        logprob = logprobs[i - 1][token_id].item()
        sentence_logprobability += logprob

    return sentence_logprobability


def _get_story_sentences_logprobs_bag(encoded_topic, inputs_for_story):
    context_len = len(encoded_topic)
    story_sentences_logprobs = [_get_sentence_logprobability(encoded_input, context_len) for encoded_input in
                                inputs_for_story]
    return story_sentences_logprobs


def _get_story_sentences_logprobs_chain(history_type, encoded_topic, encoded_sentences_of_story, inputs_for_story):
    n = len(inputs_for_story)
    context_len = len(encoded_topic)
    story_sentences_logprobs = [0]*n
    story_sentences_logprobs[0] = _get_sentence_logprobability(inputs_for_story[0], context_len)
    if history_type == 'one_prev_sent':
        for i in range(1, n):
            context_len = len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_logprobability(inputs_for_story[i], context_len)
    elif history_type == 'all_prev_sents':
        for i in range(1,n):
            context_len += len(encoded_sentences_of_story[i - 1])
            story_sentences_logprobs[i] = _get_sentence_logprobability(inputs_for_story[i], context_len)
    return story_sentences_logprobs

def get_stories_logprobs_bag(encoded_topics, inputs):
    n = len(encoded_topics)
    #logprobs_bag = [_get_story_sentences_logprobs_bag(encoded_topics[i], inputs[i]) for i in range(n)]
    logprobs_bag = [0]*n
    #logprobs_bag = Parallel(n_jobs=2)(delayed(_get_story_sentences_logprobs_bag)(encoded_topics[i], inputs[i]) for i in range(n))
    for i in range(n):
        print('story #',i)
        logprobs_bag[i] = _get_story_sentences_logprobs_bag(encoded_topics[i], inputs[i])

    return logprobs_bag


def get_stories_logprobs_chain(history_type, encoded_topics, encoded_sentences, inputs):
    n = len(encoded_topics)
    #logprobs_chain = [_get_story_sentences_logprobs_chain(history_type, encoded_topics[i], encoded_sentences[i], inputs[i]) for i in range(n)]
    logprobs_chain = [0]*n
    #logprobs_chain = Parallel(n_jobs=4)(delayed(_get_story_sentences_logprobs_chain)(history_type,encoded_topics[i], encoded_sentences[i], inputs[i]) for i in range(n))
    for i in range(n):
        print('story #',i)
        #print(encoded_topics[i])
        logprobs_chain[i] = _get_story_sentences_logprobs_chain(history_type,encoded_topics[i], encoded_sentences[i], inputs[i])
    return logprobs_chain

def get_stories_logprobs_chain_start(history_type, encoded_topics, encoded_sentences, inputs, start, end):
    logprobs_chain = []
    for i in range(start, end):
        print('story #',i)
        logprobs_chain.append(_get_story_sentences_logprobs_chain(history_type,encoded_topics[i], encoded_sentences[i], inputs[i]))
    return logprobs_chain