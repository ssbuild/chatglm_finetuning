# @Time    : 2023/3/25 18:36
# @Author  : tk
import random
import typing
from enum import Enum
import numpy as np

from aigc_zoo.model_zoo.chatglm.llm_model import ChatGLMTokenizer


class DataStrategy(Enum):
    truncation = 1
    singlesliding = 2
    doublesliding = 3



class TokenIdsFinal:
    @classmethod
    def process(cls,input_ids: typing.List,sptoken,max_seq_length,tokenizer):
        ctxlen = input_ids.index(sptoken[-1])
        mask_position = ctxlen - 1
        labels = [-100] * ctxlen + input_ids[mask_position + 1:]

        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        ctxlen = np.asarray(ctxlen, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))

        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
            'ctxlen': ctxlen
        }
        return d







def build_template_chatglm(query, answer = None, history=None):
    prompt = ''
    sid = 0
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += query if sid == 0 else "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_chatglm2(query, answer = None, history=None):
    prompt = ''
    sid = 1
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt


def build_template_default(query, answer = None, history=None):
    prompt = ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None, history=None):
    prompt = ''
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    if history is not None:
        for q,a in history:
            prompt += "{}{}{}{}".format(tok_ins,q,tok_res,a)

    prompt += "{}{}{}".format(tok_ins, query, tok_res)
    if answer is not None:
        prompt += answer
    return prompt


# 切换模版
build_template = build_template_chatglm

#对prompt 截断
class TokenTruncation:

    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List,ensure_answer_min_length=1):
        assert ensure_answer_min_length > 0
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False) + [config.eos_token_id]

            a_max_len = max_seq_length-len(sptoken)-ensure_answer_min_length-1
            input_ids_qa = a_ids[-a_max_len:] + sptoken + b_ids + [config.eos_token_id]
            pos = 0
            while pos < len(input_ids_qa):
                if sptoken[0] in input_ids_qa[pos:pos + max_seq_length]:
                    val = input_ids_qa[pos:pos + max_seq_length][-1]
                    if val == sptoken[-1]:
                        input_ids = input_ids_qa[pos+1:pos + max_seq_length+1]
                        pos += max_seq_length + 1
                    elif val == sptoken[0]:
                        input_ids = input_ids_qa[pos + 2:pos + max_seq_length + 2]
                        pos += max_seq_length + 2
                    else:
                        input_ids = input_ids_qa[pos:pos + max_seq_length]
                        pos += max_seq_length
                else:
                    input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
                    pos += max_seq_length - 2

                d = TokenIdsFinal.process(input_ids,sptoken,max_seq_length,tokenizer)
                ds.append(d)
        return ds

#对prompt sliding
class TokenSingleSliding:

    @classmethod
    def process(cls,tokenizer: ChatGLMTokenizer,config,examples,max_seq_length,sptoken: typing.List,sliding_size,p=1):
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False) + [config.eos_token_id]

            input_ids_qa = a_ids + sptoken + b_ids + [config.eos_token_id]
            a_length = len(a_ids)
            pos = 0

            assert sliding_size < max_seq_length - 2
            while pos < len(input_ids_qa):
                if pos + max_seq_length <= a_length:
                    input_ids = input_ids_qa[pos:pos + max_seq_length-2]
                    if p > 0:
                        input_ids = input_ids[0:-p] + sptoken + input_ids[-p:]
                    else:
                        p = random.randint(0,max_seq_length-2)
                        input_ids = input_ids[0:p] + sptoken + input_ids[p:]

                    pos += sliding_size
                elif sptoken[0] in input_ids_qa[pos:pos + max_seq_length]:
                    val = input_ids_qa[pos:pos + max_seq_length][-1]
                    if val == sptoken[-1]:
                        input_ids = input_ids_qa[pos + 1:pos + max_seq_length + 1]
                        pos += max_seq_length + 1
                    elif val == sptoken[0]:
                        input_ids = input_ids_qa[pos + 2:pos + max_seq_length + 2]
                        pos += max_seq_length + 2
                    else:
                        input_ids = input_ids_qa[pos:pos + max_seq_length]
                        pos += max_seq_length
                else:
                    input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
                    pos += max_seq_length - 2

                d = TokenIdsFinal.process(input_ids, sptoken, max_seq_length, tokenizer)
                ds.append(d)
            return ds

# 对prompt sliding
class TokenDoubleSliding:
    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List, sliding_size,p=1):
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False) + [config.eos_token_id]

            input_ids_qa = a_ids + sptoken + b_ids + [config.eos_token_id]
            a_length = len(a_ids)
            pos = 0

            assert sliding_size < max_seq_length - 2
            while pos < len(input_ids_qa):
                if pos + max_seq_length <= a_length:
                    input_ids = input_ids_qa[pos:pos + max_seq_length - 2]
                    if p > 0:
                        input_ids = input_ids[0:-p] + sptoken + input_ids[-p:]
                    else:
                        p = random.randint(0, max_seq_length - 2)
                        input_ids = input_ids[0:p] + sptoken + input_ids[p:]
                    pos += sliding_size
                elif sptoken[0] in input_ids_qa[pos:pos + max_seq_length]:
                    val = input_ids_qa[pos:pos + max_seq_length][-1]
                    if val == sptoken[-1]:
                        input_ids = input_ids_qa[pos + 1:pos + max_seq_length + 1]
                        pos += max_seq_length + 1
                    elif val == sptoken[0]:
                        input_ids = input_ids_qa[pos + 2:pos + max_seq_length + 2]
                        pos += max_seq_length + 2
                    else:
                        input_ids = input_ids_qa[pos:pos + max_seq_length]
                        pos += sliding_size
                else:
                    input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
                    pos += sliding_size

                d = TokenIdsFinal.process(input_ids, sptoken, max_seq_length, tokenizer)
                ds.append(d)
        return ds