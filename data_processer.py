# @Time    : 2023/3/25 18:36
# @Author  : tk
import random
import typing
from enum import Enum
import numpy as np
from models import ChatGLMTokenizer


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


#对prompt 截断
class TokenTruncation:

    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, a_ids, b_ids, max_seq_length, sptoken: typing.List,ensure_answer_min_length=1):
        ds = []

        assert ensure_answer_min_length > 0
        input_ids_qa = a_ids[:max_seq_length-len(sptoken)-ensure_answer_min_length] + sptoken + b_ids + [config.eos_token_id] * 2
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
    def process(cls,tokenizer: ChatGLMTokenizer,config,a_ids,b_ids,max_seq_length,sptoken: typing.List,sliding_size,p=1):
        ds = []
        input_ids_qa = a_ids + sptoken + b_ids + [config.eos_token_id] * 2
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
    def process(cls, tokenizer: ChatGLMTokenizer,config, a_ids, b_ids, max_seq_length, sptoken: typing.List, sliding_size,
                p=1):
        ds = []
        input_ids_qa = a_ids + sptoken + b_ids + [config.eos_token_id] * 2
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