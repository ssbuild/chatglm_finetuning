# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import json
import random
import re
import typing
from enum import Enum
import numpy as np
import requests

from deep_training.zoo.model_zoo.glm4v.llm_model import ChatGLM4Tokenizer


class DataStrategy(Enum):
    truncation = 1




class TokenIdsMaker:
    def __init__(self, tokenizer: ChatGLM4Tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id


    def build_single_message(self, role, metadata, message, tokenize=True):
        assert role in ["system", "user", "assistant", "observation"], role
        if tokenize:
            role_tokens = [self.tokenizer.convert_tokens_to_ids(f"<|{role}|>")] + self.tokenizer.tokenizer.encode(
                f"{metadata}\n",
                disallowed_special=())
            message_tokens = self.tokenizer.tokenizer.encode(message, disallowed_special=())
            tokens = role_tokens + message_tokens
            return tokens
        else:
            return str(f"<|{role}|>{metadata}\n{message}")

    @classmethod
    def final(cls, input_ids: typing.List, labels, max_seq_length, tokenizer, input_image: typing.Optional[str]):
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen

        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))

        if input_image is None:
            input_image = ''
        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
            'img': np.asarray(bytes(input_image,encoding='utf-8'))
        }

        return d

    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        if query is not None:
            input_ids.extend(self.build_single_message(role, "", query))
        return self.tokenizer.encode(input_ids, is_split_into_words=True)

    def parse_history_from_answers(self, output, history):
        content = ""
        history = copy.deepcopy(history)
        responses = output.split("<|assistant|>")
        for response in responses:
            content = response
            history.append({"role": "assistant", "content": content})
        return content, history

    def get_prefix_tokens(self, tokenizer):
        prefix_tokens = [tokenizer.convert_tokens_to_ids("[gMASK]"), tokenizer.convert_tokens_to_ids("<sop>")]
        return prefix_tokens

    def get_tools_content(self, prompt, tools):
        content = prompt or "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
        for tool in tools:
            if tool["type"] == "function":
                function = tool["function"]
                content += f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
                content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
            elif tool["type"] == "python":
                content += "\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。"
            elif tool["type"] == "simple_browser":
                content += "\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。"
            elif tool["type"] == "cogview":
                content += "\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。"
            else:
                raise NotImplementedError(f"Unknown tool type {tool['type']}")
        return content

    def trunction(self, tokenizer: ChatGLM4Tokenizer, config, examples, max_seq_length, sup=True):
        bos_ids = self.get_prefix_tokens(tokenizer)
        assert isinstance(bos_ids, list) and len(bos_ids) == 2
        ds = []
        history = []
        input_image = None
        if examples[0][0] == "system":
            role, q, tools, input_image = examples.pop(0)
            sys_prompt = {
                "role": "system",
                "content": q,
            }
            if tools != None:
                tools = json.loads(tools)
                assert isinstance(tools, list)
                sys_prompt["content"] = self.get_tools_content(q, tools)
        else:
            sys_prompt = {
                "role": "system",
                "content": "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
            }

        history += [sys_prompt]

        for sid, (q_role, q, a, img) in enumerate(examples):
            if img is not None:
                input_image = img
            if q_role != "observation":
                q_role = "user"
            history += [{
                "role": q_role,
                "content": q,
            }]
            a_ids = self.build_chat_input(query=None, history=history)
            _, history = self.parse_history_from_answers(a, history)
            b_ids = self.tokenizer.encode(self.tokenizer.encode(a), is_split_into_words=True)
            role_tokens = [self.tokenizer.convert_tokens_to_ids("<|assistant|>")]
            while len(a_ids) + len(b_ids) > max_seq_length - len(role_tokens) - 3:
                if len(b_ids) > len(a_ids):
                    b_ids.pop(-1)
                else:
                    a_ids.pop(0)
            assert len(b_ids) > 0
            b_ids += [self.eos_token_id]
            a_ids = a_ids + role_tokens
            input_ids = a_ids + b_ids
            labels = copy.deepcopy(input_ids) if not sup else [-100] * len(a_ids) + copy.deepcopy(b_ids)
            input_ids = bos_ids + input_ids
            labels = bos_ids + labels
            assert len(input_ids) <= max_seq_length
            ds.append(self.final(input_ids, labels, max_seq_length, tokenizer, input_image))

        return ds

