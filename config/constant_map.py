# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

train_info_models = {
    'chatglm': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm/chatglm-6b',
        'config_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b',
    },
    'chatglm-6b-int4': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4',
        'config_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4',
    },
    'chatglm-6b-int8': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8',
        'config_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8',
    },

}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

train_model_config = train_info_models['chatglm']