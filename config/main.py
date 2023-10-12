# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/31 14:43
import json
import os
import torch
import yaml
from transformers import BitsAndBytesConfig
from transformers.utils import strtobool

from config.colossalai_config import colossalai_strategy
from config.constant_map import *



# 加载

__CUR_PATH__ = os.path.abspath(os.path.dirname(__file__))
TRAIN_CONFIG_CACHE = os.path.join(__CUR_PATH__,"config_path_cache.json")


def load_config():
    if "train_config" not in os.environ:
        if not os.path.exists(TRAIN_CONFIG_CACHE):
            raise ValueError("train_config is not set environ")
        with open(TRAIN_CONFIG_CACHE, mode='r', encoding='utf-8') as __f:
            __config_path = json.load(__f)
    else:
        __config_path = os.path.abspath(os.environ.get("train_config"))
        #缓存，方便推理不传环境变量
        with open(TRAIN_CONFIG_CACHE, mode='w', encoding='utf-8') as __f:
            json.dump(__config_path, __f)

    with open(__config_path,mode='r',encoding='utf-8') as __f:
        train_info_args = yaml.full_load(__f)
    return train_info_args

train_info_args = load_config()
global_args = train_info_args.pop("global_args")
train_model_config = MODELS_MAP[global_args["model_name"]]
assert global_args["trainer_backend"] in ["pl","hf","cl","ac"]

# ensure str
global_args["precision"] = str(global_args["precision"])

if global_args["quantization_config"]:
    #精度
    if global_args["precision"] == "auto":
        global_args["quantization_config"]["bnb_4bit_compute_dtype"] ="bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    global_args["quantization_config"] = BitsAndBytesConfig(**global_args["quantization_config"])



def merge_from_env(global_args):
    merge_config = {}
    if "trainer_backend" in os.environ:
        merge_config["trainer_backend"] = str(os.environ["trainer_backend"])
    if "enable_deepspeed" in os.environ:
        merge_config["enable_deepspeed"] = strtobool(os.environ["enable_deepspeed"])
    if "enable_ptv2" in os.environ:
        merge_config["enable_ptv2"] = strtobool(os.environ["enable_ptv2"])
    if "enable_lora" in os.environ:
        merge_config["enable_lora"] = strtobool(os.environ["enable_lora"])
    if "load_in_bit" in os.environ:
        merge_config["load_in_bit"] = int(os.environ["load_in_bit"])
    if merge_config:
        global_args.update(merge_config)

merge_from_env(global_args)

def patch_args(train_info_args):
    assert global_args["enable_lora"] + global_args["enable_ptv2"] <= 1 , ValueError("lora ptv2 cannot open at same time")

    #更新模型配置
    train_info_args.update(train_model_config)

    if global_args["trainer_backend"] == "cl":
        train_info_args["strategy"] = colossalai_strategy[train_info_args["strategy"]]

    if global_args['quantization_config'] is not None:
        global_args['quantization_config'].load_in_4bit = global_args["load_in_bit"] == 4
        global_args['quantization_config'].load_in_8bit = global_args["load_in_bit"] == 8
        if global_args["load_in_bit"] == 0:
            global_args["quantization_config"] = None



    if global_args["enable_lora"]:
        # 检查lora adalora是否开启
        assert train_info_args.get('lora', {}).get('with_lora', False) + \
               train_info_args.get('adalora', {}).get('with_lora', False) + \
               train_info_args.get('ia3', {}).get('with_lora', False) == 1, ValueError(
            'lora adalora ia3 can set one at same time !')

        model_type = train_model_config['model_type']
        if train_info_args.get('lora', {}).get('with_lora', False):
            train_info_args["lora"]["target_modules"] = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
        elif train_info_args.get('adalora', {}).get('with_lora', False):
            train_info_args["adalora"]["target_modules"] = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[model_type]
        else:
            train_info_args["ia3"]["target_modules"] = TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_type]
            train_info_args["ia3"]["feedforward_modules"] = TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[model_type]

        train_info_args.pop('prompt', None)

    elif global_args["enable_ptv2"]:
        train_info_args.pop('lora', None)
        train_info_args.pop('adalora', None)
        train_info_args.pop('ia3', None)
        if "gradient_checkpointing" in train_info_args:
            train_info_args[ "gradient_checkpointing" ] = False

        assert "prompt" in train_info_args
        train_info_args["prompt"]["with_prompt"] = True
    else:
        train_info_args.pop('lora',None)
        train_info_args.pop('adalora', None)
        train_info_args.pop('ia3', None)
        train_info_args.pop('prompt', None)

    # 预处理
    if 'rwkv' in (train_info_args['model_type'] or train_info_args['model_name_or_path']).lower():
        train_info_args['use_fast_tokenizer'] = True



patch_args(train_info_args)

def get_deepspeed_config(precision='fp16'):
    '''
        lora prompt finetuning   deepspeed_offload.json
        普通  finetuning          deepspeed.json
    '''
    # 是否开启deepspeed
    if not global_args["enable_deepspeed"]:
        return None
    precision = str(precision).lower()
    # 选择 deepspeed 配置文件
    is_need_update_config = False
    if global_args["enable_lora"] or global_args["enable_ptv2"]:
        is_need_update_config = True
        filename = os.path.join(os.path.dirname(__file__), 'deepspeed_offload.json')
    else:
        filename = os.path.join(os.path.dirname(__file__), 'deepspeed.json')


    with open(filename, mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())

    #lora offload 同步优化器配置
    if is_need_update_config:
        optimizer = deepspeed_config.get('optimizer',None)
        if optimizer:
            if global_args["trainer_backend"] == 'hf':
                optimizer[ 'params' ][ 'betas' ] = (train_info_args.get('adam_beta1', 0.9),train_info_args.get('adam_beta2', 0.999),)
                optimizer[ 'params' ][ 'lr' ] = train_info_args.get('learning_rate', 2e-5)
                optimizer[ 'params' ][ 'eps' ] = train_info_args.get('adam_epsilon', 1e-8)
                # deepspeed_offload 优化器有效
                train_info_args[ 'optim' ] = optimizer[ 'type' ]
            else:
                optimizer['params']['betas'] = train_info_args.get('optimizer_betas', (0.9, 0.999))
                optimizer['params']['lr'] = train_info_args.get('learning_rate', 2e-5)
                optimizer['params']['eps'] = train_info_args.get('adam_epsilon', 1e-8)
                # deepspeed_offload 优化器有效
                train_info_args['optimizer'] = optimizer['type']

    if precision == 'bf16':
        if 'fp16' in deepspeed_config:
            deepspeed_config["fp16"]["enbale"] = False
        if 'bf16' in deepspeed_config:
            deepspeed_config["bf16"]["enbale"] = True
        else:
            deepspeed_config['bf16'] = {"enbale": True}
    elif precision == 'fp16':
        if 'bf16' in deepspeed_config:
            deepspeed_config["bf16"]["enbale"] = False

    return deepspeed_config

