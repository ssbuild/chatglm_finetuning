# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/31 14:43

import json
import os

# 模块配置， 默认启用lora
enable_deepspeed = False
enable_ptv2 = False
enable_lora = True
load_in_bit = 0  # 4 load_in_4bit, 8 load_in_8bit  other  0


if enable_lora:
    from config.sft_config_lora import *
elif enable_ptv2:
    from config.sft_config_ptv2 import *
else:
    from config.sft_config import *

if global_args['quantization_config'] is not None:
    global_args['quantization_config'].load_in_4bit = load_in_bit == 4
    global_args['quantization_config'].load_in_8bit = load_in_bit == 8
    if load_in_bit == 0:
        global_args["quantization_config"] = None


if enable_lora:
    enable_ptv2 = False

    #检查lora adalora是否开启
    if 'lora' not in train_info_args and 'adalora' not in train_info_args:
        raise ValueError('please config lora or adalora')
    if train_info_args.get('lora',{}).get('with_lora',False) and train_info_args.get('adalora',{}).get('with_lora',False):
        raise Exception('lora and adalora can set one at same time !')

    train_info_args.pop('prompt', None)
elif enable_ptv2:
    enable_lora = False
    global_args['quantization_config'] = None
    train_info_args.pop('lora', None)
    train_info_args.pop('adalora', None)
else:
    enable_ptv2 = False
    enable_lora = False
    train_info_args.pop('lora',None)
    train_info_args.pop('adalora', None)
    train_info_args.pop('prompt', None)

#预处理
if 'rwkv' in train_info_args['tokenizer_name'].lower():
    train_info_args['use_fast_tokenizer'] = True


def get_deepspeed_config(precision='fp16'):
    '''
        lora prompt finetuning 使用 deepspeed_offload.json
        普通finetuning 使用deepspeed.json
    '''
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None
    precision = str(precision).lower()
    # 选择 deepspeed 配置文件
    is_need_update_config = False
    if enable_lora or enable_ptv2:
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

