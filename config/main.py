# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/31 14:43

import json
import os

# 模块配置， 默认启用lora
enable_deepspeed = False
enable_ptv2 = False
enable_lora = True
enable_int8 = False # qlora int8
enable_int4 = False # qlora int4


if enable_lora:
    from config.sft_config_lora import *
elif enable_ptv2:
    from config.sft_config_ptv2 import *
else:
    from config.sft_config import *



if enable_lora:
    enable_ptv2 = False
    if enable_int4:
        global_args['load_in_4bit'] = True
        global_args['load_in_8bit'] = False

    if enable_int8:
        global_args['load_in_4bit'] = False
        global_args['load_in_8bit'] = True

    if not enable_int4:
        global_args['quantization_config'] = None

    #检查lora adalora是否开启
    if 'lora' not in train_info_args and 'adalora' not in train_info_args:
        raise ValueError('please config lora or adalora')
    if train_info_args.get('lora',{}).get('with_lora',False) and train_info_args.get('adalora',{}).get('with_lora',False):
        raise Exception('lora and adalora can set one at same time !')

    train_info_args.pop('prompt', None)
elif enable_ptv2:
    enable_lora = False
    global_args['load_in_4bit'] = False
    global_args['load_in_8bit'] = False
    train_info_args.pop('lora', None)
    train_info_args.pop('adalora', None)
else:
    enable_ptv2 = False
    enable_lora = False
    # global_args['load_in_4bit'] = False
    # global_args['load_in_8bit'] = False
    train_info_args.pop('lora',None)
    train_info_args.pop('adalora', None)
    train_info_args.pop('prompt', None)

#预处理
if 'rwkv' in train_info_args['tokenizer_name'].lower():
    train_info_args['use_fast_tokenizer'] = True


def get_deepspeed_config():
    '''
        lora prompt finetuning 使用 deepspeed_offload.json
        普通finetuning 使用deepspeed.json
    '''
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None

    # 选择 deepspeed 配置文件
    is_need_update_config = False
    if enable_lora:
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
    return deepspeed_config

