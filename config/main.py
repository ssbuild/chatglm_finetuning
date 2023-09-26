# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/31 14:43
import json
import os
import torch
from transformers import BitsAndBytesConfig

# 全局配置
global_args = {
    # 训练配置
    **dict(
        trainer_backend ='pl', # one of pl , hf
        enable_deepspeed = False,
        enable_ptv2 = False,
        enable_lora = True,
        load_in_bit = 0,  # 4 load_in_4bit, 8 load_in_8bit  other  0
    ),

    "pre_seq_len": 32,  # p-tuning-v2 参数 , None 禁用p-tuning-v2
    "prefix_projection": False,  # p-tuning-v2 参数
    "num_layers_freeze": -1,  # 非lora,非p-tuning 模式 ， <= config.json num_layers

    #与 transformers config合并
    "config_merge": {
    },
    # qlora
    "quantization_config":  BitsAndBytesConfig(
        load_in_8bit =False,
        load_in_4bit = False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
}

if not global_args["enable_ptv2"]:
    global_args["pre_seq_len"] = None

global_args["config_merge"].update({"pre_seq_len": global_args["pre_seq_len"],
                                    "prefix_projection": global_args["prefix_projection"]})



if global_args["enable_lora"]:
    from config.sft_config_lora import train_info_args,train_info_args_hf,train_model_config
elif global_args["enable_ptv2"]:
    from config.sft_config_ptv2 import train_info_args,train_info_args_hf,train_model_config
else:
    from config.sft_config import train_info_args,train_info_args_hf,train_model_config


if global_args["trainer_backend"] == "hf":
    train_info_args = train_info_args_hf




def patch_args(train_info_args):
    assert global_args["enable_lora"] + global_args["enable_ptv2"] <= 1 , ValueError("lora ptv2 cannot open at same time")

    if global_args['quantization_config'] is not None:
        global_args['quantization_config'].load_in_4bit = global_args["load_in_bit"] == 4
        global_args['quantization_config'].load_in_8bit = global_args["load_in_bit"] == 8
        if global_args["load_in_bit"] == 0:
            global_args["quantization_config"] = None

    if global_args["enable_lora"]:
        #检查lora adalora是否开启
        if 'lora' not in train_info_args and 'adalora' not in train_info_args:
            raise ValueError('please config lora or adalora')
        if train_info_args.get('lora',{}).get('with_lora',False) and train_info_args.get('adalora',{}).get('with_lora',False):
            raise Exception('lora and adalora can set one at same time !')

        train_info_args.pop('prompt', None)
    elif global_args["enable_ptv2"]:
        train_info_args.pop('lora', None)
        train_info_args.pop('adalora', None)
        if hasattr(train_info_args,"gradient_checkpointing"):
            train_info_args.gradient_checkpointing = False
    else:
        train_info_args.pop('lora',None)
        train_info_args.pop('adalora', None)
        train_info_args.pop('prompt', None)

    # 预处理
    if 'rwkv' in train_info_args[ 'tokenizer_name' ].lower():
        train_info_args[ 'use_fast_tokenizer' ] = True



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
    if global_args["enable_lora"]:
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

