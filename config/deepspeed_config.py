# -*- coding: utf-8 -*-
# @Time:  19:02
# @Author: tk
# @File：deepspeed_config
import json
import os

#lora 模式暂时不支持deepspeed
enable_deepspeed = False

def get_deepspeed_config():
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None
    with open(os.path.join(os.path.dirname(__file__),'deepspeed.json'), mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())
    return deepspeed_config