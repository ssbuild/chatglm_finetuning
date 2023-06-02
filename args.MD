
## 切换训练模式配置
    修改 config/main.py
    # 模块配置， 默认启用lora
    enable_deepspeed = False
    enable_ptv2 = False
    enable_lora = True
    enable_int8 = False # qlora int8
    enable_int4 = False # qlora int4

### 单机多卡
```text
可见的前两块卡
train_info_args = {
    'devices': 2,
}

# 第一块 和 第三块卡
train_info_args = {
    'devices': [0,2],
}
```

### 多机多卡训练
```text
例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### 超大数据集
    修改data_utils.py "data_backend": "lmdb" 

## 精度训练
    Trainer.precision = '16' # 半精度训练 "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"



### lora finetuning
```text
        global_args = {
           "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
           "num_layers_freeze": -1, # 非lora,非p-tuning 模式 ， <= config.json num_layers
           "pre_seq_len":  None,    #p-tuning-v2 参数
           "prefix_projection": False, #p-tuning-v2 参数
           "num_layers": -1, # 是否使用骨干网络的全部层数 最大1-28， -1 表示全层, 否则只用只用N层
        }
         lora_info_args = {
               'with_lora': True,  # 是否启用lora模块
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```
    

### ptuning v2


```text
    global_args = {
            "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
            "num_layers_freeze": -1, # 非lora,非p-tuning 模式 ， <= config.json num_layers
            "pre_seq_len":  32,    #p-tuning-v2 参数
            "prefix_projection": False, #p-tuning-v2 参数
            "num_layers": -1, # 是否使用骨干网络的全部层数 最大1-28， -1 表示全层, 否则只用只用N层
        }

```


### freeze

    
```text
    global_args = {
            "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
            "num_layers_freeze": 14, # 非lora,非p-tuning 模式 ， <= config.json num_layers
            "pre_seq_len":  None,    #p-tuning-v2 参数
            "prefix_projection": False, #p-tuning-v2 参数
            "num_layers": -1, # 是否使用骨干网络的全部层数 最大1-28， -1 表示全层, 否则只用只用N层
    }
    
 

```

## 全参数微调

```text
    global_args = {
            "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
            "num_layers_freeze": -1, # 非lora,非p-tuning 模式 ， <= config.json num_layers
            "pre_seq_len":  None,    #p-tuning-v2 参数
            "prefix_projection": False, #p-tuning-v2 参数
            "num_layers": -1, # 是否使用骨干网络的全部层数 最大1-28， -1 表示全层, 否则只用只用N层
        }
   lora_info_args = {
       'with_lora': False,  # 是否启用lora模块
        ...
   }
   adalora_info_args = {
       'with_lora': False,  # 是否启用lora模块
        ...
   }
```


