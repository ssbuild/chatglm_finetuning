##  
    4月12日 deep_training 0.1.2.post0 fix load_in_8bit in lora v1 , lora v2 .
    4月11日 增加 lora v2 以及 adalora, 另外官方fix eos_token , 请更新tokenizer_config.json
    4月10日 增加一直冻结前N层微调方式，根据需要修改models.py 变量 global_num_layers_freeze
    4月07日 官方精简了词表和权重，配置已同步，建议重新下载权重信息 deep_training 最低要求 0.1.1
    4月02日 增加p-tuning-v2训练, 建议训练前删除缓存数据 rm -rf output
    3月29日 使得GPU更充分利用
    3月28日 支持加载chatglm-6b-int4权重 (修改 对应配置文件quantization_bit 4 or 8)
    3月27日 fix eos bugs
    3月26日 增加三种数据策略，欢迎指正

## 注意事项 ！
    dev 分支加一些新功能和想法 如果求稳定，请使用 stable-vocab130528 分支
    

## 1.安装
  - pip install -i https://pypi.org/simple -U deep_training>=0.1.2 cpm_kernels transformers>=4.26.1 deepspeed
    

## 2.更新详情
- [deep_training](https://github.com/ssbuild/deep_training)

## 3.深度学习常规任务例子

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf_finetuning](https://github.com/ssbuild/llm_rlhf_finetuning)
- [chatglm_rlhf_finetuning](https://github.com/ssbuild/chatglm_rlhf_finetuning)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)



## 4.ChatGLM 预训练权重

- [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [chatglm-6b-int8](https://huggingface.co/THUDM/chatglm-6b-int8)
- [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)
    




## 5.数据示例
   第三方羊驼数据 https://github.com/hikariming/alpaca_chinese_dataset
    单条数据示例
```json
 {
     "id": 0, "paragraph": [
         {
             "q": "从南京到上海的路线",
             "a":  "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
         }
     ]
 }
```
    或者
```json
 {
     "id": 0, "paragraph": [
         {
             "q": "从南京到上海的路线",
             "a": [
                 "你好，南京到上海的路线如下：",
                 "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
                 "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
                 "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
             ]
         }
     ]
 }
```
   多轮会话
```json
 {
     "id": 0, "paragraph": [
        {
           "q": "你好",
           "a": "我是机器人，有什么可以帮助你的？"
        },
         {
             "q": "从南京到上海的路线",
             "a":  "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
         }
     ]
 }

```



## 6.生成训练record
    python data_utils.py
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 7.推理
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理lora微调模型
     python infer.py

### 硬件需求

| **量化等级**    | **最低 GPU 显存** |
| -------------- | ----------------- |
| FP16（无量化）   | 13 GB             |
| INT8           | 10 GB              |
| INT4           | 6 GB               |

   

![inference](1.png)

## 8.训练
    支持4种微调方式 
- 冻结 N 层 修改models.py global_num_layers_freeze   
- ptuning v2    
- lora
- 正常微调
```text
    训练精度 可以修改 config.json precision 16 32
    如果优化器不是lion ,可以尝试将config/config.json precision 改为32 ，然后修改 train.py Trainner 添加参数 precision=16 或者precision='bf16'
    python train.py
```

```text
多机多卡训练 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### 是否开启lora finetuning
    with_lora = True ， lora 和 adalora 不能同时开启

### 是否开启ptuning v2
    对于 deepspeed模式 , ptuning v2  只支持 stage 0
    修改对应配置文件
    "pre_seq_len": 32,
    "prefix_projection": false
### int高效训练方式
   1. p-tuning-v2   使用chatglm-6b-int4 权重 ，配置文件quantization_bit=4
   2. lora int8     配置文件quantization_bit=0，使用官方标准float 16权重文件，修改modes.py load_in_8bit = True 
修改data_utils.py 对应with_lora=True 
   注意 lora int8 多卡训练  CUDA_VISIBLE_DEVICES=显卡数量 和 data_utils.py devices 保持一致，否则容易出故障 ， 建议使用 CUDA_VISIBLE_DEVICES=显卡数量 python train.py

## 9.是否开启deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed


## 10. Reference
    https://github.com/THUDM/ChatGLM-6B

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/chatglm_finetuning&type=Date)](https://star-history.com/#ssbuild/chatglm_finetuning&Date)

