##  
    4月12日 deep_training 0.1.2.post0 fix lora v1 , lova v2 load_in_8bit 没生效bug.
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



## 4.ChatGLM 预训练权重

- [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)
    




## 5.数据示例
    给出一种qa 数据示例 , 如果想跟母模型保持一致，可以参考https://huggingface.co/THUDM/chatglm-6b 数据组织结构.
    单条数据示例1
    {
        "id": 0, "paragraph": [
            #一轮会话
            {
                "q": "从南京到上海的路线",
                "a": [
                    "你好，南京到上海的路线如下：",
                    "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
                    "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
                    "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
                ]
            }
            #二轮....
        ]
    }
    单条数据示例2
    {"id": 0, "paragraph": [

        {
            "q": "写一个诗歌，关于冬天",
            "a": [
                "冬夜寂静冷，",
                 "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。",
                 " ",
                 "雪花融化成冰，",
                 "像那雪花飘洒，",
                 "在寒冷的冬天，",
                 "感受春天的喜悦。",
                 " 冬日里，",
                 "风雪渐消，",
                 "一片寂静，",
                 "把快乐和温暖带回家。"
            ]
        }
        ]
    }



## 6.生成训练record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 7.推理
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer.py

### 硬件需求

| **量化等级**    | **最低 GPU 显存** |
| -------------- | ----------------- |
| FP16（无量化）   | 13 GB             |
| INT8           | 10 GB              |
| INT4           | 6 GB               |

   

![inference](1.png)

## 8.训练
    完整参数为 config.json ， 建议全层训练
    若显存不足 ， 可以修改 config_small.json num_layers 层数
    训练精度 可以修改 config_small.json precision 16 32

    如果优化器不是lion ,可以尝试将config/config_small.json precision 改为32 ，然后修改 train.py Trainner 添加参数 precision=16 或者precision='bf16' 
    
    python train.py


## 9.是否开启lora finetuning
    with_lora

## 10. 是否开启ptuning v2
    对于 deepspeed模式 , ptuning v2  只支持 stage 0
    修改对应配置文件
    "pre_seq_len": 32,
    "prefix_projection": false
## int高效训练方式
   1. p-tuning-v2   使用chatglm-6b-int4 权重 ，配置文件quantization_bit=4
   2. lora int8     配置文件quantization_bit=0，使用官方标准float 16权重文件，修改modes.py load_in_8bit = True 修改data_utils.py lora节的target dtype为None

## 11.是否开启deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed



## 12. Reference
    https://github.com/THUDM/ChatGLM-6B
