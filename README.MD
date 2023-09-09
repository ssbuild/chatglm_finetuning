##  statement
- [deep_training](https://github.com/ssbuild/deep_training)

```text
    dev 分支加一些新功能和想法 如果求稳定，请使用 stable分支
    08-16 解除 chatglm-6b-int4 全参训练限制， 推理可选使用 Rope NtkScale , 不训练扩展推理长度
    08-02 增加 muti lora infer 例子, 手动升级 aigc_zoo , pip install -U git+https://github.com/ssbuild/aigc_zoo.git --force-reinstall --no-deps 
    07-18 微调经验分享见: https://github.com/ssbuild/aigc_zoo#训练经验分享
    06-26 chatglm2_finetuning 移至(https://github.com/ssbuild/chatglm2_finetuning)
    06-13 support resize_token_embeddings
    06-01 支持lora deepspeed 训练，0.1.9 和 0.1.10合并
    05-27 add qlora transformers>=4.30
    05-12 fix lora int8 多卡训练 , ppo training move to https://github.com/ssbuild/rlhf_chatglm
    04-28 deep_training 0.1.3 pytorch-lightning 改名 ligntning ，旧版本 deep_training <= 0.1.2
    04-23 增加lora 保存hf权重（修改infer_lora_finetuning.py enable_merge_weight 选项），超大数据集见训练节说明
    04-12 deep_training 0.1.2.post0 fix load_in_8bit in lora
    04-11 升级 lora 以及 adalora, 另外官方fix eos_token , 请更新tokenizer_config.json
    04-10 增加一直冻结前N层微调方式，根据需要修改models.py 变量 global_num_layers_freeze
    04-07 官方精简了词表和权重，配置已同步，建议重新下载权重信息 deep_training 最低要求 0.1.1
    04-02 增加p-tuning-v2训练, 建议训练前删除缓存数据 rm -rf output
    03-28 支持加载chatglm-6b-int4权重 (修改 对应配置文件quantization_bit 4 or 8)
    03-27 fix eos
    03-26 增加三种数据策略
```
   


## install
  - pip install -U -r requirements.txt
  - 如果无法安装 , 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt  


## weight

- [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)   支持四种微调方式
- [chatglm-6b-int8](https://huggingface.co/THUDM/chatglm-6b-int8)  支持四种微调方式
- [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)  支持四种微调方式
    


## data sample
   open_data https://github.com/ssbuild/open_data
    
单条数据示例
```text
p prefix  optional
q question optional
a answer   must

```
```json
 {
    "id": 0, 
    "p": "我是qwen训练的模型",
    "paragraph": [
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
或者 

```json
 {
    "id": 0,
    "conversations": [
      {
        "from": "system",
        "value": "我是qwen训练的模型"
      },
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是机器人，有什么可以帮助你的？"
      },
      {
        "from": "user",
        "value": "从南京到上海的路线"
      },
      {
        "from": "assistant",
        "value": "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
      }
     ]
 }
```



## infer
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理lora微调模型
     python infer.py


| **量化等级**    | **最低 GPU 显存** |
| -------------- | ----------------- |
| FP16（无量化）   | 13 GB             |
| INT8           | 10 GB              |
| INT4           | 6 GB               |

   

![inference](data/1.png)





## training
```text
    制作数据
    python data_utils.py
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    训练
    python train.py
```

## 训练参数
[训练参数](args.MD)




## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [baichuan2_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)
- [aigc_evals](https://github.com/ssbuild/aigc_evals)

## 
    纯粹而干净的代码


## 关于续写
    训练 q 置空
    推理如下
    model = pl_model.get_glm_model()
    model.generate_for_continue_writing(tokenizer, "请帮我续写一段关于春天文字的文字", max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
    

## Reference
    https://github.com/THUDM/ChatGLM-6B




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/chatglm_finetuning&type=Date)](https://star-history.com/#ssbuild/chatglm_finetuning&Date)

