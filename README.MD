##  statement
- [deep_training](https://github.com/ssbuild/deep_training)

```text
    10-09 support accelerator trainer
    10-07 support colossalai trainer
    09-26 support transformers trainer
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
    03-27 fix eos and support chatglm-6b-int4权重 (修改 对应配置文件quantization_bit 4 or 8)
    03-26 完善数据策略
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
q question optional
a answer   must

```
```json
{"id": 1, "paragraph": [{"q": "从南京到上海的路线", "a": ["你好，南京到上海的路线如下：", "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。", "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。", "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"]}]}
{"id": 2, "paragraph": [{"q": "写一个诗歌，关于冬天", "a": ["冬夜寂静冷，", "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。", " ", "雪花融化成冰，", "像那雪花飘洒，", "在寒冷的冬天，", "感受春天的喜悦。", " 冬日里，", "风雪渐消，", "一片寂静，", "把快乐和温暖带回家。"]}]}
{"id": 3, "paragraph": [{"q": "晚上睡不着应该怎么办", "a": ["晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:", "", "1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。", "2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。", "3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。", "4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。", "5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。", "6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。", "", "如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"]}]}
```

或者

```json
{"id": 1, "conversations": [{"from": "user", "value": "从南京到上海的路线"}, {"from": "assistant", "value": ["你好，南京到上海的路线如下：", "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。", "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。", "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"]}]}
{"id": 2, "conversations": [{"from": "user", "value": "写一个诗歌，关于冬天"}, {"from": "assistant", "value": ["冬夜寂静冷，", "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。", " ", "雪花融化成冰，", "像那雪花飘洒，", "在寒冷的冬天，", "感受春天的喜悦。", " 冬日里，", "风雪渐消，", "一片寂静，", "把快乐和温暖带回家。"]}]}
{"id": 3, "conversations": [{"from": "user", "value": "晚上睡不着应该怎么办"}, {"from": "assistant", "value": ["晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:", "", "1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。", "2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。", "3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。", "4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。", "5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。", "6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。", "", "如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"]}]}
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
# 制作数据
cd scripts
bash train_full.sh -m dataset 
or
bash train_lora.sh -m dataset 
or
bash train_ptv2.sh -m dataset 

注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)

# 全参数训练 
    bash train_full.sh -m train
    
# lora adalora ia3 
    bash train_lora.sh -m train
    
# ptv2
    bash train_ptv2.sh -m train
```

## 训练参数
[训练参数](args.MD)




## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [moss_finetuning](https://github.com/ssbuild/moss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [skywork_finetuning](https://github.com/ssbuild/skywork_finetuning)
- [bluelm_finetuning](https://github.com/ssbuild/bluelm_finetuning)
- [yi_finetuning](https://github.com/ssbuild/yi_finetuning)

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

