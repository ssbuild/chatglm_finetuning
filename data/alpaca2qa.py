# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 8:50

# @misc{alpaca_chinese_dataset,
#   author = {Liu, Beiming and Huang, Kunhao and Jiao, Lihua and He, Yuchen and Zhang, Ruiqin and Liang, Yuan and Wang, Yingshan},
#   title = {Chinese Alpaca Dataset},
#   year = {2023},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/hikariming/alpaca_chinese_dataset}},
# }

# 数据参考 https://github.com/hikariming/alpaca_chinese_dataset
#
### 样例

# ``` json
# [
#   {
#       "instruction": "根据给定的坐标确定最近的机场。",
#       "input": "40.728157, -73.794853",
#       "output": "距给定坐标最近的机场是纽约市的拉瓜迪亚机场 (LGA)。"
#   },
#   {
#      "instruction": "输出不同种类水果的列表",
#       "input": "",
#       "output": "1. 苹果\n2. 香蕉\n3. 橘子\n4. 芒果\n5. 草莓\n6. 葡萄\n7. 蓝莓\n8. 樱桃\n9. 猕猴桃\n10. 甜瓜\n11.菠萝\n12. 李子\n13.桃子"
#   },
#   {
#      "instruction": "找出字串中隐藏的信息",
#      "input": "业余咖啡",
#      "output": "隐藏的消息是“咖啡因爱好者”。"
#   }
#  ]
# ```
import json

PROMPT_DICT = {
    "prompt_input": (
        "下面是一个指令,描述了一个任务,搭配一个输入,提供进一步的上下文。"
        "编写适当的输出完成请求。\n\n"
        "### 指令:\n{instruction}\n\n### 请求:\n{input}\n\n### 响应:"
    ),
    "prompt_no_input": (
        "下面是一个指令,描述了一个任务。"
        "编写适当的输出完成请求。\n\n"
        "### 指令:\n{instruction}\n\n### 响应:"
    ),
}


def alaca2qa(src,dst):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    with open(src,mode='r',encoding='utf-8') as f:
        list_data_dict = json.loads(f.read())
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}" for example in list_data_dict]

    with open(dst, mode='w', encoding='utf-8',newline='\n') as f:

        for i,(s, t) in enumerate(zip(sources, targets)):
            paragraph = [
                {
                    'q': s,
                    'a': [t]
                }
            ]
            f.write(json.dumps({'id': i+1 ,'paragraph' : paragraph },ensure_ascii=False) +'\n')


if __name__ == '__main__':
    src = r'./data/zh-data-part-00.json'
    dst = r'./data/alpaca-part-00.json'
    alaca2qa(src,dst)