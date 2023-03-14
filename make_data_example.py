# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50


import json


x1 = {
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

x2 = {"id": 0, "paragraph": [

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

x = [x1,x2]

with open('./data/finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(100):
        for j in range(len(x)):
            index += 1
            x[j]['id'] = index
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )
