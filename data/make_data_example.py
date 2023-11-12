# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50



import json


x1 = [
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


x2 = [

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


x3 = [

    {
        "q": "晚上睡不着应该怎么办",
        "a": [
            "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:",
            "",
            "1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。",
            "2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。",
            "3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。",
            "4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。",
            "5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。",
            "6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。",
            "",
            "如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"
        ]
    }
]




x = [x1,x2,x3]

with open('./finetune_train_paragraph.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(50):
        for j in range(len(x)):
            index += 1

            conversations = {
                "id": index,
                "paragraph": x[j]
            }
            f.write(json.dumps(conversations,ensure_ascii=False) + '\n' )



with open('./finetune_train_conversations.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(50):
        for j in range(len(x)):
            index += 1

            conversation = []
            for item in x[j]:
                role = item.get("role","user")
                if role == "system":
                    conversation.append( {
                        "from":  item.get("role","user"),
                        "value": item["q"]
                    })
                else:
                    conversation.append({
                        "from": item.get("role", "user"),
                        "value": item["q"]
                    })
                    conversation.append({
                        "from": "assistant",
                        "value": item["a"]
                    })

            conversations = {
                "id": index,
                "conversations": conversation
            }
            f.write(json.dumps(conversations,ensure_ascii=False) + '\n' )