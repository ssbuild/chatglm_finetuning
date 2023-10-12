# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 11:25
import numpy as np
from sacrebleu.metrics import BLEU
from rouge import Rouge


def evaluate(data):
    bleu_scorer_obj = BLEU()
    rouge_scorer_obj = Rouge()
    bleu_score = []
    for d in data:
        score = bleu_scorer_obj.sentence_score(
            hypothesis=d['text'],
            references=d['ref'],
        )
        bleu_score.append(score.score)

    bleu_score = np.average(np.asarray(bleu_score))

    rouge_score = []
    for d in data:
        score = rouge_scorer_obj.get_scores(
            hyps=[d['text']],
            refs=d['ref'],
        )
        rouge_score.append(score[0]["rouge-l"]["f"])

    rouge_score = np.average(np.asarray(rouge_score))

    return {
        "bleu_score": bleu_score,
        "rouge-l_score": rouge_score
    }



if __name__ == '__main__':
    data = [
        {
            "text": "to make people trustworthy you need to trust them",
            "ref": ["the way to make people trustworthy is to trust them"]
        },
    ]

    result = evaluate(data)
    print(result)


