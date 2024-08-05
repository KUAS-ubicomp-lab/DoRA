import os

import pandas as pd
from rouge_score import rouge_scorer


def average_scores(scores):
    avg_score = sum(scores) / len(scores)
    return avg_score


def load_expert_data():
    data = {}
    for root, ds, fs in os.walk("../expert_evaluation/depression.csv"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            data[fn.split('.')[0]] = [texts, labels]
    return data


def rouge_score(generated_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    reference_texts = load_expert_data()

    for gen_text, ref_text in zip(generated_texts, reference_texts):
        score = scorer.score(ref_text, gen_text)
        rouge_scores.append(score)

    for idx, rouge_score in enumerate(rouge_scores):
        print(f"ROUGE Scores for pair {idx + 1}:")
        print(f"ROUGE-1: {rouge_score['rouge1']}")
        print(f"ROUGE-L: {rouge_score['rougeL']}")

    # Average ROUGE Scores
    avg_rouge1 = average_scores([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rougeL = average_scores([score['rougeL'].fmeasure for score in rouge_scores])
    return avg_rouge1, avg_rougeL

