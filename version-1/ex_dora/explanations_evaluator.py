import os

import pandas as pd
import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer

from bart_scorer import BARTScorer


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


def BART_score(generated_texts):
    scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    bart_scores = []
    reference_texts = load_expert_data()

    for gen_text, ref_text in zip(generated_texts, reference_texts):
        score = scorer.score(gen_text, ref_text)
        bart_scores.append(score)
    print("BARTScores:", bart_scores)

    # Average BART Scores
    avg_bart_score = average_scores(bart_scores)
    return avg_bart_score


def BERT_score(generated_texts):
    scorer = BERTScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    bert_scores = []
    reference_texts = load_expert_data()

    for gen_text, ref_text in zip(generated_texts, reference_texts):
        score = scorer.score(gen_text, ref_text)
        bert_scores.append(score)
    print("BERTScores:", bert_scores)

    # Average BERT Scores
    avg_bert_score = average_scores(bert_scores)
    return avg_bert_score
