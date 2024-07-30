import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(engine):
    tokenizer = AutoTokenizer.from_pretrained(engine)
    model = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def generate_explanations(utterance, in_context_demonstrations, engine, max_length=200, num_return_sequences=3):
    model, tokenizer, device = load_model_and_tokenizer(engine)
    # The prompt is adjusted to emphasize the depressive elements of both the in-context demonstrations and
    # the input utterance. This helps guide the model to focus on recognizing and explaining depressive content.
    prompt = "Below are examples with depressive elements and their explanations:\n"
    for example in in_context_demonstrations:
        prompt += f"Example: {example}\nExplanation: This example shows signs of depression because...\n"
    prompt += (f"\nAnalyze the following text for depressive elements and provide explanations:\nExample: {utterance}"
               f"\nExplanation:")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=0.95
        )
    explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return explanations


def rank_explanations(explanations, utterance, in_context_demonstrations, dsm_criteria,
                      similarity_model, decay_factor=0.85):
    context_embeddings = similarity_model.encode(in_context_demonstrations)
    input_embedding = similarity_model.encode([utterance])
    dsm_embeddings = similarity_model.encode(dsm_criteria)
    explanation_embeddings = similarity_model.encode(explanations)

    relevance_scores = []
    for explanation_embedding in explanation_embeddings:
        context_similarity = util.cos_sim(explanation_embedding, context_embeddings).mean().item()
        input_similarity = util.cos_sim(explanation_embedding, input_embedding).item()
        dsm_similarity = util.cos_sim(explanation_embedding, dsm_embeddings).mean().item()
        relevance_score = (context_similarity + input_similarity + dsm_similarity) / 3
        relevance_scores.append(relevance_score)

    # Normalize relevance scores to probabilities.
    relevance_probabilities = np.array(relevance_scores) / np.sum(relevance_scores)

    # Calculate ERR. Expected Reciprocal Rank (ERR) (https://dl.acm.org/doi/10.1145/1645953.1646033) is a
    # probabilistic framework to rank the generated explanations.
    err_scores = []
    running_probability = 1.0
    for i, probability in enumerate(relevance_probabilities):
        err_score = running_probability * probability / (i + 1.0)
        err_scores.append(err_score)
        running_probability *= (1 - probability * decay_factor)

    # Rank explanations based on ERR scores.
    ranked_explanations = sorted(zip(explanations, err_scores), key=lambda x: x[1], reverse=True)
    logger.info(f"Ranked Explanations {ranked_explanations}")
    return ranked_explanations


def main():
    in_context_demonstrations = [
        "She felt overwhelmed by the constant demands at work and home.",
        "He was anxious about the upcoming exams and had trouble sleeping."
    ]
    # Choose model: 'plm/Mistral-7B-Instruct-v0.2' or 'plm/gemma-7b-it'
    engine = 'plm/Mistral-7B-Instruct-v0.2'
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # DSM-5 criteria for depression
    dsm_criteria = [
        "Persistent sad, anxious, or 'empty' mood",
        "Feelings of hopelessness or pessimism",
        "Irritability",
        "Feelings of guilt, worthlessness, or helplessness",
        "Loss of interest or pleasure in hobbies and activities"
    ]
    input_text = "I'm struggling to find motivation and everything seems pointless."

    explanations = generate_explanations(input_text, in_context_demonstrations, engine)
    ranked_explanations = rank_explanations(explanations, input_text, in_context_demonstrations, dsm_criteria,
                                            similarity_model)
    for idx, (explanation, score) in enumerate(ranked_explanations, 1):
        print(f"Rank {idx} (Score: {score:.4f}): {explanation}")


if __name__ == '__main__':
    main()
