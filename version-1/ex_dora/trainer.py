import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util

import explanations_generator


def extract_features(input_texts, context_examples, explanations, dsm_criteria,
                     model_name='sentence-transformers/all-MiniLM-L6-v2'):
    feature_extractor = SentenceTransformer(model_name)
    context_embeddings = feature_extractor.encode(context_examples)
    input_embeddings = feature_extractor.encode(input_texts)
    dsm_embeddings = feature_extractor.encode(dsm_criteria)
    explanation_embeddings = feature_extractor.encode(explanations)
    return context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings


class ExplanationRankingModel(nn.Module):
    def __init__(self, input_dim):
        super(ExplanationRankingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomRankingLoss(nn.Module):
    def __init__(self, decay_factor=0.85, lambda_diversity=0.5):
        super(CustomRankingLoss, self).__init__()
        self.decay_factor = decay_factor
        self.lambda_diversity = lambda_diversity

    def forward(self, predictions, target_scores, context_embeddings, input_embeddings, dsm_embeddings,
                explanation_embeddings):
        relevance_scores = predictions.squeeze()

        # Normalize relevance scores to probabilities
        relevance_probabilities = torch.softmax(relevance_scores, dim=0).cpu().numpy()

        # Calculate ERR
        err_scores = []
        running_prob = 1.0
        for i, prob in enumerate(relevance_probabilities):
            err_score = running_prob * prob / (i + 1)
            err_scores.append(err_score)
            running_prob *= (1 - prob * self.decay_factor)

        err_scores, selected_indices = explanations_generator.relevance_diversity_scoring(
            decay_factor=self.decay_factor,
            explanation_embeddings=explanation_embeddings,
            lambda_diversity=self.lambda_diversity,
            relevance_probabilities=relevance_probabilities
        )

        # Calculate combined ERR and MMR loss
        err_loss = -sum(err_scores)
        mmr_loss = -sum(
            [util.cos_sim(explanation_embeddings[i], explanation_embeddings[selected_indices]).max().item() for i in
             selected_indices])
        combined_loss = err_loss + self.lambda_diversity * mmr_loss

        return torch.tensor(combined_loss, requires_grad=True).to(predictions.device)
