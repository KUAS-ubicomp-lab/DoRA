import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, Dataset

import explanations_generator

current_dir = os.path.dirname(os.path.abspath(__file__))


def extract_features(input_texts, in_context_demonstrations, explanations, dsm_criteria,
                     similarity_model='sentence-transformers/all-MiniLM-L6-v2'):
    feature_extractor = SentenceTransformer(similarity_model)
    context_embeddings = feature_extractor.encode(in_context_demonstrations, convert_to_tensor=True)
    input_embeddings = feature_extractor.encode(input_texts, convert_to_tensor=True)
    dsm_embeddings = feature_extractor.encode(dsm_criteria, convert_to_tensor=True)
    explanation_embeddings = feature_extractor.encode(explanations, convert_to_tensor=True)

    context_embeddings = context_embeddings.mean(dim=0, keepdim=True)
    input_embeddings = input_embeddings.mean(dim=0, keepdim=True)
    dsm_embeddings = dsm_embeddings.mean(dim=0, keepdim=True)
    explanation_embeddings = explanation_embeddings.mean(dim=0, keepdim=True)

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


class ExplanationRankingDataset(Dataset):
    def __init__(self, context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings, relevance_scores):
        self.context_embeddings = torch.tensor(context_embeddings, dtype=torch.float32)
        self.input_embeddings = torch.tensor(input_embeddings, dtype=torch.float32)
        self.dsm_embeddings = torch.tensor(dsm_embeddings, dtype=torch.float32)
        self.explanation_embeddings = torch.tensor(explanation_embeddings, dtype=torch.float32)
        self.relevance_scores = torch.tensor(relevance_scores, dtype=torch.float32)

        print("Context Embeddings Shape:", self.context_embeddings.shape)
        print("Input Embeddings Shape:", self.input_embeddings.shape)
        print("DSM Embeddings Shape:", self.dsm_embeddings.shape)
        print("Explanation Embeddings Shape:", self.explanation_embeddings.shape)

    def __len__(self):
        return len(self.relevance_scores)

    def __getitem__(self, idx):
        # Ensure the idx exists within each embedding set
        context_emb = self.context_embeddings[idx % len(self.context_embeddings)]
        input_emb = self.input_embeddings[idx % len(self.input_embeddings)]
        dsm_emb = self.dsm_embeddings[idx % len(self.dsm_embeddings)]
        explanation_emb = self.explanation_embeddings[idx % len(self.explanation_embeddings)]

        # Flatten and concatenate all embeddings
        input_feature = torch.cat((context_emb, input_emb, dsm_emb, explanation_emb), dim=-1)
        target_score = self.relevance_scores[idx]

        return input_feature, target_score


class CustomRankingLoss(nn.Module):
    def __init__(self, decay_factor=0.85, lambda_diversity=0.5):
        super(CustomRankingLoss, self).__init__()
        self.decay_factor = decay_factor
        self.lambda_diversity = lambda_diversity

    def forward(self, predictions, target_scores, context_embeddings, input_embeddings, dsm_embeddings,
                explanation_embeddings):
        relevance_scores = predictions.squeeze()

        # Normalize relevance scores to probabilities
        relevance_probabilities = torch.softmax(relevance_scores, dim=0).detach().cpu().numpy()

        err_scores, selected_indices = explanations_generator.relevance_diversity_scoring(
            relevance_probabilities=relevance_probabilities,
            explanation_embeddings=explanation_embeddings,
            decay_factor=self.decay_factor,
            lambda_diversity=self.lambda_diversity
        )

        # Calculate combined ERR and MMR loss
        err_loss = -sum(err_scores)
        mmr_loss = -sum(
            [util.cos_sim(explanation_embeddings[i], explanation_embeddings[selected_indices]).max().item() for i in
             selected_indices])
        combined_loss = err_loss + self.lambda_diversity * mmr_loss

        return torch.tensor(combined_loss, requires_grad=True).to(predictions.device)


def prepare_data(context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings, relevance_scores):
    dataset = ExplanationRankingDataset(context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings,
                                        relevance_scores)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def train_model(ranking_model, train_data, context_embeddings, input_embeddings, dsm_embeddings,
                explanation_embeddings, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranking_model.to(device)
    optimizer = optim.Adam(ranking_model.parameters(), lr=learning_rate)
    loss_fn = CustomRankingLoss().to(device)
    ranking_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for input_features, target_scores in train_data:
            input_features, target_scores = input_features.to(device), target_scores.to(device)
            optimizer.zero_grad()
            outputs = ranking_model(input_features)
            loss = loss_fn(outputs, target_scores, context_embeddings, input_embeddings, dsm_embeddings,
                           explanation_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')


def rank_explanations(model, explanations, input_texts, context_examples, dsm_criteria):
    context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings = extract_features(input_texts,
                                                                                                    context_examples,
                                                                                                    explanations,
                                                                                                    dsm_criteria)
    input_features = torch.cat((context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings), dim=1)
    input_features = input_features.cuda()
    model.eval()
    with torch.no_grad():
        relevance_scores = model(input_features).cpu().numpy()
    relevance_scores = relevance_scores.flatten()
    ranked_explanations = [explanations[i] for i in np.argsort(-relevance_scores)]
    return ranked_explanations


def main():
    in_context_demonstrations = [
        "She felt overwhelmed by the constant demands at work and home.",
        "He was anxious about the upcoming exams and had trouble sleeping."
    ]
    input_texts = [
        "I'm struggling to find motivation and everything seems pointless.",
        "I don't feel like doing anything anymore, even the things I used to love."
    ]
    dsm_criteria = [
        "Persistent sad, anxious, or 'empty' mood",
        "Feelings of hopelessness or pessimism",
        "Irritability",
        "Feelings of guilt, worthlessness, or helplessness",
        "Loss of interest or pleasure in hobbies and activities"
    ]
    # Choose model: 'plm/Mistral-7B-Instruct-v0.2' or 'plm/gemma-7b-it'
    engine = os.path.join(current_dir, '..', 'plm', 'Mistral-7B-Instruct-v0.2')
    explanations = explanations_generator.generate_explanations(input_texts, in_context_demonstrations, engine)
    relevance_scores = [0.9, 0.7, 0.85]

    context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings = extract_features(
        input_texts=input_texts,
        in_context_demonstrations=in_context_demonstrations,
        explanations=explanations,
        dsm_criteria=dsm_criteria)

    train_data = prepare_data(context_embeddings, input_embeddings, dsm_embeddings, explanation_embeddings,
                              relevance_scores)
    input_dim = (context_embeddings.shape[-1] + input_embeddings.shape[-1] + dsm_embeddings.shape[-1]
                 + explanation_embeddings.shape[-1])
    ranking_model = ExplanationRankingModel(input_dim)

    train_model(ranking_model, train_data, context_embeddings, input_embeddings, dsm_embeddings,
                explanation_embeddings, epochs=10, learning_rate=0.001)

    ranked_explanations = rank_explanations(ranking_model, explanations, input_texts, in_context_demonstrations,
                                            dsm_criteria)
    for idx, explanation in enumerate(ranked_explanations, 1):
        print(f"Rank {idx}: {explanation}")


if __name__ == '__main__':
    main()
