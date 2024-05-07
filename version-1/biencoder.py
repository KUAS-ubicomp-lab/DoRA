import collections
import logging
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from utils.biencoder_data import BiEncoderSample

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiencoderInput",
    [
        "utterance_ids",
        "utterance_segments",
        "context_ids",
        "ctx_segments",
        "encoder_type",
    ],
)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/utterance and context/candidate encoders."""

    def __init__(
            self,
            utterance_model: nn.Module,
            ctx_model: nn.Module,
            fix_uttr_encoder: bool = False,
            fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.utterance_model = utterance_model
        self.ctx_model = ctx_model
        self.fix_uttr_encoder = fix_uttr_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
            sub_model: nn.Module,
            ids: T,
            segments: T,
            positions: T,
            attn_mask: T,
            fix_encoder: bool = False,
            representation_token_pos=0,
    ) -> (T, T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        positions,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    positions,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
            self,
            utterance_ids: T,
            utterance_segments: T,
            utterance_positions: T,
            utterance_attn_mask: T,
            context_ids: T,
            ctx_segments: T,
            ctx_attn_mask: T,
            encoder_type: str = None,
            representation_token_pos=0,
    ) -> Tuple[T, T]:
        utterance_encoder = (
            self.utterance_model
            if encoder_type is None or encoder_type == "classification"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            utterance_encoder,
            utterance_ids,
            utterance_segments,
            utterance_positions,
            utterance_attn_mask,
            self.fix_uttr_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.utterance_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out

    @classmethod
    def generate_biencoder_input(
            cls,
            samples: List[BiEncoderSample],
            shuffle: bool = True,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param shuffle: shuffles demonstration pools
        :return: BiEncoderBatch tuple
        """
        utterance_tensors = []
        ctx_tensors = []

        for sample in samples:
            if shuffle:
                candidate_ctxs = sample.candidate_samples
                candidate_ctxs[np.random.choice(len(candidate_ctxs))]
            else:
                sample.candidate_samples[0]

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        utterance_tensor = torch.cat([q.view(1, -1) for q in utterance_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        utterance_segments = torch.zeros_like(utterance_tensor)

        return BiEncoderBatch(
            utterance_tensor,
            utterance_segments,
            ctxs_tensor,
            ctx_segments,
            "classification",
        )


class BiEncoder_list_ranking_loss:
    @staticmethod
    def get_scores(uttr_vector: T, ctx_vectors: T) -> T:
        scores = BiEncoder_list_ranking_loss.get_similarity_function()
        return scores(uttr_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

    def calculate_loss(self,
                       uttr_vectors: T,
                       ctx_vectors: T,
                       rank_loss_index
                       ):
        dot_product_scores = self.get_scores(uttr_vectors, ctx_vectors)
        batch_size = dot_product_scores.size(0)
        scores = dot_product_scores.view(batch_size, -1)
        rank_demonstration_number = scores.size(-1)

        rank_demonstration_precision = torch.diagonal(scores)
        rank_demonstration_precision = torch.transpose(rank_demonstration_precision, 1, 0)
        rank_position = (1/torch.arange(1, 1+rank_demonstration_number).view(1, -1).repeat(batch_size, 1).
                         to(uttr_vectors))
        list_ranking_loss = lambda_list_ranking_loss(rank_demonstration_precision, rank_position)
        return list_ranking_loss


def dot_product_scores(uttr_vectors: T, ctx_vectors: T) -> T:
    r = torch.matmul(uttr_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def lambda_list_ranking_loss(rank_demonstration_precision, rank_position):
    device = rank_demonstration_precision.device
    rank_demonstration_precision = rank_demonstration_precision.clone()
    rank_position = rank_position.clone()
    _ = 1e8 if rank_demonstration_precision.dtype == torch.float32 else 1e4

    padded_rank_position = rank_position == -1
    rank_demonstration_precision[padded_rank_position] = float('-inf')
    rank_position[padded_rank_position] = float('-inf')

    # Sorting the true and predicted relevancy scores
    rank_demonstration_precision_sorted, index_predicted = rank_demonstration_precision.sort(descending=True, dim=-1)

    # Masking out the pairs of indices containing index of a padded element
    sorted_rank_demonstration_precision = torch.gather(rank_position, dim=1, index=index_predicted)
    sorted_distance = sorted_rank_demonstration_precision[:, :, None] - sorted_rank_demonstration_precision[:, None, :]
    padded_rank_mask = torch.isfinite(sorted_distance)
    padded_rank_mask = padded_rank_mask & (sorted_distance > 0)

    sorted_rank_demonstration_precision.clamp_(min=1e-5)
    rank_demonstration_indexes = 1./torch.arange(1, rank_demonstration_precision.shape[1] + 1).to(device)
    topk_weights = torch.abs(rank_demonstration_indexes.view(1, -1, 1) - rank_demonstration_indexes.view(1, 1, -1))

    # Clamping the array entries to maintain correct backprop (log(0) and division by 0)
    ranking_diffs =((rank_demonstration_precision_sorted[:, :, None] - rank_demonstration_precision_sorted[:, None, :]).
                    clamp(min=-50, max=50))
    ranking_diffs.masked_fill_(torch.isnan(ranking_diffs), min=1e-5)
    ranking_diffs_exp = torch.exp(-ranking_diffs)

    losses = torch.log(1. + ranking_diffs_exp) * topk_weights
    loss = torch.mean(losses[padded_rank_mask])

    logger.info('loss:{}'.format(loss))
    return loss
