from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn


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


def dot_product_scores(uttr_vectors: T, ctx_vectors: T) -> T:
    r = torch.matmul(uttr_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r
