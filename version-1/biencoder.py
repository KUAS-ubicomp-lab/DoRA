import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


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


def dot_product_scores(uttr_vectors: T, ctx_vectors: T) -> T:
    r = torch.matmul(uttr_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(uttr_vector: T, ctx_vectors: T):
    return F.cosine_similarity(uttr_vector, ctx_vectors, dim=1)