from dataclasses import dataclass
from typing import Optional, Union, List, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        metadata = [x.pop("metadata") for x in features]
        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"id": x.pop("label")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
        batch_size = len(features)
        has_pad_mask = 'pad_mask' in features[0]
        if has_pad_mask:
            pad_mask_s = [x.pop('pad_mask') for x in features]
            for i, pad_mask in enumerate(pad_mask_s):
                pad_mask_s[i] = pad_mask[:self.max_length]

            max_len = max(list(map(lambda x: len(x), pad_mask_s)))
            pad_mask_s_tensor = torch.zeros(size=[batch_size, max_len])
            for i, pad_mask in enumerate(pad_mask_s):
                pad_mask_s_tensor[i, :len(pad_mask)] = pad_mask

            pad_mask_s_tensor.contiguous()
