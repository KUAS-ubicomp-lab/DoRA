from dataclasses import dataclass
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
