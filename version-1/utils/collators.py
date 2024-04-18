from dataclasses import dataclass
from typing import Optional, Union

from transformers import BertTokenizer
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollator:
    tokenizer: BertTokenizer
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
