import collections
import logging
from typing import List

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class BiEncoderSample(object):
    utterance: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
