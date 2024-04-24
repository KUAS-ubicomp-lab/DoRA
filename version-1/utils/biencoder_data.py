import collections
import logging
from typing import List

from datasets import dataset_dict, load_dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["post", "label"])


class BiEncoderSample(object):
    utterance: str
    candidate_samples: List[BiEncoderPassage]


@dataset_dict.add("rsdd")
def get_reddit():
    dataset = load_dataset("data/rsdd")
    return dataset


class RSDDDataset(Dataset):
    def __init__(
            self,
            file: str,
            task_name,
            setup_type,
            top_k,
            loss_type=None,
            rank_loss_top_sample=2,
            rank_loss_factor=1,
            rank_candidate_number=5,
            max_instances=None,
            selector: DictConfig = None,
            special_token: str = None,
            encoder_type: str = None,
            shuffle_candidates: bool = False,
            normalize: bool = False,
            query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_candidates=shuffle_candidates,
            query_special_suffix=query_special_suffix,
        )
        self.max_instances = max_instances
        assert loss_type in ['list_ranking']
        if loss_type == 'list_ranking':
            assert rank_candidate_number > 3, 'RSDDDataset.rank_candidate:{}'.format(rank_candidate_number)
        self.rank_candidate_num = rank_candidate_number
        self.loss_type = loss_type
        self.rank_loss_factor = rank_loss_factor
        self.rank_loss_top_sample = rank_loss_top_sample
        self.top_k = top_k
        self.max_instances = None
        self.task_name = task_name
        self.file = file
        self.data_files = []
        self.data = []
        self.normalize = normalize
        self.dataset = dataset_dict.IterableDatasetDict[task_name]()
        self.setup_type = setup_type
        assert self.setup_type in ["posts", "id", "label"]

        logger.info("Data files: %s", self.data_files)
