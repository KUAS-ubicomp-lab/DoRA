import logging

import hydra
import hydra.utils as hutils
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .utils.collators import DataCollator

logger = logging.getLogger(__name__)


class WSWscorer:
    def __init__(self, cfg, accelerator) -> None:
        print('cfg:\n{}'.format(cfg))
        print('cfg.dataset_reader:\n{}'.format(cfg.dataset_reader))
        self.cuda_device = cfg.cuda_device
        self.dataset_reader = hutils.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        tokenizer = BertTokenizer.from_pretrained('bert-base')
        collator = DataCollator(tokenizer=tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=collator)
        self.model = hutils.instantiate(cfg.model)
        logger.info('self.scorer pretrained model type:{}'.format(type(self.model)))
        self.output_file = cfg.output_file
        self.accelerator = accelerator

        self.model = self.model.to(self.cuda_device)
        self.model = self.model.eval()
        self.cfg = cfg
        self.input_history = []


@hydra.main(config_path="configs", config_name="scorer")
def main():
    accelerator = Accelerator()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
