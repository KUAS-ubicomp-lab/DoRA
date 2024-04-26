import json
import logging

import hydra
import hydra.utils as hutils
import torch
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .utils.collators import DataCollator

logger = logging.getLogger(__name__)


class DemonstrationsScorer:
    def __init__(self, cfg, accelerator) -> None:
        print('cfg:\n{}'.format(cfg))
        print('cfg.dataset_reader:\n{}'.format(cfg.dataset_reader))
        self.cuda_device = cfg.cuda_device
        self.dataset_reader = hutils.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        tokenizer = BertTokenizer.from_pretrained('bert-base')
        collator = DataCollator(tokenizer=tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=collator)
        self.model = hutils.instantiate("plm/wsw")
        logger.info('self.scorer pretrained model type:{}'.format(type(self.model)))
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        self.model = self.model.to(self.cuda_device)
        self.model = self.model.eval()

    def forward(self):
        if self.accelerator.is_main_process:
            data_loader = tqdm.tqdm(self.dataloader)
        else:
            data_loader = self.dataloader

        ranking_candidates = []
        for i, entry in enumerate(data_loader):
            meta_data = entry.pop("meta_data")
            position_ids = entry.attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(entry.attention_mask == 0, 1)

            with torch.no_grad():
                output = self.model(input_ids=entry.input_ids, attention_mask=entry.attention_mask,
                                    position_ids=position_ids)
                loss = self.cross_entropy_loss(entry=entry, output=output)

            for metadata, loss in zip(meta_data, loss):
                metadata['score'] = loss

            if i == 0:
                logger.info(f"Prompt: {meta_data[0]['prompt']}")
            ranking_candidates.extend(meta_data)

        with open(f"{self.output_file}tmp_{self.accelerator.device}.bin", "w") as f:
            json.dump(ranking_candidates, f)

    def cross_entropy_loss(self, entry, output):
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = entry.input_ids[..., 1:].contiguous()
        pad_token_id = self.dataset_reader.tokenizer.pad_token_id
        pad_mask = torch.nn.functional.pad(entry.labels,
                                           (shift_labels.shape[-1] - entry.labels.shape[-1], 0),
                                           value=pad_token_id)
        shift_labels.masked_fill_(pad_mask == pad_token_id, pad_token_id)
        loss_function = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1)).view(shift_labels.size())
        loss = loss.cpu().detach().numpy().tolist()
        return loss


@hydra.main(config_path="configs", config_name="scorer")
def main():
    accelerator = Accelerator()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
