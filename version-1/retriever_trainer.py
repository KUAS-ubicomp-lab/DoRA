import logging

import torch
from h5py.h5t import cfg
from omegaconf import DictConfig
from transformers import Trainer, PreTrainedTokenizerBase

from .biencoder import BiEncoder, BiEncoder_list_ranking_loss
from .demonstrations_finder import find_demonstrations
from .demonstrations_scorer import DemonstrationsScorer
from .utils import training_args, biencoder_data, collators

logger = logging.getLogger()


class RetrieverTrainer(object):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.biencoder = BiEncoder(cfg)
        self.loss_type = cfg.loss_type
        assert self.loss_type in ['list_ranking']
        self.train_dataset = biencoder_data
        self.training_args = training_args

        self.trainer = Trainer(
            model=self.biencoder.from_pretrained(cfg.model, cfg.model_config),
            args=self.training_args,
            train_dataset=self.train_dataset,
            tokenizer=PreTrainedTokenizerBase,
            data_collator=collators,
            loss=self.loss_type
        )

    def get_data_iterator(
            self,
            batch_size: int,
            is_train_set: bool,
            shuffle=True,
            shuffle_seed: int = 0,
            offset: int = 0,
            rank: int = 0,
    ):
        if is_train_set:
            self.train_dataset = biencoder_data
            self.training_args = training_args
            self.trainer = Trainer(
                model=self.biencoder.from_pretrained(cfg.model, cfg.model_config),
            )
            self.training_args.batch_size = batch_size
            self.training_args.is_train_set = is_train_set
            self.training_args.rank = rank
            self.training_args.shuffle_seed = shuffle_seed
            self.training_args.shuffle_seed = shuffle_seed
            self.training_args.offset = offset
            self.training_args.epochs = cfg.epochs
            self.training_args.steps_per_epoch = cfg.steps_per_epoch
            self.training_args.validation_steps = cfg.validation_steps
            self.training_args.shuffle = shuffle

    def train_epoch(self,
                    epoch: int,
                    data_iterator
                    ):
        cfg = self.cfg
        epoch_loss = 0

        cumulative_loss_step = cfg.train.cumulative_loss_step
        self.biencoder.train()
        epoch_batches = data_iterator.max_iterations
        for i, samples_batch in enumerate(
                data_iterator.iterate_data(epoch=epoch)
        ):
            if self.loss_type == 'list_ranking':
                biencoder_batches = BiEncoder.generate_biencoder_input(
                    samples_batch
                )
                find_demonstrations(biencoder_batches)
                ranking_batches = DemonstrationsScorer.ranking_candidates()
                loss = BiEncoder_list_ranking_loss.calculate_loss(ranking_batches, cfg)
                epoch_loss += loss.item()
                cumulative_loss_step += loss.item()
            else:
                raise NotImplementedError('self.loss_type:{}'.format(self.loss_type))

        epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        logger.info("Average loss per epoch=%f", epoch_loss)

    def train(self):
        with torch.device('cuda:1'):
            self.trainer.train()
            self.trainer.save_model(self.training_args.output_dir)
            self.trainer.tokenizer.save_pretrained(self.training_args.output_dir)
