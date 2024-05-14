import logging

from omegaconf import DictConfig
from .biencoder import BiEncoder, BiEncoder_list_ranking_loss
from .demonstrations_finder import find_demonstrations
from .demonstrations_scorer import DemonstrationsScorer

logger = logging.getLogger()


class RetrieverTrainer(object):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.biencoder = BiEncoder(cfg)
        self.loss_type = cfg.loss_type

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
