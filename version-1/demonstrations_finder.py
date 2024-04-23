import numpy as np


class DemonstrationsFinder:
    def __init__(self, ctx) -> None:
        self.output_path = ctx.output_path
        self.task_name = ctx.task_name
        assert ctx.dataset_split in ['train', 'validation', 'test']
        self.is_train = ctx.dataset_split == 'train'
        self.setup_type = ctx.setup_type
        assert self.setup_type in ['classification', 'regression']
        self.task = RetrieverTask.from_name(ctx.task_name)(ctx.dataset_split,
                                                           ctx.setup_type,
                                                           ds_size=None if "ds_size" not in ctx else ctx.ds_size)
        print("started creating the corpus")
        self.corpus = self.task.get_corpus()


def search(tokenized_query, is_train, idx, candidates):
    global retriever
    scores = retriever.get_scores(tokenized_query)
    near_demonstration_ids = list(np.argsort(scores)[::-1][:candidates])
    near_demonstration_ids = near_demonstration_ids[1:] if is_train else near_demonstration_ids
    return [{"id": int(a)} for a in near_demonstration_ids], idx


def _search(args):
    tokenized_query, is_train, idx, candidates = args
    return search(tokenized_query, is_train, idx, candidates)


class RetrieverTask:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_name(cls, name):
        task_list = {}
        return task_list[name]
