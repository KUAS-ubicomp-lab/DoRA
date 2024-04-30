import hydra
import tqdm
import json
import numpy as np
import multiprocessing

from .utils import biencoder_data


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
        self.corpus = biencoder_data.RSDDDataset()
        print("Finished creating the corpus")


class RetrieverTask:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_name(cls, name):
        task_list = {}
        return task_list[name]


def search(tokenized_query, is_train, idx, candidates):
    global retriever
    scores = retriever.get_scores(tokenized_query)
    near_demonstration_ids = list(np.argsort(scores)[::-1][:candidates])
    near_demonstration_ids = near_demonstration_ids[1:] if is_train else near_demonstration_ids
    return [{"id": int(a)} for a in near_demonstration_ids], idx


def _search(args):
    tokenized_query, is_train, idx, candidates = args
    return search(tokenized_query, is_train, idx, candidates)


def find_demonstrations(ctx):
    global retriever
    demonstrations_finder = DemonstrationsFinder(ctx).corpus
    tokenized_queries = [demonstrations_finder.task.get_field(input_query)
                         for input_query in demonstrations_finder.task.dataset]

    demonstrations_pool = multiprocessing.Pool(processes=None, initializer=retriever, initargs=(demonstrations_finder,))
    demonstrations_pool.close()

    list_of_demonstrations = list(demonstrations_finder.task.dataset)[:100]
    ctx_pre = [[tokenized_query, demonstrations_finder.is_train, idx, demonstrations_finder.candidates]
               for idx, tokenized_query in enumerate(tokenized_queries)]
    ctx_post = []
    with tqdm.tqdm(total=len(ctx_pre)) as progress_bar:
        for start, end in enumerate(demonstrations_pool.imap_unordered(_search, ctx_pre)):
            progress_bar.update()
            ctx_post.append(end)
    for ctx, idx in ctx_post:
        list_of_demonstrations[idx]['idx'] = idx
        list_of_demonstrations[ctx]['ctx_candidates'] = ctx
    return list_of_demonstrations


@hydra.main(config_path="config", config_name="demonstrations_finder")
def main(ctx):
    list_of_demonstrations = find_demonstrations(ctx=ctx)
    with open(ctx.output_path, "w") as f:
        json.dump(list_of_demonstrations, f)


if __name__ == '__main__':
    main()
