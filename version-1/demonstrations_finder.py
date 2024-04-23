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


class RetrieverTask:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_name(cls, name):
        task_list = {}
        return task_list[name]
