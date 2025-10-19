import transformers

class BaseTrainer(transformers.Trainer):
    def __init__(self, *, train_loader=None, eval_loader=None, **kwargs):
        super().__init__(**kwargs)
        self._train_loader = train_loader
        self._eval_loader  = eval_loader

    def get_train_dataloader(self):
        if self._train_loader is None:
            raise ValueError("You passed train_loader=None")
        return self._train_loader

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_loader is None and eval_dataset is not None:
            return super().get_eval_dataloader(eval_dataset)
        return self._eval_loader