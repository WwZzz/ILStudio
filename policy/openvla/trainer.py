import os
from typing import Optional
import transformers
from policy.trainer import BaseTrainer

class Trainer(BaseTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call=_internal_call)
        save_directory = output_dir if output_dir is not None else self.args.output_dir
        if self.is_world_process_zero():
            self.model.processor.save_pretrained(save_directory)