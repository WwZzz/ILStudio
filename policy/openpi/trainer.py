import os
from typing import Optional
import transformers
from peft import PeftModel

class Trainer(transformers.Trainer):
    """
    A versatile Trainer that handles saving correctly for both PEFT and full fine-tuning.
    
    - If the model is a PeftModel, it saves a self-contained checkpoint including
      the adapter, the base model, and the tokenizer.
    - If the model is a regular Transformers model, it behaves like the default Trainer,
      saving the entire fine-tuned model.
    """
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # --- Step 1: Always call the original save_model behavior first ---
        # For a normal model, this saves the entire model.
        # For a PeftModel, this saves the adapter and its config.
        super().save_model(output_dir, _internal_call)

        # --- Step 2: If it's a PEFT model, add the extra saving steps ---
        # This check is the core of our flexible logic.
        if isinstance(self.model, PeftModel):
            if self.is_world_process_zero():
                # Get the correct output directory
                output_dir = output_dir if output_dir is not None else self.args.output_dir
                
                # Get the underlying base model
                base_model = self.model.get_base_model()

                # Save the base model and tokenizer to the same directory
                print(f"Additionally saving base model and tokenizer to {output_dir}")
                base_model.save_pretrained(output_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)