# 9. Custom Trainers

This guide explains how to implement a custom training loop by leveraging the powerful `Trainer` class from the Hugging Face `transformers` library.

## The `BaseTrainer` Class

Instead of creating trainers from scratch, IL-Studio provides a `BaseTrainer` class in `policy/trainer.py`. This class inherits directly from `transformers.Trainer`, providing a wealth of built-in features out of the box:

*   **Distributed Training**: Automatic support for Data Parallel (DP) and Distributed Data Parallel (DDP).
*   **Mixed-Precision Training**: Enable `fp16` or `bf16` with a simple flag for faster training and reduced memory usage.
*   **Logging & Checkpointing**: Integrated with TensorBoard, Weights & Biases, and handles automatic saving of checkpoints.
*   **Gradient Accumulation**: Accumulate gradients over multiple steps to simulate a larger batch size.
*   **Learning Rate Scheduling**: Built-in support for various learning rate schedulers.

The `BaseTrainer` is slightly modified to accept a pre-constructed `DataLoader` directly, which is more convenient for robotics datasets.

## Step 1: Create Your Trainer Class

For most use cases, you don't need to write a completely new trainer. Instead, you can inherit from `BaseTrainer` and override specific methods. The most common method to override is `compute_loss`.

Create a new file, e.g., `policy/act/trainer.py`, for your policy-specific trainer.

```python
# In policy/act/trainer.py
from policy.trainer import BaseTrainer

class ACTTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return a tuple (loss, ...)
        """
        # The `model` is your `ACTPolicy` instance.
        # The `inputs` are a batch from your `DataLoader`.
        
        # The forward pass of your model should return a dictionary of losses.
        loss_dict = model(**inputs)
        
        # The trainer expects a single loss value.
        # You can sum them or return the primary loss.
        loss = loss_dict['loss']
        
        return (loss, loss_dict) if return_outputs else loss
```

## Step 2: Configure the Trainer

In your main training script or configuration, you will instantiate this trainer and pass it `TrainingArguments`.

```python
# In train.py (simplified)
from transformers import TrainingArguments
from policy.act.trainer import ACTTrainer
from policy.act.act import ACTPolicy

# 1. Load your model and datasets
model = ACTPolicy.from_pretrained(...)
train_dataset = ...
train_loader = DataLoader(train_dataset, ...)

# 2. Define Training Arguments
training_args = TrainingArguments(
    output_dir="ckpt/my_act_model",
    num_train_epochs=100,
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    fp16=True,  # Enable mixed-precision
    # ... and many more options
)

# 3. Instantiate your custom trainer
trainer = ACTTrainer(
    model=model,
    args=training_args,
    train_loader=train_loader, # Pass the pre-made loader
)

# 4. Start training
trainer.train()
```

By inheriting from the `transformers.Trainer`, you get a robust, feature-rich, and industry-standard training loop with minimal custom code.

