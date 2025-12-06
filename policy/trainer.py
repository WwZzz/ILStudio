# import transformers
from transformers.trainer import Trainer

class BaseTrainer(Trainer):
    def __init__(self, *, train_loader=None, eval_loader=None, **kwargs):
        super().__init__(**kwargs)
        self._train_loader = train_loader
        self._eval_loader  = eval_loader

    def get_train_dataloader(self):
        if self._train_loader is None:
            raise ValueError("You passed train_loader=None")
        
        # Check if the underlying dataset is RLDS
        # RLDS datasets use tensorflow and don't work well with accelerator.prepare
        is_rlds = self._is_rlds_dataloader(self._train_loader)
        
        if is_rlds:
            # For RLDS datasets, return the loader without accelerator.prepare
            # This is because tensorflow datasets handle distributed training internally
            return self._train_loader
        else:
            # For regular PyTorch datasets, use accelerator.prepare
            return self.accelerator.prepare(self._train_loader)
    
    def _is_rlds_dataloader(self, dataloader):
        """Check if a DataLoader wraps an RLDS dataset"""
        try:
            # Try to access the underlying dataset
            if hasattr(dataloader, 'dataset'):
                dataset = dataloader.dataset
                
                # Check for RLDS dataset indicators
                # RLDS datasets are typically wrapped in IterableDataset with dlimp.DLataset
                if hasattr(dataset, 'dataset'):
                    import dlimp as dl
                    if isinstance(dataset.dataset, dl.DLataset):
                        return True
                
                # Also check if it's directly a DLataset
                import dlimp as dl
                if isinstance(dataset, dl.DLataset):
                    return True
            
            return False
        except (ImportError, AttributeError):
            # If we can't determine, assume it's not RLDS (safer default for PyTorch)
            return False

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_loader is None and eval_dataset is not None:
            return super().get_eval_dataloader(eval_dataset)
        
        # Similar logic for eval loader
        if self._eval_loader is not None:
            is_rlds = self._is_rlds_dataloader(self._eval_loader)
            if is_rlds:
                return self._eval_loader
            else:
                return self.accelerator.prepare(self._eval_loader)
        
        return self._eval_loader
    