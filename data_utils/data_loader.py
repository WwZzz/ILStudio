import os
import threading
import time
import numpy as np
import itertools
import torch
import fnmatch
import queue
import json
import warnings
import importlib
import torch.distributed as dist
import dlimp as dl
from typing import Optional, Any, List, Union, Tuple
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from .dataset_wrappers import WrappedDataset, WrappedIterableDataset, MapToIterableDataset

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

PrefetchResult = Union[
    Tuple[Any, List[Any], int], # (iterator, prefetched_batches_list, epoch_fetched)
    Exception,
    type(StopIteration) # Use StopIteration class as signal
]

class BackgroundPrefetcher:
    """
    Aggressively prefetches the next epoch's iterator and *multiple* initial batches (k) in a background thread to completely hide cold 
    start latency and worker ramp-up time.
    
    Requires persistent_workers=True on the wrapped DataLoader.
    Requires the wrapped DataLoader to have a __len__ method.
    Does not handle skipping batches internally.
    """
    
    def __init__(self, loader: DataLoader, num_batches_to_prefetch: Optional[int] = None): 
        internal_len = -1
        try:
            internal_len = len(loader)
            if internal_len <= 0:
                raise ValueError("DataLoader length must be positive.")
        except TypeError:
            raise TypeError("BackgroundPrefetcher requires the wrapped DataLoader to have a __len__ method.")
             
        # logger.info(f"[BGPrefetcher] Initializing... Wrapped DataLoader len={internal_len}")
        
        if not hasattr(loader, 'persistent_workers') or not loader.persistent_workers:
            raise ValueError("BackgroundPrefetcher requires persistent_workers=True on the wrapped DataLoader.")
            
        self.loader = loader
        
        # Determine k: Number of batches to prefetch aggressively
        if num_batches_to_prefetch is None:
            # Default heuristic: prefetch factor + num workers might be enough
            self.num_batches_to_prefetch = getattr(loader, 'prefetch_factor', 2) + getattr(loader, 'num_workers', 1)
        else:
            self.num_batches_to_prefetch = max(1, num_batches_to_prefetch) # Must prefetch at least 1
        # logger.info(f"[BGPrefetcher] Will prefetch {self.num_batches_to_prefetch} batches per epoch.")

        self.epoch = 0 # Start at epoch 0
        self.prefetch_thread: Optional[threading.Thread] = None
        self.data_queue = queue.Queue(maxsize=1) 
        self._stop_event = threading.Event()

        # --- Initial blocking prefetch for Epoch 0 ---
        # logger.info(f"[BGPrefetcher] Performing initial (blocking) prefetch for starting Epoch {self.epoch} ({self.num_batches_to_prefetch} batches)... ---")
        start_time = time.time()
        self._prefetch_job(self.epoch) # Run directly, puts result in queue
        # logger.info(f"[BGPrefetcher] Initial prefetch done in {time.time() - start_time:.2f}s ---")
        # Result for Epoch 0 is now in the queue.

    def _prefetch_job(self, epoch_to_fetch: int):
        """
        Runs in background thread. Sets sampler, creates iterator, 
        fetches the first `k` batches, puts result in queue.
        """
        thread_id = threading.get_ident()
        if self._stop_event.is_set():
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Stop event set, skipping prefetch for Epoch {epoch_to_fetch}.")
            try: self.data_queue.put_nowait(StopIteration) 
            except queue.Full: pass 
            return
            
        prefetched_batches = []
        epoch_iter = None
        try:
            # 1. Set sampler epoch
            if isinstance(self.loader.sampler, Sampler) and hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(epoch_to_fetch)
                
            # 2. Create iterator and fetch first k batches (cold start bottleneck spread over k batches)
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Starting cold start for Epoch {epoch_to_fetch} (fetching {self.num_batches_to_prefetch} batches)... ---")
            start_cold_start = time.time()
            
            epoch_iter = iter(self.loader) # <-- Create iterator
            
            for i in range(self.num_batches_to_prefetch):
                batch = next(epoch_iter) # <-- Fetch batch (potentially blocking)
                prefetched_batches.append(batch)
                # Log timing for the very first batch fetch of the cold start
                if i == 0:
                    first_batch_time = time.time() - start_cold_start
                    # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Fetched *first* batch of Epoch {epoch_to_fetch} in {first_batch_time:.2f}s.")

            total_cold_start_time = time.time() - start_cold_start
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) *Finished* cold start for Epoch {epoch_to_fetch} (fetched {len(prefetched_batches)} batches) in {total_cold_start_time:.2f}s total. ---")
            
            # 3. Put successful result in queue (iterator is now advanced past the prefetched batches)
            self.data_queue.put((epoch_iter, prefetched_batches, epoch_to_fetch))
            
        except StopIteration:
            # This happens if the dataloader has less than k batches
            total_cold_start_time = time.time() - start_cold_start
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) DataLoader exhausted during prefetch for Epoch {epoch_to_fetch} after fetching {len(prefetched_batches)} batches (needed {self.num_batches_to_prefetch}). Total time: {total_cold_start_time:.2f}s.")
            # Put the partially fetched batches (if any) and signal exhaustion
            # We pass None for the iterator as it's exhausted
            self.data_queue.put((None, prefetched_batches, epoch_to_fetch)) 
             
        except Exception as e:
            import traceback
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Prefetch failed for Epoch {epoch_to_fetch}: {e}\n{traceback.format_exc()} ---")
            self.data_queue.put(e)

    def _start_background_prefetch(self, epoch_to_fetch: int):
        """ Safely starts the background prefetch thread """
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            # logger.info(f"[BGPrefetcher] (Main) Joining previous prefetch thread (Epoch {epoch_to_fetch-1})...")
            self.prefetch_thread.join(timeout=60) # Increased timeout
            # if self.prefetch_thread.is_alive():
                # logger.info(f"[BGPrefetcher] (Main) Previous prefetch thread did not join cleanly!")
            # else:
            #      # logger.info(f"[BGPrefetcher] (Main) Previous prefetch thread joined.")
            
        if self._stop_event.is_set():
            # logger.info(f"[BGPrefetcher] (Main) Stop event set, not starting prefetch for Epoch {epoch_to_fetch}.")
            return

        # logger.info(f"[BGPrefetcher] (Main) Starting background prefetch thread for Epoch {epoch_to_fetch}... ---")
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_job, 
            args=(epoch_to_fetch,),
            daemon=True
        )
        self.prefetch_thread.start()

    def __iter__(self):
        """ Trainer calls this at the start of each epoch (e.g., Epoch N) """
        
        current_epoch_requested = self.epoch 
        # logger.info(f"[BGPrefetcher] (Main) __iter__ called for Epoch {current_epoch_requested}. Waiting for prefetched data... ---")
        
        # 1. Block and get the result for the CURRENT epoch (Epoch N)
        try:
            data: PrefetchResult = self.data_queue.get(timeout=1800) # 30 minutes timeout
        except queue.Empty:
            # logger.info(f"Timeout waiting for prefetched data for Epoch {current_epoch_requested}.")
            raise TimeoutError(f"Timeout waiting for prefetched data for Epoch {current_epoch_requested}.")

        # Check for error signals
        if isinstance(data, Exception): 
            # logger.info(f"[BGPrefetcher] (Main) Received exception from prefetch job for Epoch {current_epoch_requested}.")
            raise data
        # StopIteration signal is handled within the generator now
            
        # Unpack successful prefetch result (iterator might be None if exhausted during prefetch)
        current_iter, prefetched_batches, fetched_epoch = data
        # logger.info(f"[BGPrefetcher] (Main) Successfully received {len(prefetched_batches)} prefetched batches for Epoch {fetched_epoch}. ---")

        if fetched_epoch != current_epoch_requested:
            # logger.info(f"[BGPrefetcher] Epoch mismatch! Expected {current_epoch_requested}, got {fetched_epoch}. Adjusting internal epoch counter.")
            current_epoch_requested = fetched_epoch # Trust the fetched data

        # 2. Start prefetching the *NEXT* epoch (N+1) in the background *NOW*
        next_epoch_to_prefetch = current_epoch_requested + 1
        self._start_background_prefetch(next_epoch_to_prefetch)

        # 3. Define and return the generator for the CURRENT epoch (Epoch N)
        # Pass the prefetched batches and the (potentially None) iterator
        return self._generator(prefetched_batches, current_iter, current_epoch_requested)


    def _generator(self, prefetched_list: List[Any], iterator: Optional[Any], epoch_num: int):
        """ Generator yields prefetched batches, then remaining batches for one epoch. """
        total_yielded_this_epoch = 0
        try:
            # logger.info(f"[BGPrefetcher] (Generator) Starting Epoch {epoch_num} with {len(prefetched_list)} prefetched batches.")
            
            # 1. Yield the prefetched batches first
            for i, batch in enumerate(prefetched_list):
                # # logger.info(f"[BGPrefetcher] (Generator) Yielding prefetched batch {i} for Epoch {epoch_num}...")
                yield batch
                total_yielded_this_epoch += 1

            # 2. Yield remaining batches from the iterator (if it exists and wasn't exhausted)
            if iterator is not None:
                # # logger.info(f"[BGPrefetcher] (Generator) Yielding remaining batches for Epoch {epoch_num}...")
                for batch in iterator:
                    yield batch
                    total_yielded_this_epoch += 1
            # else:
                # logger.info(f"[BGPrefetcher] (Generator) Iterator was already exhausted during prefetch for Epoch {epoch_num}.")


            # logger.info(f"\n[BGPrefetcher] (Generator) Epoch {epoch_num} exhausted after {total_yielded_this_epoch} batches.")

        except Exception as e:
             # logger.info(f"[BGPrefetcher] (Generator) Error during iteration for Epoch {epoch_num}: {e}", exc_info=True)
             raise e
        finally:
            # Update epoch counter AFTER the generator is exhausted
            # logger.info(f"[BGPrefetcher] (Generator) Incrementing epoch counter after Epoch {epoch_num} finished.")
            self.epoch = epoch_num + 1

    def __len__(self):
        # Must return the correct length for Trainer's calculations
        return len(self.loader) 
    
    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def batch_sampler(self):
        return getattr(self.loader, 'batch_sampler', None)

    def set_epoch(self, epoch: int):
        # logger.info(f"[BGPrefetcher] (Main) Trainer called set_epoch({epoch}). Internal epoch is {self.epoch}. (Call ignored by prefetcher internal logic)")
        pass

    def shutdown(self):
        # logger.info("[BGPrefetcher] Shutting down...")
        self._stop_event.set()
        # Try to unblock queue if thread is waiting to put
        try: self.data_queue.put(StopIteration, block=False) 
        except queue.Full: # logger.info("[BGPrefetcher] Queue full during shutdown signal.")
            pass
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            # logger.info("[BGPrefetcher] Joining background thread...")
            self.prefetch_thread.join(timeout=10) 
            # if self.prefetch_thread.is_alive():
                # logger.info("[BGPrefetcher] Background thread did not join cleanly.")
        # Drain queue
        while not self.data_queue.empty():
            try: self.data_queue.get_nowait()
            except queue.Empty: break
        # logger.info("[BGPrefetcher] Shutdown complete.")
 
def get_dataloader(train_dataset, val_dataset=None, processor=None, collator=None, args=None):
    """
    Create DataLoader from single dataset or multiple datasets.
    
    Args:
        train_dataset: Single dataset or list of datasets
        val_dataset: Optional validation dataset
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
    
    Returns:
        (train_loader, eval_loader)
    """
    if isinstance(train_dataset, list):
        # Multiple datasets - use torchdata for mixing
        
        train_loader = _create_mixed_dataloader(train_dataset, processor, collator, args)
        
        # Handle validation dataset
        eval_loader = None
        if val_dataset is not None:
            if isinstance(val_dataset, list):
                eval_loader = _create_mixed_dataloader(val_dataset, processor, collator, args, is_training=False)
            else:
                eval_loader = _create_single_dataloader(val_dataset, processor, collator, args, is_training=False)
        
        return train_loader, eval_loader
    
    else:
        # Single dataset - use existing logic
        return _create_single_dataloader(train_dataset, processor, collator, args, is_training=True)

def is_rlds_data(ds):
    return isinstance(ds, dl.DLataset)

def is_map_data(dataset):
    return hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')

def is_iter_data(dataset):
    return hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))

def _create_single_dataloader(dataset, processor, collator, args, is_training=True):
    """Create DataLoader for a single dataset (map-style or iterable)"""
    # Identify the type of the dataset: iter or map
    if is_map_data(dataset):
        wrapped_data = WrappedDataset(dataset, processor)
        sampler = DistributedSampler(wrapped_data) if is_training and is_distributed() else None
        persistent_workers = getattr(args, 'dataloader_persistent_workers', False) if args.dataloader_num_workers>0 else False
        prefetch_factor = getattr(args, 'dataloader_prefetch_factor', 2) if args.dataloader_num_workers>0 else None
        loader = DataLoader(
            wrapped_data,
            batch_size=args.per_device_train_batch_size,
            shuffle=is_training,
            # sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
            drop_last=is_training,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        if getattr(args, 'background_prefetch', False):
            loader = BackgroundPrefetcher(loader, getattr(args, 'dataloader_prefetch_factor', 2))
    elif is_iter_data(dataset):
        if hasattr(dataset, 'dataset') and is_rlds_data(dataset.dataset):
            # RLDS dataset
            # set tf data options here
            import tensorflow as tf
            dataset.dataset = dataset.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            dataset.dataset = dataset.dataset.with_ram_budget(1)
            wrapped_data = WrappedIterableDataset(dataset, processor)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            loader = DataLoader(
                wrapped_data,  
                batch_size=batch_size,
                num_workers=args.dataloader_num_workers,
                collate_fn=collator,
                persistent_workers=True,
            )
        else:
            # Pytorch Iterable dataset
            # Wrap iterable dataset with processor
            try:
                from torchdata.datapipes.iter import IterableWrapper
            except ImportError:
                raise ImportError(
                    "torchdata is required for iterable dataset support. "
                    "Install it with: pip install torchdata"
                )
            
            wrapped_data = WrappedIterableDataset(dataset, processor)
            pipe = IterableWrapper(wrapped_data, deepcopy=False)
            # For iterable datasets, we cannot use DistributedSampler
            pipe = pipe.sharding_filter()
            pipe = pipe.cycle()
            buffer_size = getattr(args, 'shuffle_buffer_size', 1000)
            pipe = pipe.shuffle(buffer_size=buffer_size)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            #### Prefetch is handled by DataLoader
            loader = DataLoader(
                pipe,
                batch_size=batch_size,  
                # num_workers=args.dataloader_num_workers,
                collate_fn=collator, 
                pin_memory=args.dataloader_pin_memory,
                # persistent_workers=args.dataloader_num_workers>0,
                drop_last=True,
            )
    else:
        raise ValueError("Dataset must be either map-style or iterable.")
    return loader, None if is_training else loader

def _create_mixed_dataloader(datasets, processor, collator, args, is_training=True):
    """
    Create DataLoader for multiple datasets using torchdata pipeline.
    
    Args:
        datasets: List of datasets (can be map-style or iterable)
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
        is_training: Whether this is for training (affects shuffle, drop_last)
    
    Returns:
        DataLoader with mixed pipeline
    """
    # Create DataLoader for MapStyleDataset, IterableDataset, and RLDSDataset, respectively
    all_map_datasets, all_iter_datasets, all_rlds_datasets = [], [], []
    for data in datasets:
        if is_map_data(data): all_map_datasets.append(data)
        elif is_iter_data(data):
            if is_rlds_data(data):
                all_rlds_datasets.append(data)
            else:
                all_iter_datasets.append(data)
        else:
            raise TypeError("Dataset must be either map-style or iterable.")
    # Mix map-style datasets using ConcatDataset
    if len(all_map_datasets)>0:
        mixed_map_data = torch.utils.data.ConcatDataset(all_map_datasets)
        map_loader = _create_single_dataloader(mixed_map_data, processor, collator, args, is_training=is_training)
    else:
        map_loader = None
    # mix iterable datasets using huggingface's datasets
    if len(all_iter_datasets)>0:
        from datasets import interleave_datasets
        mixed_iter_data = interleave_datasets(all_iter_datasets
        )
        iter_loader = _create_single_dataloader(mixed_iter_data, processor, collator, args, is_training=is_training)
    else:
        iter_loader = None
    # mix rlds datasets using tf.data
    if len(all_rlds_datasets)>0:
        import dlimp as dl
        mixed_rlds_data = dl.DLataset.sample_from_datasets([ds.dataset for ds in all_rlds_datasets])
        rlds_loader = _create_single_dataloader(mixed_rlds_data, processor, collator, args, is_training=is_training)
    else:
        rlds_loader = None
    # Combine all loaders using SampleMultiplexer    
    if rlds_loader is None and iter_loader is None: return map_loader
    elif map_loader is None and rlds_loader is None: return iter_loader
    elif map_loader is None and iter_loader is None: return rlds_loader
    else:
        raise NotImplementedError("Mixing map-style, iterable, and RLDS datasets is not yet implemented.")
    
