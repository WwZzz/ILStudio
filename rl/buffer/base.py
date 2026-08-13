"""Storage and lifecycle shared by all ILStudio RL buffers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from rl.base import MetaTransition, Rollout


@dataclass(frozen=True)
class BufferBatch:
    """Transitions selected from a buffer together with their logical indices."""

    transitions: Tuple[MetaTransition, ...]
    indices: np.ndarray
    rollout: Optional[Rollout] = None
    source_rollout: Optional[Rollout] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        indices = np.asarray(self.indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        if len(transitions) != len(indices):
            raise ValueError("transitions and indices must have equal length")
        if self.rollout is not None:
            if not isinstance(self.rollout, Rollout):
                raise TypeError("batch rollout must be Rollout or None")
            if self.rollout.transitions != transitions:
                raise ValueError("batch rollout must contain batch transitions")
        if self.source_rollout is not None:
            if not isinstance(self.source_rollout, Rollout):
                raise TypeError("batch source_rollout must be Rollout or None")
            if self.rollout is None:
                raise ValueError("source_rollout requires a selected batch rollout")
            source_transition_ids = {
                id(transition) for transition in self.source_rollout.transitions
            }
            if any(
                id(transition) not in source_transition_ids
                for transition in transitions
            ):
                raise ValueError("batch transitions must come from source_rollout")
            source_decision_ids = {
                decision.decision_id for decision in self.source_rollout.decisions
            }
            if any(
                decision.decision_id not in source_decision_ids
                for decision in self.rollout.decisions
            ):
                raise ValueError("batch decisions must come from source_rollout")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("batch metadata must be a mapping")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "indices", indices.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self):
        return iter(self.transitions)

    def select_decisions(self, decision_ids) -> "BufferBatch":
        """Select complete policy decisions while retaining rollout context."""

        if self.rollout is None:
            raise ValueError("decision selection requires a rollout-aware batch")
        decision_ids = tuple(decision_ids)
        if not decision_ids:
            raise ValueError("decision selection cannot be empty")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("decision selection cannot contain duplicates")
        requested = set(decision_ids)
        known = {decision.decision_id for decision in self.rollout.decisions}
        missing = requested - known
        if missing:
            raise KeyError(f"unknown rollout decision ids: {sorted(missing)!r}")
        selected_steps = tuple(
            step for step in self.rollout.steps if step.decision_id in requested
        )
        selected_decisions = tuple(
            decision
            for decision in self.rollout.decisions
            if decision.decision_id in requested
        )
        selected_rollout = Rollout(selected_steps, selected_decisions)
        transition_indices = {
            id(transition): int(index)
            for transition, index in zip(self.transitions, self.indices)
        }
        selected_indices = np.asarray(
            [transition_indices[id(step.transition)] for step in selected_steps],
            dtype=np.int64,
        )
        return type(self)(
            transitions=selected_rollout.transitions,
            indices=selected_indices,
            rollout=selected_rollout,
            source_rollout=self.source_rollout or self.rollout,
            metadata=self.metadata,
        )


class TransitionStorage:
    """In-memory logical sequence with optional bounded ring eviction.

    The storage owns ordering, indexing, capacity, and serialization.  Buffer
    subclasses add rollout or replay semantics without reimplementing storage.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        capacity: Optional[int] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive or None")
        if overwrite and capacity is None:
            raise ValueError("overwrite storage requires a finite capacity")

        self.capacity = capacity
        self.overwrite = bool(overwrite)
        self._items = [] if capacity is None else [None] * capacity
        self._start = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[MetaTransition]:
        for index in range(len(self)):
            yield self[index]

    def _normalize_index(self, index: int) -> int:
        index = int(index)
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("transition index out of range")
        return index

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._size))]

        logical_index = self._normalize_index(index)
        if self.capacity is None:
            return self._items[logical_index]
        physical_index = (self._start + logical_index) % self.capacity
        return self._items[physical_index]

    @staticmethod
    def _validate_transition(transition: MetaTransition) -> None:
        if not isinstance(transition, MetaTransition):
            raise TypeError("buffer entries must be MetaTransition")

    def append(self, transition: MetaTransition) -> Optional[MetaTransition]:
        """Append a transition and return an evicted oldest item, if any."""

        self._validate_transition(transition)
        if self.capacity is None:
            self._items.append(transition)
            self._size += 1
            return None

        if self._size < self.capacity:
            physical_index = (self._start + self._size) % self.capacity
            self._items[physical_index] = transition
            self._size += 1
            return None

        if not self.overwrite:
            raise BufferError(f"buffer capacity {self.capacity} has been reached")

        evicted = self._items[self._start]
        self._items[self._start] = transition
        self._start = (self._start + 1) % self.capacity
        return evicted

    def extend(
        self,
        transitions: Iterable[MetaTransition],
    ) -> Tuple[MetaTransition, ...]:
        transitions = tuple(transitions)
        for transition in transitions:
            self._validate_transition(transition)

        if (
            not self.overwrite
            and self.capacity is not None
            and self._size + len(transitions) > self.capacity
        ):
            raise BufferError(f"buffer capacity {self.capacity} would be exceeded")

        evicted = []
        for transition in transitions:
            old_transition = self.append(transition)
            if old_transition is not None:
                evicted.append(old_transition)
        return tuple(evicted)

    def clear(self) -> None:
        self._items = [] if self.capacity is None else [None] * self.capacity
        self._start = 0
        self._size = 0

    def to_list(self) -> list:
        return list(self)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "overwrite": self.overwrite,
            "transitions": self.to_list(),
        }

    def validate_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("storage state must be a dict")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported transition storage state version")
        if state.get("capacity") != self.capacity:
            raise ValueError(
                f"storage capacity mismatch: {state.get('capacity')} != {self.capacity}"
            )
        if bool(state.get("overwrite")) != self.overwrite:
            raise ValueError("storage overwrite mode mismatch")
        transitions = state.get("transitions")
        if not isinstance(transitions, (list, tuple)):
            raise TypeError("storage transitions must be a list or tuple")
        if self.capacity is not None and len(transitions) > self.capacity:
            raise ValueError("storage state exceeds configured capacity")
        for transition in transitions:
            self._validate_transition(transition)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.validate_state_dict(state)
        transitions = tuple(state["transitions"])
        self.clear()
        self.extend(transitions)


class BaseBuffer(ABC):
    """Common buffer API backed by ``TransitionStorage``."""

    STATE_VERSION = 1

    def __init__(self, storage: TransitionStorage) -> None:
        if not isinstance(storage, TransitionStorage):
            raise TypeError("storage must be TransitionStorage")
        self.storage = storage

    @property
    @abstractmethod
    def buffer_type(self) -> str:
        """Stable type name used to validate checkpoints."""

    @property
    def capacity(self) -> Optional[int]:
        return self.storage.capacity

    @property
    def transitions(self) -> Tuple[MetaTransition, ...]:
        return tuple(self.storage)

    def __len__(self) -> int:
        return len(self.storage)

    @property
    def num_env_steps(self) -> int:
        """Atomic environment steps represented by this buffer."""

        return len(self)

    def __getitem__(self, index):
        return self.storage[index]

    def _validate_extend(self, transitions: Sequence[MetaTransition]) -> None:
        for transition in transitions:
            TransitionStorage._validate_transition(transition)

    def _after_extend(
        self,
        transitions: Sequence[MetaTransition],
        evicted: Sequence[MetaTransition],
    ) -> None:
        del transitions, evicted

    def add(self, transition: MetaTransition) -> Optional[MetaTransition]:
        self._validate_extend((transition,))
        evicted = self.storage.append(transition)
        self._after_extend(
            (transition,),
            () if evicted is None else (evicted,),
        )
        return evicted

    def extend(
        self,
        transitions: Iterable[MetaTransition],
    ) -> Tuple[MetaTransition, ...]:
        transitions = tuple(transitions)
        self._validate_extend(transitions)
        evicted = self.storage.extend(transitions)
        self._after_extend(transitions, evicted)
        return evicted

    def get_batch(self, indices: Iterable[int]) -> BufferBatch:
        index_array = np.asarray(list(indices), dtype=np.int64)
        if index_array.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        transitions = tuple(self.storage[int(index)] for index in index_array)
        return BufferBatch(transitions=transitions, indices=index_array)

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[BufferBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                break
            yield self.get_batch(batch_indices)

    def clear(self) -> None:
        self.storage.clear()
        self._reset_metadata()

    def _state_metadata(self) -> Dict[str, Any]:
        return {}

    def _validate_state_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        num_transitions: int,
    ) -> None:
        del metadata, num_transitions

    def _load_state_metadata(self, metadata: Dict[str, Any]) -> None:
        del metadata

    def _reset_metadata(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "buffer_type": self.buffer_type,
            "storage": self.storage.state_dict(),
            "metadata": self._state_metadata(),
        }

    def checkpoint_state_dict(self) -> Dict[str, Any]:
        """Return replay persistence state.

        In-memory buffers use their ordinary state dict. A future disk-backed
        buffer can override this hook to return shard or database references
        without teaching RLRunner about its storage backend.
        """

        return self.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("buffer state must be a dict")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported buffer state version")
        if state.get("buffer_type") != self.buffer_type:
            raise ValueError(
                f"buffer type mismatch: {state.get('buffer_type')} != {self.buffer_type}"
            )

        storage_state = state.get("storage")
        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("buffer metadata must be a dict")
        self.storage.validate_state_dict(storage_state)
        self._validate_state_metadata(
            metadata,
            num_transitions=len(storage_state["transitions"]),
        )

        self.storage.load_state_dict(storage_state)
        self._load_state_metadata(metadata)

    def load_checkpoint_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay persistence state produced by the matching hook."""

        self.load_state_dict(state)
