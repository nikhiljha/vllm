from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vllm.prefix import Prefix


@dataclass
class PrefixPolicy(ABC):
    num_gpu_blocks: int
    num_cpu_blocks: int
    num_disk_blocks: int

    @abstractmethod
    def hit_prefix(self, prefix: Prefix):
        """Called when a Prefix is used by a SequenceGroup. Must be called immediately after the hit (i.e. before prefix internal state is changed)."""
        ...

    # @abstractmethod
    # def miss_prefix(self, prefix: Prefix):
    #     """Called when a Prefix is not used by a SequenceGroup."""
    #     ...

    @abstractmethod
    def evict_prefix_gpu(self) -> Prefix | None:
        """Return the next Prefix to evict to CPU Memory, or None if no Prefixes can be evicted."""
        ...

    @abstractmethod
    def evict_prefix_cpu(self) -> Prefix | None:
        """Return the next Prefix to evict to Disk, or None if no Prefixes can be evicted."""
        ...


@dataclass
class FIFOPrefixPolicy(PrefixPolicy):
    """A prefix policy that evicts the oldest unused prefix from GPU first."""

    gpu_prefix_queue: list[Prefix] = field(default_factory=list)
    cpu_prefix_queue: list[Prefix] = field(default_factory=list)

    def hit_prefix(self, prefix: Prefix):
        if prefix.on_gpu:
            self.gpu_prefix_queue.remove(prefix)
            self.gpu_prefix_queue.append(prefix)
        elif prefix.on_cpu:
            self.cpu_prefix_queue.remove(prefix)
            self.gpu_prefix_queue.append(prefix)
        else:
            self.gpu_prefix_queue.append(prefix)
    
    def need_to_evict_gpu(self) -> bool:
        return sum(prefix.num_blocks for prefix in self.gpu_prefix_queue) > self.num_gpu_blocks // 22
    
    # def miss_prefix(self, _):
    #     # We don't need to do anything for FIFO.
    #     pass

    def evict_prefix_gpu(self) -> Prefix | None:
        if len(self.gpu_prefix_queue) != 0:
            # Pick leftmost prefix where refcount is 0
            for prefix in self.gpu_prefix_queue:
                if prefix.ref_count == 0:
                    self.gpu_prefix_queue.remove(prefix)
                    return prefix
        return None
    
    def evict_prefix_cpu(self) -> Prefix | None:
        if len(self.cpu_prefix_queue) != 0:
            # Pick leftmost prefix where refcount is 0
            for prefix in self.cpu_prefix_queue:
                if prefix.ref_count == 0:
                    self.cpu_prefix_queue.remove(prefix)
                    return prefix
        return None


class PrefixPolicyFactory:
    _POLICY_REGISTRY = {
        "fifo": FIFOPrefixPolicy,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> PrefixPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
