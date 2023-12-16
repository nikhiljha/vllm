from abc import ABC, abstractmethod
from collections import defaultdict, deque
import enum
import random
from typing import Callable, List
from datetime import datetime
from enum import Enum

# Define the prefix class, which is a collection of prefix (a sequence of tokens).
# The class contains the following main methods:
# 1. A match method that checks if a prefix matches a given sequence of tokens.
# 2. A swapping method that can load or offload the prefix to or from GPU
# 3. An update_frequency method that updates the frequency of the prefix.
# 4. A get_status method that tells if the prefix is on GPU or not.


class PrefixLocation(Enum):
    """PrefixLocation describes where a prefix is physically stored."""

    GPU = enum.auto()
    CPU = enum.auto()
    DISK = enum.auto()
    # If a prefix is in the NONE state, it may be removed from the prefix pool.
    NONE = enum.auto()


class Prefix:
    def __init__(self, prefix_id, token_ids, block_size):
        self.prefix_id = prefix_id
        self.token_ids = token_ids
        self.length = len(token_ids)
        print("prefix length: ", self.length)
        print("block size: ", block_size)
        assert self.length % block_size == 0
        self.location = PrefixLocation.NONE
        self.block_table = None
        # a lock to prevent multiple sequence from calculating the same prefix
        # gets set when the prefix hasn't been calculated yet, or is being swapped in
        self.load_in_progress = False
        # whether or not this prefix has been used before swap or eviction
        self.utilized = False

    def get_block_table_num(self) -> List[int]:
        return [block.block_number for block in self.block_table]

    def match(self, tokens):
        return tokens[: self.length] == self.token_ids

    # whether the prefix is on GPU or not
    def is_on_gpu(self):
        return self.location == PrefixLocation.GPU

    def get_load_in_progress(self) -> bool:
        return self.load_in_progress

    def set_load_in_progress(self, load_in_progress) -> None:
        self.load_in_progress = load_in_progress

    def set_location(self, location: PrefixLocation) -> None:
        if self.location != location:
            self.utilized = False
        self.location = location

    def get_location(self) -> PrefixLocation:
        return self.location

    def get_length(self):
        return self.length
    
    def utilize(self) -> bool:
        already_utilized = self.utilized
        self.utilized = True
        return already_utilized


# Define the prefix pool class, which is a collection of prefixes.
# The class contains the following main methods:
# 1. add a prefix to the pool, with a computed hash
# 2. TODO: create subprefix, if one is a prefix of the other: they can share some memory blocks
# 3. efficient_search: given a sequence of tokens, find the longest prefix in the pool that matches the sequence
# 4. fixed_search: given the prefix's hash, find the prefix in the pool
# 5. TODO: approximate_search: given a sequence of tokens, find the similar prefixes in the pool


class PrefixMissType(Enum):
    COMPULSORY = enum.auto()
    CAPACITY = enum.auto()
    COHERENCE = enum.auto()


class PrefixEvictionPolicy(ABC):
    @abstractmethod
    def append(self, prefix_id: int) -> None:
        pass

    @abstractmethod
    def remove(self, prefix_id: int) -> None:
        pass

    @abstractmethod
    def hit(self, prefix_id: int) -> None:
        pass

    @abstractmethod
    def next_prefix_to_evict(self, filter: Callable[[int], bool]) -> int | None:
        pass


class EvictViaLRU(PrefixEvictionPolicy):
    def __init__(self):
        self.last_used_times: dict[int, datetime] = {}

    def append(self, prefix_id: int) -> None:
        self.last_used_times[prefix_id] = datetime.now()

    def remove(self, prefix_id: int) -> None:
        del self.last_used_times[prefix_id]

    def hit(self, prefix_id: int) -> None:
        self.last_used_times[prefix_id] = datetime.now()

    def get_candidates(self, filter: Callable[[int], bool]) -> list[int]:
        return [prefix_id for prefix_id in self.last_used_times if filter(prefix_id)]

    def next_prefix_to_evict(self, filter: Callable[[int], bool]) -> int | None:
        candidates = self.get_candidates(filter)
        if len(candidates) == 0:
            return None
        return min(candidates, key=self.last_used_times.get)


class EvictViaMRU(EvictViaLRU):
    def next_prefix_to_evict(self, filter: Callable[[int], bool]) -> int | None:
        candidates = self.get_candidates(filter)
        if len(candidates) == 0:
            return None
        return max(candidates, key=self.last_used_times.get)


class EvictViaRandom(PrefixEvictionPolicy):
    def __init__(self):
        self.prefix_ids: set[int] = set()

    def append(self, prefix_id: int) -> None:
        self.prefix_ids.add(prefix_id)

    def remove(self, prefix_id: int) -> None:
        self.prefix_ids.remove(prefix_id)

    def hit(self, prefix_id: int) -> None:
        pass

    def next_prefix_to_evict(self, filter: Callable[[int], bool]) -> int | None:
        candidates = [prefix_id for prefix_id in self.prefix_ids if filter(prefix_id)]
        if len(candidates) == 0:
            return None
        return random.choice(candidates)


class EvictViaFIFO(PrefixEvictionPolicy):
    def __init__(self):
        self.prefix_ids: deque[int] = deque()

    def append(self, prefix_id: int) -> None:
        self.prefix_ids.append(prefix_id)

    def remove(self, prefix_id: int) -> None:
        self.prefix_ids.remove(prefix_id)

    def hit(self, prefix_id: int) -> None:
        self.prefix_ids.remove(prefix_id)
        self.prefix_ids.append(prefix_id)

    def next_prefix_to_evict(self, filter: Callable[[int], bool]) -> int | None:
        for prefix_id in self.prefix_ids:
            if filter(prefix_id):
                return prefix_id
        return None


class PrefixPool:
    def __init__(self, block_size: int, max_gpu_prefixes: int, max_cpu_prefixes: int, max_disk_prefixes: int):
        self.prefixes: list[Prefix] = []
        self.prefixes_hash: dict[int, Prefix] = {}
        self.block_size = block_size

        # Cache Parameters
        # NOTE(njha -> kevwang): Configure parameters for benchmarking here.
        self.max_prefixes: dict[PrefixLocation, int] = {
            PrefixLocation.GPU: max_gpu_prefixes,
            PrefixLocation.CPU: max_cpu_prefixes,
            PrefixLocation.DISK: max_disk_prefixes,
        }
        
        print(f"max_prefixes: {self.max_prefixes}")

        # NOTE(njha -> kevwang): You can change the eviction policy here.
        self.eviction_policies: dict[PrefixLocation, PrefixEvictionPolicy] = {
            PrefixLocation.GPU: EvictViaLRU(),
            PrefixLocation.CPU: EvictViaLRU(),
            PrefixLocation.DISK: EvictViaLRU(),
        }

        # Statistics for benchmarking
        self.num_hits = 0
        self.num_misses: dict[PrefixMissType, int] = defaultdict(int)

    def hit_prefix(self, prefix: Prefix):
        if prefix.location == PrefixLocation.NONE:
            return
        self.eviction_policies[prefix.location].hit(prefix.prefix_id)

    def miss_prefix(self, miss_type: PrefixMissType):
        self.num_misses[miss_type] += 1

    def get_num_on(self, location: PrefixLocation) -> int:
        return len([prefix for prefix in self.prefixes if prefix.location == location])

    def get_on(self, location: PrefixLocation) -> list[Prefix]:
        return [prefix for prefix in self.prefixes if prefix.location == location]

    def set_location(self, prefix: Prefix, new_location: PrefixLocation) -> None:
        if prefix.location != PrefixLocation.NONE:
            self.eviction_policies[prefix.location].remove(prefix.prefix_id)
        prefix.set_location(new_location)
        if new_location == PrefixLocation.NONE:
            return
        self.eviction_policies[new_location].append(prefix.prefix_id)
        self.hit_prefix(prefix)

    def add_prefix(self, token_ids: List[int]) -> Prefix:
        # generate prefix_id
        prefix_id = len(self.prefixes)
        # create a new prefix
        prefix = Prefix(prefix_id, token_ids, self.block_size)
        self.prefixes.append(prefix)
        # @TODO: compute the hash of the prefix
        prefix_hash = hash(tuple(prefix.token_ids))
        # self.prefixes_hash[prefix.prefix_id] = prefix_hash
        self.prefixes_hash[prefix_hash] = prefix.prefix_id
        return prefix
    
    def next_prefix_to_evict(self, location: PrefixLocation, filter: Callable[[int], bool]) -> Prefix | None:
        prefix_id = self.eviction_policies[location].next_prefix_to_evict(filter)
        return self.prefixes[prefix_id] if prefix_id is not None else None

    # @TODO: this one should also come with a method to identify the prefix
    def efficient_search(self, token_ids: List[int]):
        # improve this search
        for prefix in self.prefixes:
            if prefix.match(token_ids):
                return prefix
        return None

    # use this first, if we already know from the application which part of the tokens are prefix.
    def fixed_search(self, prefix_hash):
        if prefix_hash not in self.prefixes_hash:
            return None
        # print("Found prefix in the pool.")
        prefix_id = self.prefixes_hash[prefix_hash]
        return self.prefixes[prefix_id]
