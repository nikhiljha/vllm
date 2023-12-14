from collections import defaultdict
import enum
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

# Define the prefix class, which is a collection of prefix (a sequence of tokens).
# The class contains the following main methods:
# 1. A match method that checks if a prefix matches a given sequence of tokens.
# 2. A swapping method that can load or offload the prefix to or from GPU
# 3. An update_frequency method that updates the frequency of the prefix.
# 4. A get_status method that tells if the prefix is on GPU or not.

class PrefixLocation(Enum):
    GPU = enum.auto()
    CPU = enum.auto()
    DISK = enum.auto()
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
    
    def get_block_table_num(self) -> List[int]:
        return [block.block_number for block in self.block_table]
    
    def match(self, tokens):
        return tokens[:self.length] == self.token_ids
   
    # whether the prefix is on GPU or not
    def is_on_gpu(self):
        return self.location == PrefixLocation.GPU
    
    def get_load_in_progress(self) -> bool:
        return self.load_in_progress
    
    def set_load_in_progress(self, load_in_progress) -> None:
        self.load_in_progress = load_in_progress
    
    def set_location(self, location: PrefixLocation) -> None:
        self.location = location
    
    def get_location(self) -> PrefixLocation:
        return self.location
    
    def get_length(self):
        return self.length


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

class PrefixPool:
    def __init__(self, block_size: int):
        self.prefixes: list[Prefix] = []
        self.prefixes_hash: dict[int, Prefix] = {}
        self.block_size = block_size

        # Constants
        # TODO(njha): Calculate this in some sane way.
        self.max_prefixes: dict[PrefixLocation, int] = {
            PrefixLocation.GPU: 1,
            PrefixLocation.CPU: 100,
            PrefixLocation.DISK: 0,
            PrefixLocation.NONE: 0,
        }

        # Caching-related attributes
        # Map from prefix ID to last used time
        self.last_used_times: dict[int, datetime] = {}

        # Statistics for benchmarking
        self.num_hits = 0
        self.num_misses: dict[PrefixMissType, int] = defaultdict(int)
    
    def hit_prefix(self, prefix_id: int):
        self.last_used_times[prefix_id] = datetime.now()

    def miss_prefix(self, prefix_id: int, miss_type: PrefixMissType):
        self.num_misses[miss_type] += 1
        self.last_used_times[prefix_id] = datetime.now()
    
    def get_num_on(self, location: PrefixLocation) -> int:
        return len([prefix for prefix in self.prefixes if prefix.location == location])
    
    def get_lru_prefix(self, location: PrefixLocation) -> Prefix | None:
        if len(self.last_used_times) == 0:
            return None
        prefix_id = min((prefix.prefix_id for prefix in self.prefixes if prefix.location == location), key=self.last_used_times.get)
        return self.prefixes[prefix_id]
    
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
        self.last_used_times[prefix.prefix_id] = datetime.now()
        return prefix
        
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

