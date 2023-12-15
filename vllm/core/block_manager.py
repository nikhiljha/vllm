"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.prefix import Prefix, PrefixLocation


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_device_blocks: dict[Device, int],
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_device_blocks = num_device_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_device_blocks[Device.GPU])
        self.allocators: Dict[Device, BlockAllocator] = {
            Device.GPU: BlockAllocator(Device.GPU, block_size, num_device_blocks[Device.GPU]),
            Device.CPU: BlockAllocator(Device.CPU, block_size, num_device_blocks[Device.CPU]),
            Device.DISK: BlockAllocator(Device.DISK, block_size, num_device_blocks[Device.DISK])
        }

        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)

        if seq_group.prefix is not None and seq_group.prefix.is_on_gpu():
            num_required_blocks -= seq_group.prefix.get_length() // self.block_size 

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.allocators[Device.GPU].get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)

        block_table: BlockTable = []
        prefix_block_table: BlockTable = []
        num_prefix_blocks = 0
        if seq_group.prefix is not None:
            # prefix is already on gpu or will be swapped in before the actual computation
            if seq_group.prefix.is_on_gpu():
                num_prompt_blocks -= seq_group.prefix.get_length() // self.block_size
                for block in seq_group.prefix.block_table:
                    block.ref_count += seq_group.num_seqs()
                    block_table.append(block)
                # TODO: will need to perform the copy-on-write if prefix length is not a multiple of block size
                    
            # allocate blocks for the prefix, we need to calculate the prefix's kv in this run
            elif not seq_group.prefix.get_load_in_progress():
                num_prefix_blocks = seq_group.prefix.get_length() // self.block_size
                seq_group.prefix.set_load_in_progress(True)

        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.allocators[Device.GPU].allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)
            if logical_idx < num_prefix_blocks:
                block.ref_count += 1
                prefix_block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()
        
        if num_prefix_blocks > 0:
            seq_group.prefix.block_table = prefix_block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.allocators[Device.GPU].get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # re-use a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                block = self.allocators[Device.GPU].allocate()
                block_table.append(block)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.allocators[Device.GPU].allocate()
            block_table[-1] = new_block
            self.allocators[Device.GPU].free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.allocators[Device.GPU].get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def can_swap_in_prefix(self, prefix: Prefix) -> bool:
        blocks = prefix.block_table
        num_free_blocks = self.allocators[Device.GPU].get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks)
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        if seq_group.prefix is not None:
            # make sure to swap in the prefix first
            assert seq_group.prefix.is_on_gpu()

        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            if seq_group.prefix is not None:
                for block in seq_group.prefix.block_table:
                    new_block_table.append(block)
                    block.ref_count += 1

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.allocators[Device.GPU].allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.allocators[Device.CPU].free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.allocators[Device.CPU].get_num_free_blocks()

    PREFIX_LOCATION_TO_DEVICE = {
        PrefixLocation.GPU: Device.GPU,
        PrefixLocation.CPU: Device.CPU,
        PrefixLocation.DISK: Device.DISK
    }
    
    def swap_prefix(self, prefix: Prefix, target: PrefixLocation) -> Dict[int, int]:
        # Move the prefix to the target location
        if prefix.get_location() == target:
            raise ValueError(f"Prefix {prefix.prefix_id} is already on {target}")

        if target == PrefixLocation.NONE:
            raise ValueError(f"Cannot swap to NONE, call BlockManager.evict_prefix instead")

        target_device = self.PREFIX_LOCATION_TO_DEVICE[target]
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        new_block_table = []
        block_table = prefix.block_table
        for i, block in enumerate(block_table):
            new_block = self.allocators[target_device].allocate()
            mapping[block] = new_block
            new_block_table.append(new_block)
            # Free the old block
            assert block.ref_count == 1
            self.allocators[block.device].free(block)
        prefix.block_table = new_block_table
        return {
            block.block_number: new_block.block_number
            for block, new_block in mapping.items()
        }
    
    def can_swap_prefix_to(self, prefix: Prefix, target: PrefixLocation) -> bool:
        blocks = prefix.block_table
        target_device = self.PREFIX_LOCATION_TO_DEVICE[target]
        return len(blocks) <= self.allocators[target_device].get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                # do not swap out the prefix
                if seq_group.prefix is not None and gpu_block in seq_group.prefix.block_table:
                    self.allocators[Device.GPU].free(gpu_block)
                    continue

                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.allocators[Device.CPU].allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.allocators[Device.GPU].free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping
    
    def evict_prefix(self, prefix: Prefix) -> None:
        # CPU block -> Gone
        block_table = prefix.block_table
        for block in block_table:
            # Free the GPU block swapped out to CPU.
            assert block.ref_count == 1
            self.allocators[block.device].free(block)

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.allocators[Device.GPU].free(block)
            else:
                self.allocators[Device.CPU].free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.allocators[Device.GPU].get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.allocators[Device.CPU].get_num_free_blocks()
