from typing import List
from vllm.prefix import PrefixLocation

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time

class PREFIX_PRIORITY(Policy):
    
        def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
        ) -> float:
            if seq_group.prefix is None:
                return 2
            match seq_group.prefix.location:
                case PrefixLocation.GPU:
                    return 3
                case PrefixLocation.CPU:
                    return 2
                case PrefixLocation.DISK:
                    return 1
                case PrefixLocation.NONE:
                    return 0

class PREFIX_PRIORITY_EQ(Policy):
    
        def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
        ) -> float:
            if seq_group.prefix is None:
                return 0

            if seq_group.prefix.is_on_gpu():
                return 2
            else:
                # We have the prefix but it's elsewhere, so we want as much time
                # as possible to swap it in.
                return 0

class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'prefix_priority': PREFIX_PRIORITY,
        'prefix_priority_eq': PREFIX_PRIORITY_EQ,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
