from typing import List

from vllm.sequence import SequenceGroup


class SequencePolicy:

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


class FCFS(SequencePolicy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


class SequencePolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> SequencePolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
