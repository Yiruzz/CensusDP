import operator
from typing import Callable, List


class SerializableAggregateFunction:
    '''Serializable callable that aggregates values from selected indices
    and applies a comparison operator.
    '''
    def __init__(self, indices: List[int], value: int, op: Callable = operator.eq):
        self.indices = indices
        self.value = value
        self.op = op

    def __call__(self, contingency_var):
        total = sum(contingency_var[i] for i in self.indices)
        return self.op(total, self.value)

class SerializableSubarrayFunction:
    '''Serializable callable that extracts a subarray and applies a function
    to the re-indexed subset.
    '''
    def __init__(self, start: int, end: int, func: Callable):
        self.start = start
        self.end = end
        self.func = func

    def __call__(self, joint_array):
        sub_dict = {i - self.start: joint_array[i] for i in range(self.start, self.end)}
        return self.func(sub_dict)