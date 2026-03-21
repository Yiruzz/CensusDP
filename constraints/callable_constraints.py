import operator
from typing import Callable, List

class AggregateConstraintFunction:
    """
    A callable constraint that encapsulates indices, target value, and a comparison operator.
    This class replaces lambda functions for serialization and multiprocessing compatibility.
    When called with a contingency vector, it evaluates the aggregated constraint 
    over the specified indices using the provided operator.
    """
    def __init__(self, indices: List[int], value: int, op: Callable = operator.eq):
        self.indices = indices
        self.value = value
        self.op = op

    def __call__(self, contingency_var):
        total = sum(contingency_var[i] for i in self.indices) 
        return self.op(total, self.value)
    
class SubarrayConstraintFunction:
    """
    A callable constraint that applies a given constraint function to a subarray
    of a joint array, defined by start and end indices.
    This class replaces lambda functions for serialization and multiprocessing compatibility.
    """
    def __init__(self, start: int, end: int, constraint: Callable):
        self.start = start
        self.end = end
        self.constraint = constraint

    def __call__(self, joint_array):
        sub_dict = {i - self.start: joint_array[i] for i in range(self.start, self.end)}
        return self.constraint(sub_dict) 