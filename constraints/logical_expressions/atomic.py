import numpy as np
from typing import Any
from abc import ABC

from .base import LogicalExpression


class AtomicExpression(LogicalExpression, ABC):
    """Base class for simple comparison expressions (leaf nodes).

    Subclasses should implement `reduce` to return a boolean np.ndarray.
    """
    def __init__(self, variable_id: str, value: Any) -> None:
        self.variable_id = variable_id
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.variable_id}', {self.value})"

class TrueExpression(LogicalExpression):
    '''Represents a logical expression that is always True.
    Useful as a default or placeholder expression when no filtering is needed.'''
    def reduce(self, domain) -> np.ndarray:
        return np.ones(domain.n_cells, dtype=bool)

class FalseExpression(LogicalExpression):
    '''Represents a logical expression that is always False.'''
    def reduce(self, domain) -> np.ndarray:
        return np.zeros(domain.n_cells, dtype=bool)


class Equal(AtomicExpression):
    '''Represents an equality comparison: domain[variable] == value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '==', self.value)


class GreaterThan(AtomicExpression):
    '''Represents a greater-than comparison: domain[variable] > value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '>', self.value)


class GreaterThanOrEqual(AtomicExpression):
    '''Represents a greater-than-or-equal comparison: domain[variable] >= value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '>=', self.value)


class LessThan(AtomicExpression):
    '''Represents a less-than comparison: domain[variable] < value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '<', self.value)


class LessThanOrEqual(AtomicExpression):
    '''Represents a less-than-or-equal comparison: domain[variable] <= value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '<=', self.value)


class NotEqual(AtomicExpression):
    '''Represents an inequality comparison: domain[variable] != value'''
    def reduce(self, domain) -> np.ndarray:
        return domain.mask_compare(self.variable_id, '!=', self.value)

# NOTE: Add more atomic constraints as needed
