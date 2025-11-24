import pandas as pd
from typing import Any
from abc import ABC

from .base import LogicalExpression


class AtomicExpression(LogicalExpression, ABC):
    """Base class for simple comparison expressions (leaf nodes).

    Subclasses should implement `reduce` to return a boolean Series.
    """
    def __init__(self, variable_id: str, value: Any) -> None:
        self.variable_id = variable_id
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.variable_id}', {self.value})"

class TrueExpression(LogicalExpression):
    '''Represents a logical expression that is always True.
    Useful as a default or placeholder expression when no filtering is needed.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=contingency_df.index, dtype=bool)

class FalseExpression(LogicalExpression):
    '''Represents a logical expression that is always False.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=contingency_df.index, dtype=bool)


class Equal(AtomicExpression):
    '''Represents an equality comparison: df[variable] == value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] == self.value


class GreaterThan(AtomicExpression):
    '''Represents a greater-than comparison: df[variable] > value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] > self.value


class GreaterThanOrEqual(AtomicExpression):
    '''Represents a greater-than-or-equal comparison: df[variable] >= value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] >= self.value


class LessThan(AtomicExpression):
    '''Represents a less-than comparison: df[variable] < value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] < self.value


class LessThanOrEqual(AtomicExpression):
    '''Represents a less-than-or-equal comparison: df[variable] <= value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] <= self.value


class NotEqual(AtomicExpression):
    '''Represents an inequality comparison: df[variable] != value'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return contingency_df[self.variable_id] != self.value
    
# NOTE: Add more atomic constraints as needed

