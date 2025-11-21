from abc import ABC
from typing import List
import pandas as pd

from .base import LogicalConstraint


class CompoundConstraint(LogicalConstraint, ABC):
    """Abstract class for constraints combining other constraints."""

    def __init__(self, expressions: List[LogicalConstraint]):
        for arg in expressions:
            if not isinstance(arg, LogicalConstraint):
                raise TypeError(
                    f"All sub-expressions must be LogicalExpression objects. Got: {type(arg)} in {self.__class__.__name__}."
                )
        self.expressions = expressions

    def __repr__(self) -> str:
        exprs_str = ", ".join(repr(e) for e in self.expressions)
        return f"{self.__class__.__name__}({exprs_str})"


class NaryExpression(CompoundConstraint, ABC):
    '''Class for n-ary logical expressions. (Or, And)
    
    Constructor recieves a variable number of LogicalConstraint arguments.
    '''
    def __init__(self, *args: LogicalConstraint):
        if not args:
            raise ValueError(f"{self.__class__.__name__} requires at least one argument.")
        super().__init__(list(args))


class UnaryExpression(CompoundConstraint, ABC):
    '''Class for unary logical expressions. (Not)'''
    def __init__(self, expression: LogicalConstraint):
        super().__init__([expression])


class BinaryExpression(CompoundConstraint, ABC):
    '''Class for binary logical expressions. (Implies, Equivalent)'''
    def __init__(self, antecedent: LogicalConstraint, consequent: LogicalConstraint):
        super().__init__([antecedent, consequent])


class And(NaryExpression):
    '''Class for logical AND expression.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        reduced_expressions = [expr.reduce(contingency_df) for expr in self.expressions]
        result = reduced_expressions[0]
        for series in reduced_expressions[1:]:
            result = result & series
        return result


class Or(NaryExpression):
    '''Class for logical OR expression.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        reduced_expressions = [expr.reduce(contingency_df) for expr in self.expressions]
        result = reduced_expressions[0]
        for series in reduced_expressions[1:]:
            result = result | series
        return result


class Not(UnaryExpression):
    '''Class for logical NOT expression.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        return ~self.expressions[0].reduce(contingency_df)


class Implies(BinaryExpression):
    '''Class for logical IMPLIES expression.'''
    def reduce(self, contingency_df: pd.DataFrame) -> pd.Series:
        antecedent = self.expressions[0].reduce(contingency_df)
        consequent = self.expressions[1].reduce(contingency_df)
        return (~antecedent) | consequent
