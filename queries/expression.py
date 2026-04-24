import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, List


class Expr(ABC):
    """Base class for boolean expressions over the contingency domain.

    All expressions must implement `evaluate(df)` returning a boolean Series
    aligned with `df.index`. Boolean operators produce new compound expressions.
    """

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask over the contingency domain rows."""
        raise NotImplementedError()

    def __and__(self, other: 'Expr') -> 'AndExpr':
        return AndExpr(self, other)

    def __or__(self, other: 'Expr') -> 'OrExpr':
        return OrExpr(self, other)

    def __invert__(self) -> 'NotExpr':
        return NotExpr(self)


class ColumnRef:
    """A column reference used to build comparison expressions.

    Produced by `col('column_name')`. Supports all standard comparison
    operators and `.isin()`. Not an Expr itself — comparisons produce Exprs.

    Example:
        col('Age') >= 18
        col('Sex') == 'M'
        col('Status').isin([1, 2])
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '==', value)

    def __ne__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '!=', value)

    def __gt__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '>', value)

    def __ge__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '>=', value)

    def __lt__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '<', value)

    def __le__(self, value: Any) -> 'CompareExpr':
        return CompareExpr(self.name, '<=', value)

    def isin(self, values: List[Any]) -> 'InExpr':
        return InExpr(self.name, values)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"col('{self.name}')"


def col(name: str) -> ColumnRef:
    """Reference a column by name to start building an expression.

    Example:
        col('Age') >= 18
        (col('Sex') == 'M') & (col('Age') < 30)
    """
    return ColumnRef(name)


class CompareExpr(Expr):
    """A single column comparison: col op value."""

    def __init__(self, column: str, op: str, value: Any) -> None:
        self.column = column
        self.op = op
        self.value = value

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        c = df[self.column]
        match self.op:
            case '==': return c == self.value
            case '!=': return c != self.value
            case '>':  return c > self.value
            case '>=': return c >= self.value
            case '<':  return c < self.value
            case '<=': return c <= self.value
            case _: raise ValueError(f"Unknown operator: {self.op}")

    def __repr__(self) -> str:
        return f"col('{self.column}') {self.op} {self.value!r}"


class AndExpr(Expr):
    """Logical AND of two expressions."""

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(df) & self.right.evaluate(df)

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


class OrExpr(Expr):
    """Logical OR of two expressions."""

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(df) | self.right.evaluate(df)

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


class NotExpr(Expr):
    """Logical NOT of an expression."""

    def __init__(self, expr: Expr) -> None:
        self.expr = expr

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return ~self.expr.evaluate(df)

    def __repr__(self) -> str:
        return f"~({self.expr!r})"


class InExpr(Expr):
    """Membership test: col.isin(values)."""

    def __init__(self, column: str, values: List[Any]) -> None:
        self.column = column
        self.values = values

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].isin(self.values)

    def __repr__(self) -> str:
        return f"col('{self.column}').isin({self.values!r})"
