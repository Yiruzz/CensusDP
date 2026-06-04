"""Mixed-radix encoding
---------------------
The columns are the digits of a mixed-radix number. Each column c has a radix/base
equal to the number of values in its domain, and a place value or stride equal
to the product of the sizes of all columns to its right. In other words, we have an
ordering of the cells that matches the lexicographic ordering of the columns.

For example, if we have three columns [c0, c1, c2] with domains with sizes [size0, size1, size2],
then the cell index of a combination of values is given by:

    flat_index = rank(c0) * (size1*size2) + rank(c1) * size2 + rank(c2) * 1

where rank(c) is the position of a value in c's sorted domain.

We can think this as a similar encoding of base 10. For example, if we have the number 123, then
the most significant number is 1 and contributes 1*10*10, the middle number is 2, and contributes 2*10, 
and the least significant number is 3 and contributes 3*1. 

The same applies to our mixed-radix encoding, but instead of base 10, we have different bases for each column 
according to the number of values in their domains.
"""

import warnings
import numpy as np

from typing import Dict, List, Mapping, Optional, Sequence, Any


class ContingencyDomain:
    """The set of contingency cells as a mixed-radix product space.

    Attributes:
        columns (List[str]): Query columns, in significance order (first = most
            significant digit, last = fastest varying).
        domains (Dict[str, np.ndarray]): Sorted array of all possible values per
            column. The cell index space is the Cartesian product of these.
        sizes (np.ndarray): Number of values per column (the radix of each digit).
        strides (np.ndarray): Place value of each column = product of sizes to its
            right (last stride == 1).
        n_cells (int): Total number of cells = product of sizes.
    """

    def __init__(self, columns: Sequence[str], domains: Mapping[str, np.ndarray]) -> None:
        self.columns: List[str] = list(columns)
        # Store each domain as a sorted, de-duplicated array so ranks/positions
        # are well defined and reproduce the lexicographic ordering.
        self.domains: Dict[str, np.ndarray] = {c: np.asarray(domains[c]) for c in self.columns}

        self.sizes: np.ndarray = np.array([len(self.domains[c]) for c in self.columns], dtype=np.int64)

        # strides[i] = product of sizes[i+1:], computed right-to-left; last == 1.
        self.strides: np.ndarray = np.ones(len(self.columns), dtype=np.int64)
        for i in range(len(self.columns) - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.sizes[i + 1]
 
        # Length of the contingency table (product of sizes).)
        self.n_cells: int = int(np.prod(self.sizes))


    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, columns: Sequence[str], data, declared: Optional[Mapping[str, Sequence]] = None) -> 'ContingencyDomain':
        """Build a domain, preferring user-declared value sets over inferred ones.

        For each column, the value set is taken from declared when provided;
        otherwise it is inferred from the observed data (np.sort(unique)) and a
        warning is emitted, since an inferred domain is data-dependent (not DP-safe)
        and may omit valid-but-absent values.

        Args:
            columns: Query columns, in significance order.
            data: A pandas DataFrame holding at least columns (used only for the
                inferred fallback).
            declared: Optional mapping {column: possible_values}. May be None
                or omit columns; missing columns fall back to inference.

        Returns:
            ContingencyDomain: The constructed domain.
        """
        declared = declared or {}
        domains: Dict[str, np.ndarray] = {}
        for c in columns:
            if c in declared and declared[c] is not None:
                # np.unique sorts and de-duplicates - required for the rank/index space.
                domains[c] = np.unique(np.asarray(declared[c]))
            else:
                warnings.warn(
                    f"No domain declared for column '{c}'; inferring it from the data "
                    f"(np.sort(unique)). This is data-dependent (not DP-safe) and may "
                    f"omit valid-but-absent values. Pass domain={{'{c}': [...]}} to fix.",
                    stacklevel=2,
                )
                domains[c] = np.sort(data[c].unique())
        return cls(columns, domains)

    # ------------------------------------------------------------------
    # Per-axis views
    # ------------------------------------------------------------------

    def axis_ranks(self, column: str) -> np.ndarray:
        """Return the per-cell rank (mixed-domain digit) on the given column.

        Length n_cells; values in 0..sizes[i]-1. Useful for grouping cells
        into marginals without materializing values.

        Args:
            column: One of self.columns.

        Returns:
            np.ndarray: Length-n_cells integer array of digits.
        """
        i = self.columns.index(column)
        stride = int(self.strides[i])
        size = int(self.sizes[i])
        # // stride for the incremental reapeating value pattern (0 0 1 1 2 2 3 3 ...)
        # % size to wrap around the domain size (0 0 1 1 0 0 1 1 ...) 
        return (np.arange(self.n_cells, dtype=np.int64) // stride) % size

    def mask_compare(self, column: str, op: str, value: Any) -> np.ndarray:
        """Boolean mask (length n_cells) where column op value holds.

        Uses axis_ranks + searchsorted on the small domain array - always int64,
        no object arrays regardless of the column's value type.

        Args:
            column: One of self.columns.
            op: One of '==', '!=', '>', '>=', '<', '<='.
            value: The value to compare against (must be comparable to domain values).

        Returns:
            np.ndarray: Boolean array of length n_cells.
        """
        dom = self.domains[column]
        ranks = self.axis_ranks(column)
        match op:
            case '==':
                pos = np.searchsorted(dom, value)
                if pos >= len(dom) or dom[pos] != value:
                    return np.zeros(self.n_cells, dtype=bool)
                return ranks == pos
            case '!=':
                pos = np.searchsorted(dom, value)
                if pos >= len(dom) or dom[pos] != value:
                    return np.ones(self.n_cells, dtype=bool)
                return ranks != pos
            case '>':
                return ranks >= np.searchsorted(dom, value, side='right')
            case '>=':
                return ranks >= np.searchsorted(dom, value, side='left')
            case '<':
                return ranks < np.searchsorted(dom, value, side='left')
            case '<=':
                return ranks < np.searchsorted(dom, value, side='right')
            case _:
                raise ValueError(f"Unknown operator '{op}'. Expected one of ==, !=, >, >=, <, <=.")

    def mask_isin(self, column: str, values: Sequence[Any]) -> np.ndarray:
        """Boolean mask (length n_cells) where column value is in the given set.

        Finds matching ranks in the small domain array, then checks axis_ranks.

        Args:
            column: One of self.columns.
            values: Collection of values to test membership against.

        Returns:
            np.ndarray: Boolean array of length n_cells.
        """
        dom = self.domains[column]
        matching = np.flatnonzero(np.isin(dom, values))
        if len(matching) == 0:
            return np.zeros(self.n_cells, dtype=bool)
        return np.isin(self.axis_ranks(column), matching)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, data) -> np.ndarray:
        """Map records to their flat cell indices (vectorized).

        Each column's values are mapped to ranks via np.searchsorted on the
        sorted domain, then combined with the strides. Validates membership and
        raises on any out-of-domain value (could happend if the domain isn't 
        properly defined).

        Args:
            data: A pandas DataFrame (or mapping) holding self.columns.

        Returns:
            np.ndarray: Length-len(data) array of flat cell indices.

        Raises:
            ValueError: If any observed value is not present in the declared domain.
        """
        n = len(data)
        idx = np.zeros(n, dtype=np.int64)
        for i, c in enumerate(self.columns):
            values = np.asarray(data[c])
            dom = self.domains[c]
            pos = np.searchsorted(dom, values)  

            # Defensive check for out-of-domain values
            pos_clipped = np.clip(pos, 0, len(dom) - 1)
            valid = (pos < len(dom)) & (dom[pos_clipped] == values)
            if not valid.all():
                bad = np.unique(values[~valid])
                raise ValueError(
                    f"Column '{c}' contains values outside its declared domain: "
                    f"{bad.tolist()}. Extend the domain or clean the data."
                )
            
            # mixed-radix domain, stride offset by column ordering
            idx += pos * self.strides[i]
        return idx

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Recover per-column attribute values from flat cell indices.

        Inverse mixed-radix: repeatedly divide out each column's stride. Typically
        called only on the nonzero cells of a leaf, so output stays small.

        Args:
            indices: Iterable of flat cell indices.

        Returns:
            np.ndarray: Object array of shape (len(indices), len(columns)) whose
                columns align with self.columns.
        """
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(indices), len(self.columns)), dtype=object)
        remainder = indices
        # Starting from the most significant column, divide out the stride to get the digit,
        # then mod by the size to wrap around, and look up the value in the domain array.

        # 123 // 100 = 1, remainder 23
        # 23 // 10 = 2, remainder 3
        # 3 // 1 = 3, remainder 0
        # In mixed-radix, same logic applies to recover the actual data.
        for i, c in enumerate(self.columns):
            digit = remainder // self.strides[i]
            remainder = remainder % self.strides[i]
            out[:, i] = self.domains[c][digit]
        return out

    def select(self, mask: np.ndarray) -> np.ndarray:
        """Return the flat indices where a length-n_cells boolean mask is True."""
        return np.flatnonzero(mask)
