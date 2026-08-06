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

import math

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
            right (last stride == 1). Lazily computed on first access.
        n_cells (int): Total number of cells = product of sizes. Lazily computed on
            first access.
    """

    def __init__(self, columns: Sequence[str], domains: Mapping[str, np.ndarray]) -> None:
        self.columns: List[str] = list(columns)
        # Store each domain as a sorted, de-duplicated array so ranks/positions
        # are well defined and reproduce the lexicographic ordering.
        self.domains: Dict[str, np.ndarray] = {c: np.asarray(domains[c]) for c in self.columns}

        self.sizes: np.ndarray = np.array([len(self.domains[c]) for c in self.columns], dtype=np.int64)

        # The joint-space quantities (strides, n_cells) are derived lazily, not here. The
        # marginal pipeline works entirely on small per-bag subdomains and never asks the
        # global domain for its joint size or strides, so building a domain over the whole
        # census does NO joint-size arithmetic - which for that many columns would overflow
        # int64. Only the full-joint path touches them, and it computes them on demand.
        self._strides: Optional[np.ndarray] = None
        self._n_cells: Optional[int] = None

    @property
    def n_cells(self) -> int:
        """Total number of cells = product of sizes.

        Python-int product (arbitrary precision, never overflows), computed once on first
        access. np.prod would silently wrap to a negative number past 2**63.
        """
        if self._n_cells is None:
            self._n_cells = math.prod(int(s) for s in self.sizes)
        return self._n_cells

    @property
    def strides(self) -> np.ndarray:
        """Mixed-radix place values: strides[i] = product of sizes[i+1:], last == 1.

        int64 array (indexed per-column in the hot encode/decode paths), computed once on
        first access.
        """
        if self._strides is None:
            strides = np.ones(len(self.columns), dtype=np.int64)
            for i in range(len(self.columns) - 2, -1, -1):
                strides[i] = strides[i + 1] * self.sizes[i + 1]
            self._strides = strides
        return self._strides

    # ------------------------------------------------------------------
    # Per-axis views
    # ------------------------------------------------------------------

    def cell_ranks(self, cells: np.ndarray, column: str) -> np.ndarray:
        """Return the rank (mixed-radix digit) of ``column`` for the given cells only.

        Same arithmetic as axis_ranks, but evaluated on an arbitrary subset instead of
        the whole space, so the cost is O(len(cells)) rather than O(n_cells). Useful 
        when using sparse representation of data.

        Args:
            cells: Flat cell indices of this domain (any order, duplicates allowed).
            column: One of self.columns.

        Returns:
            np.ndarray: Length-len(cells) integer array of digits, aligned to ``cells``.
        """
        i = self.columns.index(column)
        stride = int(self.strides[i])
        size = int(self.sizes[i])
        # // stride for the incremental repeating value pattern (0 0 1 1 2 2 3 3 ...)
        # % size to wrap around the domain size (0 0 1 1 0 0 1 1 ...)
        return (np.asarray(cells, dtype=np.int64) // stride) % size

    def axis_ranks(self, column: str) -> np.ndarray:
        """Return the per-cell rank (mixed-domain digit) on the given column.

        Length n_cells; values in 0..sizes[i]-1. Useful for grouping cells
        into marginals without materializing values.

        The whole-space case of cell_ranks: axis_ranks(c) == cell_ranks(arange(n), c).

        Args:
            column: One of self.columns.

        Returns:
            np.ndarray: Length-n_cells integer array of digits.
        """
        return self.cell_ranks(np.arange(self.n_cells, dtype=np.int64), column)

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

    # ------------------------------------------------------------------
    # Marginals / sub-domains
    # ------------------------------------------------------------------

    def subdomain(self, columns: Sequence[str]) -> "ContingencyDomain":
        """Build the ContingencyDomain of a marginal over a subset of columns.

        The sub-domain reuses this domain's declared per-column value arrays, so
        cell ranks stay consistent. ``columns`` fixes the significance order of
        the sub-domain (first = most significant), and must be a subset of
        self.columns. This is how a junction-tree bag (or separator) gets its own
        small cell space without ever touching the global n_cells.

        Args:
            columns: Ordered subset of self.columns.

        Returns:
            ContingencyDomain over ``columns``.
        """
        missing = [c for c in columns if c not in self.domains]
        if missing:
            raise ValueError(f"Columns not in domain: {missing}")
        return ContingencyDomain(list(columns), {c: self.domains[c] for c in columns})

    def project_cells_to(self, cells: np.ndarray, columns: Sequence[str]) -> np.ndarray:
        """Map the given cells to their cell index in subdomain(columns).

        Subset counterpart of project_to: same mixed-radix group id, computed only where
        asked. The marginal microdata reconstruction needs the separator group of a bag's
        occupied cells.

        Args:
            cells: Flat cell indices of this domain (any order, duplicates allowed).
            columns: Ordered subset of self.columns (same order used to build the target
                sub-domain via subdomain()).

        Returns:
            np.ndarray: Length-len(cells) int array of target sub-cell indices, aligned
                to ``cells``.
        """
        missing = [c for c in columns if c not in self.domains]
        if missing:
            raise ValueError(f"Columns not in domain: {missing}")
        cells = np.asarray(cells, dtype=np.int64)
        gid = np.zeros(len(cells), dtype=np.int64)
        stride = 1
        # For each column in the target subdomain, compute its contribution to the mixed-radix index.
        # The contribution is the rank of the column's value in its domain, multiplied by the
        # stride (place value) of that column in the subdomain. The stride is the product of the sizes
        # of all columns to the right in the subdomain. We accumulate this contribution for each column.
        for c in reversed(list(columns)):
            gid += self.cell_ranks(cells, c) * stride
            stride *= len(self.domains[c])
        return gid

    def project_to(self, columns: Sequence[str]) -> np.ndarray:
        """Map every cell of this domain to its cell index in subdomain(columns).

        Uses the same mixed-radix group id as a marginal query: for each cell,
        the returned value is the flat index that cell projects to when the
        non-``columns`` attributes are summed out. Cells sharing a projected id
        form one marginal group. Length n_cells.

        The whole-space case of project_cells_to.

        Args:
            columns: Ordered subset of self.columns (same order used to build the
                target sub-domain via subdomain()).

        Returns:
            np.ndarray: Length-n_cells int array of target sub-cell indices.
        """
        return self.project_cells_to(np.arange(self.n_cells, dtype=np.int64), columns)


class RestrictedDomain(ContingencyDomain):
    """A ContingencyDomain with the structurally impossible cells removed.

    The declared edit constraints can forbid combinations of values. Keeping those cells and 
    pinning them to zero with a `sum(x[forbidden]) == 0` row means they are still measured, 
    still get DP noise, still become optimizer variables at the root and are still shipped 
    to every worker. This class removes them from the cell space instead, so the constraint 
    disappears rather than being enforced.

    A cell here is a position in ``valid_cells``, not a mixed-radix code. This subclass overrides
    encode/decode to map between the two.

    Attributes:
        valid_cells (np.ndarray): Mixed-radix codes that survive, strictly ascending.
        base_n_cells (int): Size of the product space before the restriction.
    """

    def __init__(self, base: ContingencyDomain, valid_cells: np.ndarray) -> None:
        super().__init__(base.columns, base.domains)
        # The strides are the BASE ones and stay that way: super().cell_ranks() decodes a
        # mixed-radix code with them, and the codes in valid_cells are of the base space.
        self._strides = base.strides
        self.base_n_cells: int = base.n_cells
        self.valid_cells: np.ndarray = np.asarray(valid_cells, dtype=np.int64)

    @property
    def n_cells(self) -> int:
        """Number of surviving cells."""
        return len(self.valid_cells)

    def cell_ranks(self, cells: np.ndarray, column: str) -> np.ndarray:
        """Rank of ``column`` for the given positions (not mixed-radix codes)."""
        return super().cell_ranks(self.valid_cells[np.asarray(cells, dtype=np.int64)], column)

    def encode(self, data) -> np.ndarray:
        """Map records to positions, refusing any that lands on a removed cell.

        Raises:
            ValueError: If any record falls on a cell the declared rules removed.
        """
        # Get the mixed-radix codes for the data
        codes = super().encode(data)
        # If the codes are valid, then they should have the same value as the valid_cells at the position where they would be inserted.
        positions = np.searchsorted(self.valid_cells, codes)
        # We clip in case that the code is greater than the largest valid cell, which would give an index out of bounds.
        np.clip(positions, 0, len(self.valid_cells) - 1, out=positions)
        missing = self.valid_cells[positions] != codes
        if missing.any(): # Case when a record falls on a cell that is outside the restricted domain
            offenders = super().decode(np.unique(codes[missing])[:3])
            raise ValueError(
                f"{int(missing.sum())} record group(s) over {list(self.columns)} fall on cells "
                f"the declared edit constraints removed, e.g. "
                f"{[dict(zip(self.columns, row)) for row in offenders]}. The records should "
                f"have been repaired at the source: either a rule has no SQL rendering, or "
                f"to_sql() and reduce() disagree for it."
            )
        return positions

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Recover per-column values from positions."""
        return super().decode(self.valid_cells[np.asarray(indices, dtype=np.int64)])
