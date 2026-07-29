"""Estimation phase for the factored (junction-tree) pipeline.

Mirrors parallel_utils/estimation_phase.py, but each node carries one marginal per
junction-tree bag instead of a single full-joint contingency vector. The bags are laid
out back-to-back in a per-node vector of length W = data_handler.marginal_width, so a
node group's decision variable is indexed by ``k * W + p`` for child k and position p -
structurally identical to the ``k * n_cells + j`` layout of the full-joint pipeline,
which is why the optimizer backends are reused unchanged.

Three constraint families glue the problem together:

  1. separator consistency  - within each node, overlapping bags must agree on their
                              shared columns (this module),
  2. within-bag constraints - the user's constraints, already emitted in [0, W) by
                              DataHandler.materialize_node_marginals,
  3. geographic consistency - per position, the children sum to the parent.
"""
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import zarr

from constraints.sparse_constraint import SparseConstraint
from data_handler import DataHandler
from domain import ContingencyDomain
from graph import JunctionTree
from optimizers import build_optimizer
from privacy import PrivacyMechanism


def _group_positions(gid: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket cell positions by their projected group id.

    Equivalent to ``[np.flatnonzero(gid == s) for s in range(n_groups)]`` but computed in
    one sort instead of one full scan per group - the naive form is O(n_groups * len(gid)),
    which can be expensive once a bag has many cells.

    Args:
        gid (np.ndarray): Group id per cell, as returned by DataHandler.separator_projection.
        n_groups (int): Number of groups (the separator's cell count).

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(order, bounds)`` where the positions belonging to
            group s are ``order[bounds[s]:bounds[s + 1]]``.
    """
    # The stable sort preserves the order of cells within each group, which is important
    # because the bag's cells are already ordered by their subdomain id and the separator
    # is encoded with the same mixed-radix id. So the p-th cell of bag i that projects to 
    # s is the p-th cell of bag j that projects to s.
    order = np.argsort(gid, kind="stable")

    # Count how many cells belong to each group, then compute the cumulative sum to get
    # the bounds of each group in the sorted order. The first bound is 0, and the last 
    # bound is len(gid), so bounds has length n_groups + 1.
    counts = np.bincount(gid, minlength=n_groups)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    return order, bounds


def separator_constraints(data_handler) -> List[SparseConstraint]:
    """Separator-consistency rows for ONE node, in its concatenated marginal space [0, W).

    For every junction-tree edge (i, j) and every value s of their separator, the cells of
    bag i that project to s must sum to the same total as the cells of bag j that project
    to s. Written as a single row with +1 coefficients on bag i and -1 on bag j:

        sum(x[cells of bag i -> s]) - sum(x[cells of bag j -> s]) = 0

    These rows are what make a constraint imposed on one bag propagate to every other bag
    containing its scope (by the running-intersection property), and what makes the
    microdata reconstruction of the factored pipeline exact.

    The pattern is identical for every node of the geographic tree - only the block offset
    changes - so callers compute it once and shift it per child with
    ``SparseConstraint.prune_to_active_space(k * W, active_set)``.

    Args:
        data_handler (DataHandler): Handler with a junction tree already bound via
            build_marginal_domains().

    Returns:
        List[SparseConstraint]: One row per (tree edge, separator value); empty when the
            tree has a single bag or all separators are empty.
    """
    junction_tree = data_handler.junction_tree
    assert junction_tree is not None, "No junction tree bound. Call build_marginal_domains first."

    rows: List[SparseConstraint] = []
    for i, j in junction_tree.edges():
        separator = junction_tree.separator(i, j)
        if not separator:
            # Disconnected interaction graph: the bags share no column, so there is
            # nothing to agree on. (The node total is tied by the geographic family.)
            continue

        # Number of constraints = number of separator values = number of values in the subdomain.
        n_groups = data_handler.contingency_domain.subdomain(separator).n_cells
        # Group the cells of each bag by their projected separator value, so we can sum them
        # in one row. The order of the cells within each group is preserved, which is
        # important because the separator is encoded with the same mixed-radix.
        order_i, bounds_i = _group_positions(data_handler.separator_projection(i, separator), n_groups)
        order_j, bounds_j = _group_positions(data_handler.separator_projection(j, separator), n_groups)
        offset_i = data_handler.bag_offsets[i]
        offset_j = data_handler.bag_offsets[j]

        for s in range(n_groups):
            cells_i = order_i[bounds_i[s]:bounds_i[s + 1]] + offset_i
            cells_j = order_j[bounds_j[s]:bounds_j[s + 1]] + offset_j
            rows.append(SparseConstraint(
                indices=np.concatenate([cells_i, cells_j]),
                coefs=np.concatenate([np.ones(len(cells_i)), -np.ones(len(cells_j))]),
                sense="=",
                rhs=0.0,
            ))
    return rows


def shift_to_children(rows: Sequence[SparseConstraint], n_children: int, width: int,
                      active_set: set) -> List[SparseConstraint]:
    """Replicate single-node rows into the joint child space, pruning inactive cells.

    Dropping a pruned index from a separator row is sound: pruned cells are structurally
    zero, so they contribute nothing to either side of the equality.

    Args:
        rows (Sequence[SparseConstraint]): Rows indexed in one node's space [0, width).
        n_children (int): Number of children solved jointly.
        width (int): Length of a node's concatenated marginal vector.
        active_set (set): Active joint-space indices {k * width + p}.

    Returns:
        List[SparseConstraint]: The rows shifted into joint space, fully-pruned rows dropped.
    """
    shifted: List[SparseConstraint] = []
    for k in range(n_children):
        for row in rows:
            pruned = row.prune_to_active_space(k * width, active_set)
            if pruned is not None:
                shifted.append(pruned)
    return shifted


def init_process(optimizer_params: Tuple[Any, ...], optimizer_backend: str,
                 constraints_dict: Dict[int, List], spill_dir: str, microdata_dir: str,
                 parquet_path: str, domain_dict: Dict[str, Any], hierarchical_columns: List[str],
                 query_columns: List[str], junction_tree: JunctionTree,
                 privacy_mechanism: PrivacyMechanism, query_sensitivity: int, check: bool,
                 zarr_path: str, noisy_array_name: str) -> None:
    '''Initialize the globals of a factored-pipeline worker process.

    Mirrors estimation_phase.init_process, plus the junction tree. Two things are derived
    here rather than shipped, because they are pure functions of the tree and the domain
    and would only cost pickling: the separator-constraint pattern (identical for every
    node) and the identity workload.

    Args:
        optimizer_params (Tuple): (dtype, lp_problems_dir, solver_options) for the backend.
        optimizer_backend (str): Backend name, see optimizers.build_optimizer.
        constraints_dict (Dict[int, List]): Constraints mapped by tree level.
        spill_dir (str): Directory for spilled node marginals.
        microdata_dir (str): Directory for temporary microdata files.
        parquet_path (str): Path to the input parquet file.
        domain_dict (Dict[str, Any]): Declared per-column domains for the query columns.
        hierarchical_columns (List[str]): Hierarchical column names.
        query_columns (List[str]): Query column names.
        junction_tree (JunctionTree): The bags to measure and their tree structure.
        privacy_mechanism (PrivacyMechanism): Mechanism used for the noise fallback.
        query_sensitivity (int): Squared L2 sensitivity (= number of bags).
        check (bool): Whether to verify parent/children totals per node.
        zarr_path (str): Path to the Zarr group holding pre-computed noise.
        noisy_array_name (str): Name of the noise array within that group.
    '''
    global _optimizer, _data_handler, _Q, _check, _privacy_mechanism
    global _query_sensitivity, _constraints, _noisy_arr, _separator_constraints

    _optimizer = build_optimizer(optimizer_backend, optimizer_params)

    _data_handler = DataHandler()
    _data_handler.spill_dir = spill_dir
    _data_handler.microdata_dir = microdata_dir
    _data_handler.hierarchical_columns = hierarchical_columns
    _data_handler.query_columns = query_columns
    _data_handler.file_path = parquet_path

    _data_handler.contingency_domain = ContingencyDomain(columns=query_columns, domains=domain_dict)
    _data_handler.n_cells = _data_handler.contingency_domain.n_cells
    _data_handler.build_marginal_domains(junction_tree)
    _data_handler.create_data_view()

    _separator_constraints = separator_constraints(_data_handler)

    # The optimizer derives its variable layout from query_matrix.shape, so the identity
    # over the concatenated marginal space declares "one measurement per bag cell". Every
    # row has a single nonzero, so no auxiliary q_x variables are created and the objective
    # reduces to sum_bag ||x_bag - y_bag||^2 - exactly the factored objective.
    _Q = sp.identity(_data_handler.marginal_width, format="csr", dtype=float)

    _constraints = constraints_dict
    _query_sensitivity = query_sensitivity
    _privacy_mechanism = privacy_mechanism
    _noisy_arr = zarr.open_group(zarr_path, mode="r")[noisy_array_name]
    _check = check


def combine_child_constraints(n_children: int, parent_vector: np.ndarray, support: np.ndarray,
                              children_constraints: List[List[SparseConstraint]],
                              active_set: set, width: int,
                              _separator_constraints: Sequence[SparseConstraint]) -> List[SparseConstraint]:
    '''Assemble the three constraint families for one node group, in joint space.

    Args:
        n_children (int): Number of children solved jointly.
        parent_vector (np.ndarray): The parent's concatenated marginals (length width).
        support (np.ndarray): Positions where the parent is non-zero.
        children_constraints (List[List[SparseConstraint]]): Per-child user constraints,
            already indexed in [0, width) by materialize_node_marginals.
        active_set (set): Active joint-space indices {k * width + p}.
        width (int): Length of one node's concatenated marginal vector.
        _separator_constraints (Sequence[SparseConstraint]): Single-node separator pattern.

    Returns:
        List[SparseConstraint]: All rows of the joint problem.
    '''
    joint: List[SparseConstraint] = []

    # (2) Within-bag user constraints: shift each child's rows into its block.
    for child_index, child_constraints in enumerate(children_constraints):
        for constraint in child_constraints:
            pruned = constraint.prune_to_active_space(child_index * width, active_set)
            if pruned is not None:
                joint.append(pruned)

    # (1) Separator consistency, replicated per child.
    joint.extend(shift_to_children(_separator_constraints, n_children, width, active_set))

    # (3) Geographic consistency: the children sum to the parent, position by position.
    # Only over the parent's support - where the parent is 0 no child variable exists, so
    # the sum is structurally 0 and the row would be redundant.
    coefs = np.ones(n_children)
    for position in support:
        position = int(position)
        joint.append(SparseConstraint(
            indices=np.array([k * width + position for k in range(n_children)]),
            coefs=coefs,
            sense="=",
            rhs=float(parent_vector[position]),
        ))

    return joint


def _check_node_correctness(parent_vector: np.ndarray, joint_solution: sp.csc_matrix,
                            n_children: int, width: int) -> None:
    '''Report when a bag's child marginals do not sum to the parent's.

    Checked per bag rather than on the grand total: a single total could match while the
    individual bags disagree.

    Args:
        parent_vector (np.ndarray): The parent's concatenated marginals.
        joint_solution (sp.csc_matrix): Solved children, shape (n_children * width, 1).
        n_children (int): Number of children.
        width (int): Length of one node's concatenated marginal vector.
    '''
    children_total = np.zeros(width, dtype=np.int64)
    dense = np.asarray(joint_solution.todense()).ravel()
    for k in range(n_children):
        children_total += dense[k * width:(k + 1) * width].astype(np.int64)

    mismatched = np.flatnonzero(children_total != parent_vector.astype(np.int64))
    if len(mismatched):
        position = int(mismatched[0])
        print(f"\nError: children sum to {children_total[position]} at position {position} "
              f"but the parent holds {parent_vector[position]} "
              f"({len(mismatched)} positions differ).")


def estimate_and_update_children(node_id: int, node_path: str,
                                 children_filter_dicts: List[Dict[str, Any]],
                                 children_ids: List[int], children_level: int,
                                 is_leaf: bool = False) -> float:
    '''Solve one node group jointly and spill the children's estimated marginals.

    Args:
        node_id (int): ID of the parent node.
        node_path (str): Path to the parent's spilled marginals.
        children_filter_dicts (List[Dict[str, Any]]): Filter dictionary per child.
        children_ids (List[int]): Node ID per child, for the pre-computed noise lookup.
        children_level (int): Tree level shared by all the children.
        is_leaf (bool): Whether the children are leaves.

    Returns:
        float: Seconds spent writing microdata (0.0 when the children are not leaves).
    '''
    parent_marginals = _data_handler.load_marginals(node_path)
    parent_vector = np.concatenate(parent_marginals)

    width = _data_handler.marginal_width
    n_children = len(children_filter_dicts)

    # Measure and noise each child's bags, concatenated into one length-width vector.
    children_measurements: List[np.ndarray] = []
    children_constraints: List[List[SparseConstraint]] = []
    for filter_dict, child_id in zip(children_filter_dicts, children_ids):
        marginals, constraints = _data_handler.materialize_node_marginals(
            filter_dict, _constraints[children_level])
        measurement = np.concatenate(marginals)

        # Fall back to sampling in situ only for the ways a pre-computed noise file can
        # legitimately come up short (row missing, array too narrow, unreadable). A broad
        # `except` here would also swallow programming errors and silently draw noise
        # calibrated by a different path.
        try:
            _privacy_mechanism.add_noise_from_precomputed(_noisy_arr, measurement, child_id)
        except (IndexError, ValueError, KeyError, OSError):
            _privacy_mechanism.add_noise(measurement, children_level, _query_sensitivity)

        children_measurements.append(measurement)
        children_constraints.append(constraints)

    # Positions where the parent is non-zero. Non-negativity plus the geographic family
    # force every child to 0 elsewhere, so variables are only created there.
    support = np.flatnonzero(parent_vector)
    active = [k * width + int(p) for k in range(n_children) for p in support]

    joint_constraints = combine_child_constraints(
        n_children, parent_vector, support, children_constraints,
        set(active), width, _separator_constraints)

    t1 = time.time()
    x_tilde = _optimizer.non_negative_real_estimation(
        noisy_measurements=children_measurements,
        node_id=node_id,
        constraints=joint_constraints,
        query_matrix=_Q,
        active=active,
    )
    real_time = time.time() - t1

    t1 = time.time()
    joint_solution = _optimizer.rounding_estimation(
        x_tilde=x_tilde,
        node_id=node_id,
        constraints=joint_constraints,
        active=active,
        n=n_children * width,
    )
    rounding_time = time.time() - t1

    if _check:
        _check_node_correctness(parent_vector, joint_solution, n_children, width)

    microdata_time = 0.0
    if is_leaf:
        t1 = time.time()
        children_marginals = [
            _data_handler.split_marginals(joint_solution[k * width:(k + 1) * width])
            for k in range(n_children)
        ]
        _data_handler.write_microdata_from_marginals(node_id, children_marginals, children_filter_dicts)
        microdata_time = time.time() - t1
    else:
        _data_handler.update_child_marginals(joint_solution, children_filter_dicts)

    print(f'  [Node {node_id}] - real {real_time:.1f}s - rounding {rounding_time:.1f}s')
    return microdata_time
