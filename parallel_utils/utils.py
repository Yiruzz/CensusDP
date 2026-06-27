import numpy as np
import zarr
from dask.distributed import WorkerPlugin

from privacy import MECHANISMS
from data_handler import DataHandler
from domain import ContingencyDomain
from optimizers.pyoptinterface import OptimizationModel
from optimizers.write_lp_directly import OptimizationModelLP

from typing import Any, Dict, List, Optional, Tuple


class WorkerInitializer(WorkerPlugin):
    '''Dask WorkerPlugin that initializes each worker process with shared state.'''

    def __init__(
        self,
        constraints: Dict[int, List],
        optimizer_conf: Tuple[type, Optional[str], Dict],
        optimizer_backend: str,
        data_type: str,
        input_file: str,
        spill_dir: str,
        microdata_dir: str,
        hierarchical_columns: List[str],
        query_columns: List[str],
        domains: Dict[str, np.ndarray],
        privacy_mech_name: str,
        level_params: List[float],
        Q: Any,
        sensitivity: int,
        zarr_path: str,
        noisy_array_name: str,
        check_correctness: bool,
    ) -> None:
        '''Store configuration that will be pushed to every worker via setup().

        Args:
            constraints (Dict[int, List]): Per-level constraint lists keyed by tree level index.
            optimizer_conf (Tuple[type, Optional[str], Dict]): Tuple (dtype, lp_problems_dir,
                solver_options) forwarded to the optimizer constructor.
            optimizer_backend (str): Either ``'pyoptinterface'`` or ``'write_lp'``.
            data_type (str): NumPy dtype string for contingency arrays (e.g. ``'int64'``).
            input_file (str): Absolute path to the Parquet input file.
            spill_dir (str): Directory where node vectors are spilled to disk.
            microdata_dir (str): Directory where leaf microdata Parquet files are written.
            hierarchical_columns (List[str]): Ordered list of geographic/hierarchy column names.
            query_columns (List[str]): Columns whose cross-product forms the contingency cell space.
            domains (Dict[str, np.ndarray]): Per-column sorted value arrays that define the contingency domain.
            privacy_mech_name (str): Key into ``MECHANISMS`` registry (e.g. ``'ZCDP'``).
            level_params (List[float]): Per-level privacy parameters passed to the mechanism constructor.
            Q (Any): Query matrix (sparse CSR or dense ndarray) applied to each node's histogram.
            sensitivity (int): L1/L2 sensitivity of Q (max column sum for binary Q).
            zarr_path (str): Path to the Zarr group holding pre-computed noise vectors.
            noisy_array_name (str): Array name within the Zarr group.
            check_correctness (bool): When True each worker checks parent/child consistency after solving.
        '''
        # Optimizer
        self.optimizer: Tuple[type, Optional[str], Dict] = optimizer_conf
        self.optimizer_backend: str = optimizer_backend
        self.constraints: Dict[int, List] = constraints
        self.data_type: str = data_type

        # Data Handler
        self.parquet_path: str = input_file
        self.spill_dir: str = spill_dir
        self.microdata_dir: str = microdata_dir
        self.hierarchical_columns: List[str] = hierarchical_columns
        self.query_columns: List[str] = query_columns
        self.domains: Dict[str, np.ndarray] = domains

        # Privacy mechanism
        self.privacy_mech_name: str = privacy_mech_name
        self.level_params: List[float] = level_params
        self.Q: Any = Q
        self.sensitivity: int = sensitivity

        # Noise
        self.zarr_path: str = zarr_path
        self.noisy_array_name: str = noisy_array_name

        self.check_correctness: bool = check_correctness

    def setup(self, worker: Any) -> None:
        '''Initialize worker-local state after the process has started.

        Called once per worker by Dask before any tasks are dispatched.
        Creates all objects that must live in the worker process (optimizer,
        DuckDB connection, privacy mechanism, Zarr handle).

        Args:
            worker (Any): The Dask distributed Worker instance being initialized.
        '''
        worker.optimizer = OptimizationModel(*self.optimizer) if self.optimizer_backend == 'pyoptinterface' else OptimizationModelLP(*self.optimizer)
        worker.constraints = self.constraints
        worker.dtype = self.data_type

        worker.data_handler = DataHandler()
        worker.data_handler.spill_dir = self.spill_dir
        worker.data_handler.microdata_dir = self.microdata_dir
        worker.data_handler.hierarchical_columns = self.hierarchical_columns
        worker.data_handler.query_columns = self.query_columns
        worker.data_handler.file_path = self.parquet_path

        worker.data_handler.contingency_domain = ContingencyDomain(columns=self.query_columns,
                                                                   domains=self.domains)
        worker.data_handler.n_cells = worker.data_handler.contingency_domain.n_cells
        worker.data_handler.create_data_view()

        worker.privacy_mechanism = MECHANISMS[self.privacy_mech_name](self.level_params)
        worker.Q = self.Q
        worker.q_sensitivity = self.sensitivity

        worker.noisy_arr = zarr.open_group(self.zarr_path, mode="r+")[self.noisy_array_name]
        worker.check_correctness = self.check_correctness
