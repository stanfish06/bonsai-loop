"""Type stub for the compiled _core module; keep in step with _core.pyx."""

from collections.abc import Iterable
from os import PathLike
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

class PairDistances(NamedTuple):
    values: npt.NDArray[np.float64]
    node_ids: list[str]
    uncertainty: npt.NDArray[np.float64] | None

class BonsaiOutput:
    def __init__(
        self, folder: str | PathLike[str], topology_only: bool = False
    ) -> None: ...
    @property
    def folder(self) -> str: ...
    @property
    def dialect(self) -> str: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def node_ids(self) -> list[str]: ...
    @property
    def feature_ids(self) -> list[str]: ...
    @property
    def node_index(self) -> npt.NDArray[np.int32]: ...
    @property
    def is_sample(self) -> npt.NDArray[np.bool_]: ...
    @property
    def parent_index(self) -> npt.NDArray[np.int32]: ...
    @property
    def branch_length(self) -> npt.NDArray[np.float64]: ...
    def tree_distances(
        self,
        nodes: Iterable[str] | None = None,
        leaves_only: bool = False,
        threads: int = 0,
    ) -> PairDistances: ...
    def posterior_distances(
        self,
        nodes: Iterable[str] | None = None,
        leaves_only: bool = False,
        threads: int = 0,
        with_uncertainty: bool = False,
    ) -> PairDistances: ...
    def posterior_belief(
        self, node_id: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def detect_convergence(
        self,
        nodes: Iterable[str] | None = None,
        leaves_only: bool = False,
        top_k: int = 50,
        max_pairs: int = 5_000_000,
        allow_huge: bool = False,
        threads: int = 0,
    ) -> dict[str, Any]: ...
