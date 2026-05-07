"""Differential-privacy variants for the TopDown algorithm.

Three variants are supported, all assuming sensitivity 1 (binary Q ∈ {0,1}^(n_queries × n_cells)
and within a color class queries are pairwise cell-disjoint).

  PureDP        : ε-DP via discrete Laplace.       Per-level param = ε.
  ZCDP          : ρ-zCDP via discrete Gaussian.    Per-level param = ρ.
  ApproximateDP : (ε, δ)-DP via discrete Gaussian. Per-level param = ρ + global δ.
                  Late conversion only — δ is not split across levels; it is used solely to
                  compute the equivalent (ε, δ)-DP guarantee for reporting via the Bun–Steinke
                  inequality ε = ρ_total + 2·sqrt(ρ_total · ln(1/δ)).
"""

import math
from abc import ABC, abstractmethod
from typing import List

from discretegauss import sample_dlaplace, sample_dgauss


class PrivacyMechanism(ABC):
    """Abstract base for a DP variant. Subclasses bind to a per-level parameter list at construction.

    The per-color-class budget split (level_param / num_colors) is performed inside sample_noise
    so the caller (TopDown) only needs to pass (level, num_colors). TopDown validates that
    len(level_params) matches the tree depth at construction time.
    """

    def __init__(self, level_params: List[float]) -> None:
        if any(p <= 0 for p in level_params):
            raise ValueError("All per-level privacy parameters must be > 0.")
        self.level_params: List[float] = list(level_params)

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def sample_noise(self, level: int, num_colors: int) -> int:
        """Draw one integer noise sample for a single query at this level."""

    def report_guarantee(self) -> str:
        return f"{self.name} mechanism, params={self.level_params}"


class PureDP(PrivacyMechanism):
    """ε-DP via discrete Laplace. Per-query ε_q = ε_L / num_colors; scale = 1/ε_q."""

    def sample_noise(self, level: int, num_colors: int) -> int:
        eps_q = self.level_params[level] / num_colors
        return sample_dlaplace(1.0 / eps_q)

    def report_guarantee(self) -> str:
        return f"pure epsilon-DP: total epsilon = {sum(self.level_params):.6g} (sum per-level epsilon)"


class ZCDP(PrivacyMechanism):
    """ρ-zCDP via discrete Gaussian. σ² = 1/(2·ρ_q) where ρ_q = ρ_L / num_colors."""

    def sample_noise(self, level: int, num_colors: int) -> int:
        rho_q = self.level_params[level] / num_colors
        return sample_dgauss(1.0 / (2.0 * rho_q))

    def report_guarantee(self) -> str:
        return f"rho-zCDP: total rho = {sum(self.level_params):.6g} (sum per-level rho)"


class ApproximateDP(PrivacyMechanism):
    """(ε, δ)-DP via zCDP under the hood. Late-conversion only.

    Per-level params are ρ values, exactly as in ZCDP. The global δ is NOT split across levels;
    it is used solely to report the equivalent (ε, δ)-DP guarantee via the Bun-Steinke inequality:

        ε_reported = ρ_total + 2·sqrt(ρ_total · ln(1/δ))

    where ρ_total = Σ ρ_L. This gives ~2.5× less noise than per-level (ε_L, δ_L) splitting for
    identical (ε_total, δ) — the same approach used by the 2020 US Census.

    Note: if a future utility needs the inverse direction (ε → ρ given δ), use the numerically
    stable form ρ = ε² / (c + sqrt(c²+ε))² with c = sqrt(ln(1/δ)) to avoid catastrophic cancellation.
    """

    def __init__(self, level_params: List[float], delta: float) -> None:
        super().__init__(level_params)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        self.delta: float = delta

    def sample_noise(self, level: int, num_colors: int) -> int:
        rho_q = self.level_params[level] / num_colors
        return sample_dgauss(1.0 / (2.0 * rho_q))

    def equivalent_epsilon(self) -> float:
        rho_total = sum(self.level_params)
        return rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / self.delta))

    def report_guarantee(self) -> str:
        return (
            f"(epsilon={self.equivalent_epsilon():.6g}, delta={self.delta:g})-DP "
            f"via rho-zCDP, total rho = {sum(self.level_params):.6g}"
        )
