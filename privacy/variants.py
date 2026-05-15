"""Differential-privacy variants for the TopDown algorithm.

Three variants are supported. All accept the L1 sensitivity Δ of the binary query matrix Q
as the second argument to sample_noise(). For binary Q ∈ {0,1}^(n_queries × n_cells), the L1
sensitivity is Δ = max_j Σ_i Q[i,j] (the maximum column sum, per Li et al. PODS 2010), and
the squared L2 sensitivity coincides with Δ — so the same scalar feeds both Laplace and
Gaussian noise calibration.

  PureDP        : ε-DP via discrete Laplace.       Per-level param = ε. scale = Δ / ε.
  ZCDP          : ρ-zCDP via discrete Gaussian.    Per-level param = ρ. σ² = Δ / (2ρ).
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

    sample_noise(level, sensitivity) returns one integer noise sample for a single query at this
    level, calibrated to the supplied L1 sensitivity of the query matrix. 
    """

    def __init__(self, level_params: List[float]) -> None:
        if any(p <= 0 for p in level_params):
            raise ValueError("All per-level privacy parameters must be > 0.")
        self.level_params: List[float] = list(level_params)

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def sample_noise(self, level: int, sensitivity: int) -> int:
        """Draw one integer noise sample for a single query."""

    def report_guarantee(self) -> str:
        return f"{self.name} mechanism, params={self.level_params}"


class PureDP(PrivacyMechanism):
    """ε-DP via discrete Laplace. scale = sensitivity / ε."""

    def sample_noise(self, level: int, sensitivity: int) -> int:
        return sample_dlaplace(sensitivity / self.level_params[level])

    def report_guarantee(self) -> str:
        return f"pure epsilon-DP: total epsilon = {sum(self.level_params):.6g} (sum per-level epsilon)"


class ZCDP(PrivacyMechanism):
    """ρ-zCDP via discrete Gaussian. σ² = sensitivity / (2ρ)."""

    def sample_noise(self, level: int, sensitivity: int) -> int:
        return sample_dgauss(sensitivity / (2.0 * self.level_params[level]))

    def report_guarantee(self) -> str:
        return f"rho-zCDP: total rho = {sum(self.level_params):.6g} (sum per-level rho)"


class ApproximateDP(PrivacyMechanism):
    """(ε, δ)-DP via zCDP under the hood. Late-conversion only.

    Per-level params are ρ values, exactly as in ZCDP. The global δ is NOT split across levels;
    it is used solely to report the equivalent (ε, δ)-DP guarantee via the Bun-Steinke inequality:

        ε_reported = ρ_total + 2·sqrt(ρ_total · ln(1/δ))

    where ρ_total = Σ ρ_L. This gives ~2.5× less noise than per-level (ε_L, δ_L) splitting for
    identical (ε_total, δ) - the same approach used by the 2020 US Census.

    Note: if a future utility needs the inverse direction (ε → ρ given δ), use the numerically
    stable form ρ = ε² / (c + sqrt(c²+ε))² with c = sqrt(ln(1/δ)) to avoid catastrophic cancellation.
    """

    def __init__(self, level_params: List[float], delta: float) -> None:
        super().__init__(level_params)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        self.delta: float = delta

    def sample_noise(self, level: int, sensitivity: int) -> int:
        return sample_dgauss(sensitivity / (2.0 * self.level_params[level]))

    def equivalent_epsilon(self) -> float:
        rho_total = sum(self.level_params)
        return rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / self.delta))

    def report_guarantee(self) -> str:
        return (
            f"(epsilon={self.equivalent_epsilon():.6g}, delta={self.delta:g})-DP "
            f"via rho-zCDP, total rho = {sum(self.level_params):.6g}"
        )
