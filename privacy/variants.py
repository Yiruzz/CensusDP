"""Differential-privacy variants for the TopDown algorithm.

Four variants are supported. All accept the L1 sensitivity Δ of the binary query matrix Q
as the third argument to add_noise(). For binary Q ∈ {0,1}^(n_queries × n_cells), the L1
sensitivity is Δ = max_j Σ_i Q[i,j] (the maximum column sum, per Li et al. PODS 2010), and
the squared L2 sensitivity coincides with Δ — so the same scalar feeds both Laplace and
Gaussian noise calibration.

  PureDP        : ε-DP via discrete Laplace.       Per-level param = ε. b = Δ / ε.
  ZCDP          : ρ-zCDP via discrete Gaussian.    Per-level param = ρ. σ² = Δ / (2ρ).
  ApproximateDP : (ε, δ)-DP via discrete Gaussian. Per-level param = ρ + global δ.
                  Late conversion only — δ is not split across levels; it is used solely to
                  compute the equivalent (ε, δ)-DP guarantee for reporting via the Bun–Steinke
                  inequality ε = ρ_total + 2·sqrt(ρ_total · ln(1/δ)).
  RenyiDP       : (ε, δ)-DP via discrete Gaussian, calibrated via JOINT-α Rényi-DP composition.
                  All levels share one Rényi order α*; the δ-to-ε conversion cost log(1/δ)/(α−1)
                  is paid once for the entire tree (not L times per level). Per-level σ_i are
                  derived so each level consumes an RDP share proportional to its input ε_i,
                  and Σ_i D_α*(σ_i; Δ) + log(1/δ)/(α*−1) ≤ Σε_i. In the operating σ ≳ 1 regime
                  this is mathematically equivalent to ApproximateDP's zCDP late conversion;
                  in the small-σ regime the discrete-Gaussian theta correction lets it do
                  strictly better. Scaffolding for future subsampled/heterogeneous mechanisms
                  where RDP genuinely beats zCDP.

Noise sampling is delegated to the vectorized OpenDP-backed mechanisms in noisy.py.
"""

import math
import zarr
import numpy as np
from abc import ABC, abstractmethod

from .noisy import sample_dgauss_optimized, sample_dlaplace_optimized

from typing import Dict, List, Optional, Tuple

class PrivacyMechanism(ABC):
    """Abstract base for a DP variant. Subclasses bind to a per-level parameter list at construction.

    add_noise(contingency_vector, level, sensitivity) draws calibrated integer noise from the
    appropriate OpenDP mechanism and adds it to the supplied vector in place.
    """

    def __init__(self, level_params: List[float]) -> None:
        if any(p <= 0 for p in level_params):
            raise ValueError("All per-level privacy parameters must be > 0.")
        self.level_params: List[float] = list(level_params)

    @property
    def name(self) -> str:
        return type(self).__name__
    
    @property
    def param_spec(self) -> str:
        class_name = self.name
        levels_str = "_".join(f"{p:.3f}" for p in self.level_params)
        return f"{class_name}_{levels_str}"

    @abstractmethod
    def add_noise(self, contingency_vector: np.ndarray, level: int, sensitivity: int) -> None:
        """Add calibrated discrete noise to contingency_vector in place."""

    def add_noise_from_precomputed(self, noisy_arr: zarr.Array, contingency_vector: np.ndarray, node_idx: int) -> None:
        """
        Add pre-computed noise to contingency_vector.

        Args:
            noisy_arr: Zarr array containing pre-computed noise vectors
            contingency_vector: Vector to add noise to (modified in place)
            node_idx: Node ID / row index in the noise Zarr array
        """
        noise_vec = np.asarray(noisy_arr[node_idx, :])
        if noise_vec is None:
            raise ValueError(
                f"Noise vector for node {node_idx} not pre-computed. "
            )
        contingency_vector += noise_vec

    def report_guarantee(self) -> str:
        return f"{self.name} mechanism, params={self.level_params}"
class PureDP(PrivacyMechanism):
    """ε-DP via discrete Laplace. b = sensitivity / ε."""

    def add_noise(self, contingency_vector: np.ndarray, level: int, sensitivity: int) -> None:
        scale = sensitivity / self.level_params[level]
        contingency_vector += sample_dlaplace_optimized(scale, contingency_vector.size)

    def report_guarantee(self) -> str:
        return f"pure epsilon-DP: total epsilon = {sum(self.level_params):.6g} (sum per-level epsilon)"


class ZCDP(PrivacyMechanism):
    """ρ-zCDP via discrete Gaussian. σ = sqrt(sensitivity / (2ρ))."""

    def add_noise(self, contingency_vector: np.ndarray, level: int, sensitivity: int) -> None:
        scale = math.sqrt(sensitivity / (2.0 * self.level_params[level]))
        contingency_vector += sample_dgauss_optimized(scale, contingency_vector.size)

    def report_guarantee(self) -> str:
        return f"rho-zCDP: total rho = {sum(self.level_params):.6g} (sum per-level rho)"


class ApproximateDP(ZCDP):
    """(ε, δ)-DP via zCDP under the hood. Late-conversion only.

    Per-level params are ρ values, exactly as in ZCDP — noise sampling is inherited.
    The global δ is NOT split across levels; it is used solely to report the equivalent
    (ε, δ)-DP guarantee via the Bun-Steinke inequality:

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

    def equivalent_epsilon(self) -> float:
        rho_total = sum(self.level_params)
        return rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / self.delta))

    def report_guarantee(self) -> str:
        return (
            f"(epsilon={self.equivalent_epsilon():.6g}, delta={self.delta:g})-DP "
            f"via rho-zCDP, total rho = {sum(self.level_params):.6g}"
        )


class RenyiDP(PrivacyMechanism):
    """(ε, δ)-DP via discrete Gaussian, calibrated by joint-α Renyi-DP composition.

    Math reference: Canonne–Kamath–Steinke 2020 (arXiv:2004.00010), Theorem 7.

    All L tree levels are calibrated together under one Renyi order α*. The per-level σ_i
    satisfy

        Σ_i D_α*^DG(σ_i; Δ) + log(1/δ)/(α*−1)  <=  Σ ε_i,

    so the δ-to-ε conversion cost log(1/δ)/(α*−1) is paid once for the whole tree (not L
    times). Each level consumes a share of the available RDP budget proportional to its
    input ε_i - preserving the user's allocation between levels - and the total is the
    user's (Σε_i, δ)-DP budget.

    Pipeline (run once per sensitivity, lazily on first add_noise call):
      1. For each α in `ALPHAS`, compute slack(α) = Σε_i − log(1/δ)/(α−1). Skip if ≤ 0.
      2. Per-level RDP target target_i(α) = (ε_i / Σε_j) · slack(α).
      3. σ_i(α) = smallest σ with D_α^DG(σ; Δ) ≤ target_i(α), by geometric bisection.
      4. Pick α* that minimises Σ σ_i² (total noise variance).
      5. Cache (σ_i, α*) for that sensitivity; add_noise just looks up σ_i and samples.

    D_α^DG is the exact discrete-Gaussian Rényi divergence
            D_α^DG(σ; Δ) = α·Δ/(2σ²) + Δ · log(S(σ², α−1) / S(σ², 0)) / (α−1)
    where S(σ², c) = Σ_k exp(−(k−c)²/(2σ²)) is evaluated via Poisson summation. The log-ratio
    is ≤ 0 and vanishes at integer α; it's the only term without a closed form.
    """

    # Standard Rényi-order grid. Dense in [1.1, 10] (Opacus / Mironov 2017) for moderate-ε
    # regimes, integers 11..255 for the tight-(ε, δ) tail where α* = 1 + log(1/δ)/Σε drifts
    # well past 64.
    ALPHAS: Tuple[float, ...] = tuple(
        [1.0 + x / 10.0 for x in range(1, 100)] + list(range(11, 256))
    )

    # Above σ² ≈ 30 the theta correction is exp(−2π²·30) ≈ 1e-257, below double precision.
    _THETA_NEGLIGIBLE_SIGMA_SQ: float = 30.0
    _BISECTION_STEPS: int = 60

    def __init__(self, level_params: List[float], delta: float,
                 alphas: Optional[List[float]] = None) -> None:
        super().__init__(level_params)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        self.delta: float = delta
        self.alphas: Tuple[float, ...] = tuple(alphas) if alphas is not None else self.ALPHAS
        # Per-sensitivity cache: Δ → (σ_per_level, α*). Filled lazily on first add_noise call.
        self._calibration: Dict[int, Tuple[Tuple[float, ...], float]] = {}

    # ------------------------------------------------------------------ public

    def add_noise(self, contingency_vector: np.ndarray, level: int, sensitivity: int) -> None:
        sigmas, _alpha = self._calibrate(sensitivity)
        contingency_vector += sample_dgauss_optimized(sigmas[level], contingency_vector.size)

    def report_guarantee(self) -> str:
        # If the mechanism has been used at least once, report the calibrated α* and σ_i.
        # Otherwise just report the (ε, δ) budget — sensitivity isn't known yet.
        eps_total = sum(self.level_params)
        head = (f"(epsilon={eps_total:.6g}, delta={self.delta:g})-DP via joint-alpha RDP on "
                f"discrete Gaussian; composed across {len(self.level_params)} levels.")
        if not self._calibration:
            return head + " [sigma values pending — calibrated lazily on first add_noise]"
        lines = [head]
        for sens, (sigmas, alpha) in self._calibration.items():
            lines.append(
                f"  sensitivity={sens}: alpha*={alpha:.2f}, sigma per level="
                + "[" + ", ".join(f"{s:.4f}" for s in sigmas) + "]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ internal logic

    def _calibrate(self, sensitivity: int) -> Tuple[Tuple[float, ...], float]:
        """Find (σ_per_level, α*) by sweeping the α grid and minimising Σ σ_i². Cached."""
        cached = self._calibration.get(sensitivity)
        if cached is not None:
            return cached

        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}.")
        total_eps = sum(self.level_params)
        log_inv_delta = math.log(1.0 / self.delta)
        weights = [eps / total_eps for eps in self.level_params]

        best_sigmas: Optional[Tuple[float, ...]] = None
        best_metric = math.inf
        best_alpha = math.nan
        for alpha in self.alphas:
            slack = total_eps - log_inv_delta / (alpha - 1.0)
            if slack <= 0.0:
                continue  # α too small — log(1/δ)/(α−1) eats the whole budget.
            sigmas = tuple(
                self._sigma_for_rdp_budget(alpha, w * slack, sensitivity) for w in weights
            )
            metric = sum(s * s for s in sigmas)
            if metric < best_metric:
                best_sigmas, best_metric, best_alpha = sigmas, metric, alpha

        if best_sigmas is None:
            raise ValueError(
                f"No feasible α in the grid for (Σε={total_eps}, δ={self.delta}). "
                f"Need α > 1 + log(1/δ)/Σε = {1.0 + log_inv_delta/total_eps:.3f}; "
                f"largest α in grid is {self.alphas[-1]}."
            )
        self._calibration[sensitivity] = (best_sigmas, best_alpha)
        return self._calibration[sensitivity]

    @classmethod
    def _sigma_for_rdp_budget(cls, alpha: float, rdp_budget: float, sensitivity: int) -> float:
        """Smallest σ with D_α^DG(σ; Δ) ≤ rdp_budget, via geometric bisection.

        The continuous-Gaussian closed form σ² = α·Δ/(2·rdp_budget) upper-bounds the answer
        (since D_α^DG ≤ α·Δ/(2σ²)), so it serves as the bisection ceiling. The floor 1e-6 is
        smaller than any σ a sane (ε, δ) would produce.
        """
        sigma_hi = math.sqrt(alpha * sensitivity / (2.0 * rdp_budget))
        sigma_lo = 1e-6
        for _ in range(cls._BISECTION_STEPS):
            sigma_mid = math.sqrt(sigma_lo * sigma_hi)
            if cls._discrete_gaussian_rdp(sigma_mid, alpha, sensitivity) <= rdp_budget:
                sigma_hi = sigma_mid
            else:
                sigma_lo = sigma_mid
        return sigma_hi

    @classmethod
    def _discrete_gaussian_rdp(cls, sigma: float, alpha: float, sensitivity: int) -> float:
        """Exact RDP at order α for the discrete Gaussian on a binary query with sensitivity Δ.

        For a binary query matrix the worst-case neighbour shift has Δ coordinates each equal
        to ±1, so the multivariate RDP is Δ copies of the one-dimensional curve.
        """
        sigma_sq = sigma * sigma
        base = alpha * sensitivity / (2.0 * sigma_sq)
        correction = cls._theta_log_ratio(sigma_sq, alpha - 1.0) / (alpha - 1.0)
        return base + sensitivity * correction

    @classmethod
    def _theta_log_ratio(cls, sigma_sq: float, c: float, n_terms: int = 20) -> float:
        """log(S(σ², c) / S(σ², 0)) for the discrete-Gaussian theta function, ≤ 0 always.

        Evaluated via Poisson summation
            S(σ², c) = √(2πσ²) · (1 + 2 Σ_{n≥1} exp(−2π²σ²n²) · cos(2πnc))
        whose terms decay as exp(−2π²σ²n²) - a single term is essentially exact for σ² ≳ 1.
        Returns 0 when c is integer (bound is tight there) or when σ² is large enough that
        the correction falls below double precision.

        See the following reference for more information:
        Canonne, Kamath, Steinke (2020). "The Discrete Gaussian for Differential
        Privacy." NeurIPS 2020. arXiv:2004.00010.
        """
        c_frac = c - round(c)
        if c_frac == 0.0 or sigma_sq > cls._THETA_NEGLIGIBLE_SIGMA_SQ:
            return 0.0
        num, den = 1.0, 1.0
        decay = 2.0 * math.pi * math.pi * sigma_sq  # 2π²σ²
        for n in range(1, n_terms + 1):
            term = math.exp(-decay * n * n)
            if term < 1e-300:
                break
            num += 2.0 * term * math.cos(2.0 * math.pi * n * c_frac)
            den += 2.0 * term
        return math.log(num / den) if num > 0.0 else 0.0

MECHANISMS = {
    "PureDP": PureDP,
    "ZCDP": ZCDP,
    "ApproximateDP": ApproximateDP,
    "RenyiDP": RenyiDP,
}