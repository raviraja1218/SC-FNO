"""Phase 1 Step 1.2 — Forcing profile generators.

Deterministic generators for the four pilot forcing families:
    - step:      instantaneous reactivity insertion at t = 0
    - ramp:      linear reactivity ramp over a specified duration
    - sinusoidal: bounded sinusoidal oscillation
    - grf:       Gaussian random field via RBF kernel

All generators are pure functions of their parameters + seed.
No solver calls here.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable


# ---------------------------------------------------------------- generators
def step_profile(t: np.ndarray, amplitude: float, t_insert: float = 0.0) -> np.ndarray:
    """Instantaneous step: 0 before t_insert, amplitude at/after t_insert."""
    return np.where(t >= t_insert, amplitude, 0.0)


def ramp_profile(t: np.ndarray, amplitude: float, t_ramp_start: float,
                 ramp_duration: float) -> np.ndarray:
    """Linear ramp from 0 to amplitude over [t_start, t_start + duration]."""
    rho = np.zeros_like(t, dtype=float)
    if ramp_duration <= 0:
        # degenerate case: treat as step
        rho[t >= t_ramp_start] = amplitude
        return rho
    mask = (t >= t_ramp_start) & (t < t_ramp_start + ramp_duration)
    rho[mask] = amplitude * (t[mask] - t_ramp_start) / ramp_duration
    rho[t >= t_ramp_start + ramp_duration] = amplitude
    return rho


def sinusoidal_profile(t: np.ndarray, amplitude: float, omega: float,
                       phase: float = 0.0) -> np.ndarray:
    """Bounded sinusoid: amplitude * sin(omega*t + phase)."""
    return amplitude * np.sin(omega * t + phase)


def grf_profile(t: np.ndarray, amplitude_scale: float, length_scale: float,
                seed: int) -> np.ndarray:
    """Gaussian random field with RBF kernel, zero mean, bounded by clip.

    Sample n_t correlated Gaussian values via Cholesky of the RBF covariance,
    then clip to ±3*amplitude_scale to keep physically reasonable bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    # Pairwise squared distances
    dt = t[:, None] - t[None, :]
    K = np.exp(-0.5 * (dt / length_scale) ** 2)
    # Add small jitter for numerical stability
    K += 1e-10 * np.eye(n)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(n)
    f = L @ z
    f = f / (np.std(f) + 1e-30)          # normalize to unit std
    f = amplitude_scale * f
    return np.clip(f, -3.0 * amplitude_scale, 3.0 * amplitude_scale)


# ---------------------------------------------------------------- registry
@dataclass
class ScenarioSpec:
    scenario_id: str
    family: str
    params: dict
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_pilot_registry(beta_eff: float,
                         t_grid: np.ndarray) -> list[ScenarioSpec]:
    """Build exactly 100 pilot scenarios: 25 per family.

    All parameter choices are deterministic. Seeds are explicit so every
    scenario is reproducible.
    """
    specs: list[ScenarioSpec] = []
    idx = 0

    # --- Step: 25 scenarios, amplitude sweep from -0.5 to +0.5 beta_eff
    # 5 negative, 5 small positive, 5 medium positive, 5 large positive, 5 near-critical
    step_amps = np.concatenate([
        -np.linspace(0.05, 0.45, 5) * beta_eff,   # negative
        np.linspace(0.01, 0.05, 5) * beta_eff,    # small positive
        np.linspace(0.10, 0.30, 5) * beta_eff,    # medium positive
        np.linspace(0.35, 0.45, 5) * beta_eff,    # large positive
        np.linspace(0.48, 0.50, 5) * beta_eff,    # near-critical
    ])
    for k, amp in enumerate(step_amps):
        idx += 1
        specs.append(ScenarioSpec(
            scenario_id=f"pilot_{idx:04d}",
            family="step",
            params={"amplitude": float(amp), "t_insert": 0.0},
            seed=1000 + idx,
        ))

    # --- Ramp: 25 scenarios, amplitude x duration grid
    # 5 amplitudes x 5 durations
    ramp_amps = [-0.30, -0.15, 0.05, 0.15, 0.30]
    ramp_durs = [0.5, 1.0, 2.0, 3.5, 5.0]
    for amp in ramp_amps:
        for dur in ramp_durs:
            idx += 1
            specs.append(ScenarioSpec(
                scenario_id=f"pilot_{idx:04d}",
                family="ramp",
                params={
                    "amplitude": float(amp * beta_eff),
                    "t_ramp_start": 0.0,
                    "ramp_duration": float(dur),
                },
                seed=1000 + idx,
            ))

    # --- Sinusoidal: 25 scenarios, amplitude x frequency grid
    # 5 amplitudes x 5 log-spaced frequencies
    sin_amps = [0.10, 0.15, 0.20, 0.25, 0.30]
    sin_omegas = np.logspace(np.log10(0.1), np.log10(10.0), 5)
    for amp in sin_amps:
        for omega in sin_omegas:
            idx += 1
            specs.append(ScenarioSpec(
                scenario_id=f"pilot_{idx:04d}",
                family="sinusoidal",
                params={
                    "amplitude": float(amp * beta_eff),
                    "omega": float(omega),
                    "phase": 0.0,
                },
                seed=1000 + idx,
            ))

    # --- GRF: 25 scenarios, length-scale x amplitude-scale grid
    # 5 length scales x 5 amplitude scales
    grf_lengths = [0.1, 0.3, 0.5, 1.0, 2.0]
    grf_amps = [0.05, 0.10, 0.15, 0.20, 0.25]
    for ell in grf_lengths:
        for amp in grf_amps:
            idx += 1
            specs.append(ScenarioSpec(
                scenario_id=f"pilot_{idx:04d}",
                family="grf",
                params={
                    "amplitude_scale": float(amp * beta_eff),
                    "length_scale": float(ell),
                },
                seed=1000 + idx,
            ))

    assert len(specs) == 100, f"Expected 100 scenarios, got {len(specs)}"
    # Sanity: 25 per family
    from collections import Counter
    counts = Counter(s.family for s in specs)
    assert counts == {"step": 25, "ramp": 25, "sinusoidal": 25, "grf": 25}, counts
    return specs


def generate_forcing(spec: ScenarioSpec, t_grid: np.ndarray) -> np.ndarray:
    """Dispatch to the correct generator based on spec.family."""
    p = spec.params
    if spec.family == "step":
        return step_profile(t_grid, p["amplitude"], p["t_insert"])
    if spec.family == "ramp":
        return ramp_profile(t_grid, p["amplitude"],
                            p["t_ramp_start"], p["ramp_duration"])
    if spec.family == "sinusoidal":
        return sinusoidal_profile(t_grid, p["amplitude"], p["omega"], p["phase"])
    if spec.family == "grf":
        return grf_profile(t_grid, p["amplitude_scale"],
                           p["length_scale"], spec.seed)
    raise ValueError(f"Unknown family: {spec.family}")


# ---------------------------------------------------------------- production registry
def build_production_registry(beta_eff: float,
                              t_grid: np.ndarray,
                              n_total: int = 2000,
                              per_family: int = 500) -> list[ScenarioSpec]:
    """Build 2,000 production scenarios: 500 per family (Step 1.3 Block A).

    Parameter ranges per approved Step 1.3 plan:
      - Step:       amplitude uniform in [-0.5, +0.5] * beta_eff, t_insert = 0
      - Ramp:       amplitude uniform in [-0.3, +0.3] * beta_eff,
                    duration log-uniform in [0.5, 5.0] s, t_start = 0
      - Sinusoidal: amplitude uniform in [0.05, 0.30] * beta_eff,
                    omega log-uniform in [0.05, 20] rad/s, phase uniform [0, 2pi)
      - GRF:        length_scale log-uniform in [0.05, 3.0] s,
                    amplitude_scale uniform in [0.02, 0.25] * beta_eff

    Seeds:
      - All scenarios: seed = 2000 + idx (1..n_total)
      - GRF forcing additionally uses seed = 3000 + idx
    """
    assert per_family * 4 == n_total, f"{per_family}*4={per_family*4} != {n_total}"

    specs: list[ScenarioSpec] = []
    idx = 0

    # Deterministic sampling rng for parameters (separate from GRF sampling)
    rng = np.random.default_rng(seed=2024)

    # --- Step: 500
    for k in range(per_family):
        idx += 1
        amp = rng.uniform(-0.5, 0.5) * beta_eff
        specs.append(ScenarioSpec(
            scenario_id=f"scenario_{idx:04d}",
            family="step",
            params={"amplitude": float(amp), "t_insert": 0.0},
            seed=2000 + idx,
        ))

    # --- Ramp: 500
    for k in range(per_family):
        idx += 1
        amp = rng.uniform(-0.3, 0.3) * beta_eff
        # log-uniform in [0.5, 5.0]: exp(uniform(ln 0.5, ln 5))
        dur = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        specs.append(ScenarioSpec(
            scenario_id=f"scenario_{idx:04d}",
            family="ramp",
            params={
                "amplitude": float(amp),
                "t_ramp_start": 0.0,
                "ramp_duration": dur,
            },
            seed=2000 + idx,
        ))

    # --- Sinusoidal: 500
    for k in range(per_family):
        idx += 1
        amp = rng.uniform(0.05, 0.30) * beta_eff
        omega = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        specs.append(ScenarioSpec(
            scenario_id=f"scenario_{idx:04d}",
            family="sinusoidal",
            params={
                "amplitude": float(amp),
                "omega": omega,
                "phase": phase,
            },
            seed=2000 + idx,
        ))

    # --- GRF: 500
    for k in range(per_family):
        idx += 1
        ell = float(np.exp(rng.uniform(np.log(0.05), np.log(3.0))))
        amp_scale = rng.uniform(0.02, 0.25) * beta_eff
        specs.append(ScenarioSpec(
            scenario_id=f"scenario_{idx:04d}",
            family="grf",
            params={
                "amplitude_scale": float(amp_scale),
                "length_scale": ell,
            },
            seed=3000 + idx,  # GRF uses 3000-range seed for RNG
        ))

    assert len(specs) == n_total, f"Got {len(specs)} scenarios, expected {n_total}"

    # Sanity checks
    from collections import Counter
    counts = Counter(s.family for s in specs)
    assert counts == {"step": per_family, "ramp": per_family,
                      "sinusoidal": per_family, "grf": per_family}, counts

    # Unique IDs and unique seeds
    ids = [s.scenario_id for s in specs]
    seeds = [s.seed for s in specs]
    assert len(set(ids)) == n_total, "duplicate scenario IDs"
    assert len(set(seeds)) == n_total, "duplicate seeds"

    return specs
