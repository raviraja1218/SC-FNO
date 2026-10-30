"""Phase 1 Step 1.2 Block A — Forcing generator unit tests."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest

from data.config_loader import load_config
from data.forcing_profiles import (
    step_profile, ramp_profile, sinusoidal_profile, grf_profile,
    build_pilot_registry, generate_forcing, ScenarioSpec,
)
from physics.kinetics import KineticsParameters


@pytest.fixture(scope="module")
def kin():
    return KineticsParameters.from_yaml(load_config('reactor_parameters'))


@pytest.fixture(scope="module")
def t_grid():
    return np.linspace(0.0, 10.0, 200)


def test_step_profile_before_after():
    t = np.linspace(0.0, 10.0, 101)
    rho = step_profile(t, amplitude=0.001, t_insert=2.0)
    assert np.allclose(rho[t < 2.0], 0.0)
    assert np.allclose(rho[t >= 2.0], 0.001)


def test_ramp_profile_endpoints():
    t = np.linspace(0.0, 10.0, 101)
    rho = ramp_profile(t, amplitude=0.001, t_ramp_start=0.0, ramp_duration=2.0)
    # at t=0 -> 0; at t=2 -> 0.001; at t=10 -> 0.001 (saturated)
    assert abs(rho[0]) < 1e-15
    i2 = np.argmin(np.abs(t - 2.0))
    assert abs(rho[i2] - 0.001) < 1e-12
    assert abs(rho[-1] - 0.001) < 1e-15


def test_sinusoidal_bounded():
    t = np.linspace(0.0, 10.0, 2000)
    rho = sinusoidal_profile(t, amplitude=0.002, omega=3.0)
    assert np.max(np.abs(rho)) <= 0.002 + 1e-12


def test_grf_reproducible():
    t = np.linspace(0.0, 10.0, 200)
    a = grf_profile(t, amplitude_scale=0.001, length_scale=0.5, seed=42)
    b = grf_profile(t, amplitude_scale=0.001, length_scale=0.5, seed=42)
    assert np.allclose(a, b), "GRF not reproducible with same seed"


def test_grf_different_seeds_differ():
    t = np.linspace(0.0, 10.0, 200)
    a = grf_profile(t, amplitude_scale=0.001, length_scale=0.5, seed=42)
    b = grf_profile(t, amplitude_scale=0.001, length_scale=0.5, seed=43)
    assert not np.allclose(a, b), "GRF identical for different seeds"


def test_grf_bounded():
    t = np.linspace(0.0, 10.0, 200)
    rho = grf_profile(t, amplitude_scale=0.001, length_scale=0.5, seed=42)
    assert np.max(np.abs(rho)) <= 3.0 * 0.001 + 1e-12


def test_registry_has_100_scenarios(kin, t_grid):
    specs = build_pilot_registry(kin.beta_eff, t_grid)
    assert len(specs) == 100


def test_registry_family_distribution(kin, t_grid):
    specs = build_pilot_registry(kin.beta_eff, t_grid)
    from collections import Counter
    counts = Counter(s.family for s in specs)
    assert counts == {"step": 25, "ramp": 25, "sinusoidal": 25, "grf": 25}


def test_registry_unique_ids(kin, t_grid):
    specs = build_pilot_registry(kin.beta_eff, t_grid)
    ids = [s.scenario_id for s in specs]
    assert len(set(ids)) == 100


def test_registry_creates_finite_forcing(kin, t_grid):
    specs = build_pilot_registry(kin.beta_eff, t_grid)
    for spec in specs:
        rho = generate_forcing(spec, t_grid)
        assert rho.shape == t_grid.shape, f"{spec.scenario_id}: shape mismatch"
        assert np.all(np.isfinite(rho)), f"{spec.scenario_id}: non-finite values"
        assert np.all(np.abs(rho) < 1.0), f"{spec.scenario_id}: |rho| >= 1 (unphysical)"


def test_registry_reproducibility(kin, t_grid):
    specs_a = build_pilot_registry(kin.beta_eff, t_grid)
    specs_b = build_pilot_registry(kin.beta_eff, t_grid)
    for a, b in zip(specs_a, specs_b):
        assert a.scenario_id == b.scenario_id
        assert a.seed == b.seed
        rho_a = generate_forcing(a, t_grid)
        rho_b = generate_forcing(b, t_grid)
        assert np.allclose(rho_a, rho_b), f"{a.scenario_id}: not reproducible"
