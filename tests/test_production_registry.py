"""Step 1.3 Block A — production registry unit tests."""
import sys
sys.path.insert(0, 'src')

from collections import Counter
import numpy as np
import pytest

from data.config_loader import load_config
from data.forcing_profiles import (
    build_production_registry, generate_forcing,
)
from physics.kinetics import KineticsParameters


@pytest.fixture(scope="module")
def kin():
    return KineticsParameters.from_yaml(load_config('reactor_parameters'))


@pytest.fixture(scope="module")
def t_grid():
    return np.linspace(0.0, 10.0, 200)


@pytest.fixture(scope="module")
def specs(kin, t_grid):
    return build_production_registry(kin.beta_eff, t_grid)


def test_2000_scenarios(specs):
    assert len(specs) == 2000


def test_500_per_family(specs):
    counts = Counter(s.family for s in specs)
    assert counts == {"step": 500, "ramp": 500, "sinusoidal": 500, "grf": 500}


def test_unique_ids(specs):
    ids = [s.scenario_id for s in specs]
    assert len(set(ids)) == 2000


def test_sequential_ids(specs):
    for i, s in enumerate(specs, 1):
        assert s.scenario_id == f"scenario_{i:04d}"


def test_unique_seeds(specs):
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == 2000


def test_step_amplitude_range(specs, kin):
    steps = [s for s in specs if s.family == "step"]
    amps = [s.params["amplitude"] for s in steps]
    assert all(-0.5 * kin.beta_eff - 1e-15 <= a <= 0.5 * kin.beta_eff + 1e-15
               for a in amps)
    # Ensure both negative and positive present
    assert any(a < 0 for a in amps)
    assert any(a > 0 for a in amps)


def test_ramp_duration_range(specs):
    ramps = [s for s in specs if s.family == "ramp"]
    durs = [s.params["ramp_duration"] for s in ramps]
    assert all(0.5 <= d <= 5.0 for d in durs)


def test_sinusoidal_omega_range(specs):
    sin = [s for s in specs if s.family == "sinusoidal"]
    omegas = [s.params["omega"] for s in sin]
    assert all(0.05 <= w <= 20.0 for w in omegas)


def test_grf_length_scale_range(specs):
    grfs = [s for s in specs if s.family == "grf"]
    ells = [s.params["length_scale"] for s in grfs]
    assert all(0.05 <= e <= 3.0 for e in ells)


def test_registry_reproducible(kin, t_grid):
    a = build_production_registry(kin.beta_eff, t_grid)
    b = build_production_registry(kin.beta_eff, t_grid)
    for sa, sb in zip(a, b):
        assert sa.scenario_id == sb.scenario_id
        assert sa.family == sb.family
        assert sa.seed == sb.seed
        # Params may have float precision differences at 1e-15 level
        for k in sa.params:
            if isinstance(sa.params[k], float):
                assert abs(sa.params[k] - sb.params[k]) < 1e-12


def test_forcing_finite_and_bounded(kin, t_grid, specs):
    # Sample 50 scenarios across families
    sample = [specs[i] for i in range(0, 2000, 40)]  # 50 evenly spaced
    for s in sample:
        rho = generate_forcing(s, t_grid)
        assert rho.shape == t_grid.shape, f"{s.scenario_id}: shape"
        assert np.all(np.isfinite(rho)), f"{s.scenario_id}: non-finite"
        assert np.all(np.abs(rho) < 1.0), f"{s.scenario_id}: |rho| >= 1"
