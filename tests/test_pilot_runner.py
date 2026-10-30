"""Phase 1.2 Block B — pilot solver runner smoke tests.

Runs three representative scenarios (one step, one ramp, one GRF) to verify
the batch wrapper integrates correctly. Full 100-scenario run is in the
production runner script, not in the test suite.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest

from data.config_loader import load_config
from data.forcing_profiles import build_pilot_registry, generate_forcing
from data.solver import solve_one_scenario
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters


@pytest.fixture(scope="module")
def kin():
    return KineticsParameters.from_yaml(load_config('reactor_parameters'))


@pytest.fixture(scope="module")
def th():
    return THParameters.from_yaml(load_config('thermal_hydraulics'))


@pytest.fixture(scope="module")
def t_grid():
    return np.linspace(0.0, 10.0, 200)


@pytest.fixture(scope="module")
def specs(kin, t_grid):
    return build_pilot_registry(kin.beta_eff, t_grid)


@pytest.mark.parametrize("family_idx", [0, 25, 75])
def test_pilot_single_scenario(family_idx, kin, th, t_grid, specs):
    """One scenario from step, ramp, and GRF families."""
    spec = specs[family_idx]
    rho_arr = generate_forcing(spec, t_grid)

    def rho_fn(t):
        return float(np.interp(t, t_grid, rho_arr))

    result = solve_one_scenario(
        spec, t_grid, kin, th, rho_fn,
        solver_method="BDF", rtol=1e-8, atol=1e-10,
        q_t_default=0.0, include_q=True,
        config_versions={"reactor_parameters": "v3", "thermal_hydraulics": "v2"},
    )

    assert result.success, f"{spec.scenario_id}: {result.message}"
    assert result.trajectory.shape == (200, 9)
    assert result.rho_ext.shape == (200,)
    assert np.all(np.isfinite(result.trajectory))
    assert np.all(np.isfinite(result.rho_ext))
    assert result.wall_time_s > 0.0
    assert result.nfev > 0
