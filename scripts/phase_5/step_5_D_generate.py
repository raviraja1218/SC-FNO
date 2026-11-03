"""Step 5.D.A — Generate 100 unseen scenarios (50 in-distribution, 50 extrapolation)."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import torch
from pathlib import Path
from scipy.integrate import solve_ivp

from physics.kinetics import KineticsParameters, equilibrium_precursors
from physics.thermal_hydraulics import THParameters
from training.train_sc_fno import get_kin, get_th

OUT = Path("data/raw/phase_5_unseen")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("results/phase_5/step_5_D/unseen_generation")
MANIFEST.mkdir(parents=True, exist_ok=True)

kin = get_kin(); th = get_th()

# Local numpy arrays (kin is frozen dataclass — do not mutate)
import numpy as _np
_beta_i = _np.asarray(kin.beta_i, dtype=_np.float64)
_lambda_i = _np.asarray(kin.lambda_i, dtype=_np.float64)


# Initial state (from Phase 1 v4 physics)
n0 = 1.0
C0 = equilibrium_precursors(n0, kin)
T_f0 = 900.0
T_c0 = 583.0

t_grid = np.linspace(0.0, 10.0, 200)
T_END = 10.0


def coupled_rhs(t, y, rho_ext_fn):
    n, C1, C2, C3, C4, C5, C6, T_f, T_c = y
    C = np.array([C1, C2, C3, C4, C5, C6])
    rho_ext = rho_ext_fn(t)
    rho_fb = th.alpha_f * (T_f - th.T_f_ref) + th.alpha_c * (T_c - th.T_c_ref)
    rho_total = rho_ext + rho_fb
    dndt = (rho_total - kin.beta_eff) / kin.Lambda * n + np.sum(_lambda_i * C)
    dCdt = _beta_i / kin.Lambda * n - _lambda_i * C
    dTfdt = (th.f_f * th.P0 * n - th.U * th.A * (T_f - T_c)) / (th.M_f * th.c_p_f)
    dTcdt = ((1 - th.f_f) * th.P0 * n + th.U * th.A * (T_f - T_c)
             - 2 * th.m_dot * th.c_p_c * (T_c - th.T_in)) / (th.M_c * th.c_p_c)
    return [dndt, *dCdt, dTfdt, dTcdt]


def make_step(rho, t_insert=0.0):
    return lambda t: rho if t >= t_insert else 0.0

def make_ramp(rho_max, duration, t_start=0.0):
    def f(t):
        if t <= t_start: return 0.0
        if t >= t_start + duration: return rho_max
        return rho_max * (t - t_start) / duration
    return f

def make_sinusoidal(amp, omega):
    return lambda t: amp * np.sin(omega * t)

def make_grf(amp, ell, seed):
    rng = np.random.RandomState(seed)
    # Sample RBF-kernel GP on t_grid
    dists = (t_grid[:, None] - t_grid[None, :]) ** 2
    K = np.exp(-dists / (2 * ell ** 2))
    K += 1e-6 * np.eye(len(t_grid))
    L = np.linalg.cholesky(K)
    z = rng.randn(len(t_grid))
    gp = L @ z
    gp = gp / np.std(gp) * amp
    return lambda t: float(np.interp(t, t_grid, gp))


def simulate(rho_fn, rng_seed):
    y0 = np.concatenate([[n0], C0, [T_f0, T_c0]])
    try:
        sol = solve_ivp(
            lambda t, y: coupled_rhs(t, y, rho_fn),
            [0.0, T_END], y0, method="BDF",
            rtol=1e-8, atol=1e-10, dense_output=True,
            t_eval=t_grid,
        )
        if not sol.success:
            return None, None, f"solver failed: {sol.message}"
        rho_arr = np.array([rho_fn(t) for t in t_grid])
        return sol.y.T, rho_arr, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def main():
    beta_eff = kin.beta_eff
    all_scenarios = []

    # 50 in-distribution
    for fam, count in [("step",13), ("ramp",13), ("sinusoidal",12), ("grf",12)]:
        for i in range(count):
            seed = 50000 + hash(fam) % 1000 + i
            rng = np.random.RandomState(seed)
            if fam == "step":
                rho = rng.uniform(-0.5, 0.5) * beta_eff
                fn = make_step(rho)
                meta = {"family":"step", "rho":float(rho), "seed":seed}
            elif fam == "ramp":
                rho = rng.uniform(-0.3, 0.3) * beta_eff
                dur = 10**rng.uniform(np.log10(0.5), np.log10(5.0))
                fn = make_ramp(rho, dur)
                meta = {"family":"ramp","rho":float(rho),"duration":float(dur),"seed":seed}
            elif fam == "sinusoidal":
                rho = rng.uniform(0.05, 0.30) * beta_eff
                omega = 10**rng.uniform(np.log10(0.05), np.log10(20.0))
                fn = make_sinusoidal(rho, omega)
                meta = {"family":"sinusoidal","rho":float(rho),"omega":float(omega),"seed":seed}
            else:  # grf
                amp = rng.uniform(0.02, 0.25) * beta_eff
                ell = rng.uniform(0.05, 3.0)
                fn = make_grf(amp, ell, seed)
                meta = {"family":"grf","amp":float(amp),"ell":float(ell),"seed":seed}
            meta["in_distribution"] = True
            meta["id"] = f"unseen_id_{len(all_scenarios):04d}"
            all_scenarios.append((meta, fn))

    # 50 extrapolation
    for fam, count in [("step",13), ("ramp",13), ("sinusoidal",12), ("grf",12)]:
        for i in range(count):
            seed = 60000 + hash(fam) % 1000 + i
            rng = np.random.RandomState(seed)
            if fam == "step":
                rho = rng.uniform(0.6, 0.75) * beta_eff
                fn = make_step(rho)
                meta = {"family":"step","rho":float(rho),"seed":seed}
            elif fam == "ramp":
                rho = rng.uniform(0.35, 0.45) * beta_eff
                dur = rng.uniform(0.2, 7.0)
                fn = make_ramp(rho, dur)
                meta = {"family":"ramp","rho":float(rho),"duration":float(dur),"seed":seed}
            elif fam == "sinusoidal":
                rho = rng.uniform(0.35, 0.45) * beta_eff
                omega = rng.uniform(25.0, 40.0)
                fn = make_sinusoidal(rho, omega)
                meta = {"family":"sinusoidal","rho":float(rho),"omega":float(omega),"seed":seed}
            else:
                amp = rng.uniform(0.30, 0.40) * beta_eff
                ell = rng.uniform(0.01, 0.03)
                fn = make_grf(amp, ell, seed)
                meta = {"family":"grf","amp":float(amp),"ell":float(ell),"seed":seed}
            meta["in_distribution"] = False
            meta["id"] = f"unseen_ood_{len(all_scenarios):04d}"
            all_scenarios.append((meta, fn))

    manifest_rows = []
    for meta, fn in all_scenarios:
        sid = meta["id"]
        print(f"Simulating {sid} ({meta['family']}, {'in-dist' if meta['in_distribution'] else 'OOD'})...")
        traj, rho_arr, err = simulate(fn, meta["seed"])
        if err:
            print(f"  SKIP: {err}")
            continue
        d = OUT / sid
        d.mkdir(exist_ok=True)
        np.save(d / "forcing.npy", rho_arr)
        np.save(d / "trajectory.npy", traj)
        np.save(d / "time.npy", t_grid)
        meta["solver"] = "BDF"
        meta["status"] = "ok"
        json.dump(meta, open(d / "metadata.json", "w"), indent=2)
        manifest_rows.append(meta)

    # Save manifest
    import pandas as pd
    pd.DataFrame(manifest_rows).to_csv(MANIFEST / "unseen_manifest.csv", index=False)
    print(f"\nGenerated {len(manifest_rows)}/100 scenarios")
    print(f"  In-dist: {sum(1 for r in manifest_rows if r['in_distribution'])}")
    print(f"  OOD:     {sum(1 for r in manifest_rows if not r['in_distribution'])}")


if __name__ == "__main__":
    main()
