"""Step 4B.5 correction v2 — per-channel derivative + ZERO forcing.

Rationale: models were trained against zero-forcing residual (rho_fn=zeros).
Per-channel derivative is the only legitimate correction of the original protocol.
Using actual forcing would measure a residual the models never minimized,
and would require retraining to be meaningful — that is Phase 5 scope.
"""
import sys, json, time, hashlib
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/phase_4/step_4b_5_correction")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from compute_du_dt_corrected import compute_du_dt_per_channel

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_4/phase_4b/evaluation/physics_revalidation_addendum")
OUT.mkdir(parents=True, exist_ok=True)

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

MODELS = {
    "P3-PI-FNO": {"path": "models/sc_fno/phase_3/pi_fno/best_val.pt", "wrap": "scfno"},
    "Phase4A.1": {"path": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "wrap": "scfno"},
    "P4-A": {"path": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "wrap": "scfno"},
    "P4-B": {"path": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "wrap": "unified", "pad_size": 20},
    "P4-C": {"path": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "wrap": "scfno"},
    "P4-D": {"path": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "wrap": "unified", "pad_size": 20},
}

TEST_IDS = "results/phase_1/splitting/test_ids.csv"
EXPECTED = "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2"
assert hashlib.sha256(open(TEST_IDS, "rb").read()).hexdigest() == EXPECTED
print("[OK] Test set verified")

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
test_ids = pd.read_csv(TEST_IDS)["scenario_id"].tolist()

forcing_norm = np.zeros((200, 200), dtype=np.float32)
traj_phys = np.zeros((200, 9, 200), dtype=np.float64)
for i, sid in enumerate(test_ids):
    d = RAW / sid
    f = np.load(d / "forcing.npy"); t = np.load(d / "trajectory.npy")
    forcing_norm[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    for j, k in enumerate(STATES):
        traj_phys[i, j] = t[:, j]
print(f"[OK] Loaded 200 scenarios")

from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.decoder import build_decoder
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss
from models.boundary_aware_fno import UnifiedLatentFNO
from training.train_sc_fno import get_kin, get_th

loss_fn = load_composite_loss()
kin = get_kin(); th = get_th()
print("[OK] Loss + params loaded")

T_QUERY = torch.linspace(0.5, 9.5, 50, device=device)


def load_model(cfg):
    fno = load_phase2_fno(device=device)
    if cfg["wrap"] == "unified":
        fno = UnifiedLatentFNO(fno, pad_size=cfg.get("pad_size", 20))
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno, decoder).to(device)
    ckpt = torch.load(cfg["path"], map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def evaluate_physics_zero_forcing(model, forcing_norm_all, bsz=16):
    n = len(test_ids)
    per_scn_physics = []
    per_scn_ic = []
    per_state_physics = {k: [] for k in STATES}

    for i in range(0, n, bsz):
        j = min(i + bsz, n)
        b = j - i
        xb = torch.tensor(forcing_norm_all[i:j], dtype=torch.float32).unsqueeze(1).to(device)
        try:
            with torch.enable_grad():
                tq = T_QUERY.unsqueeze(0).expand(b, -1)
                z, u_hat, du_dt = compute_du_dt_per_channel(model, xb, tq)
                u_phys = loss_fn._denorm_batch(u_hat)

                # ZERO FORCING — consistent with training protocol
                rho_fn = lambda t: torch.zeros_like(t)

                R = compute_residuals(
                    u_hat=u_phys, du_hat_dt=du_dt, t_query=tq,
                    rho_ext_fn=rho_fn, kin=kin, th=th,
                )
                stacked = R.stacked()

                for bi in range(b):
                    scalars = []
                    for sj, k in enumerate(STATES):
                        s_j = loss_fn.residual_scales[k]["residual_p95"]
                        v = ((stacked[bi, sj, :] / s_j) ** 2).mean().item()
                        scalars.append(v)
                        per_state_physics[k].append(v)
                    per_scn_physics.append(float(np.mean(scalars)))

                    t0 = torch.zeros(b, 1, device=device)
                    u0_norm = model.decoder(z, t0)
                    u0_phys = loss_fn._denorm_batch(u0_norm)
                    y0 = torch.tensor(traj_phys[i + bi, :, 0], dtype=torch.float32, device=device)
                    per_scn_ic.append(float(((u0_phys[bi, :, 0] - y0) ** 2).mean().item()))
        except Exception as e:
            print(f"  batch {i} FAILED: {type(e).__name__}: {e}")
            for _ in range(b):
                per_scn_physics.append(float("nan"))
                per_scn_ic.append(float("nan"))

    return {
        "physics_loss_mean": float(np.nanmean(per_scn_physics)) if per_scn_physics else None,
        "ic_loss_mean": float(np.nanmean(per_scn_ic)) if per_scn_ic else None,
        "per_state_physics_mean": {k: float(np.nanmean(v)) if v else None for k, v in per_state_physics.items()},
        "per_scenario_physics": per_scn_physics,
    }


print("\n=== Physics revalidation v2 (per-channel derivative + ZERO forcing) ===")
results = {}
for name, cfg in MODELS.items():
    print(f"\n--- {name} ---")
    t0 = time.perf_counter()
    model = load_model(cfg)
    r = evaluate_physics_zero_forcing(model, forcing_norm)
    r["eval_time_s"] = time.perf_counter() - t0
    results[name] = r
    print(f"  physics_loss_mean: {r['physics_loss_mean']}")
    print(f"  ic_loss_mean: {r['ic_loss_mean']}")
    print(f"  time: {r['eval_time_s']:.2f}s")

json.dump(results, open(OUT / "physics_revalidation_v2_zero_forcing.json", "w"), indent=2)

ORIGINAL = {
    "P3-PI-FNO": 1.6723e-3, "Phase4A.1": 1.9519e-6,
    "P4-A": 174.07, "P4-B": 204.41, "P4-C": 182.28, "P4-D": 225.58,
}
rows = []
for name, r in results.items():
    orig = ORIGINAL.get(name)
    corr = r["physics_loss_mean"]
    rows.append({
        "model": name,
        "original_per_sum_loss": orig,
        "corrected_per_channel_zero_forcing_loss": corr,
        "ratio_original_over_corrected": (orig / corr) if orig and corr else None,
    })
pd.DataFrame(rows).to_csv(OUT / "comparison_table_v2.csv", index=False)
print(f"\n[OK] Results: {OUT}/physics_revalidation_v2_zero_forcing.json")
print(f"[OK] Comparison: {OUT}/comparison_table_v2.csv")
