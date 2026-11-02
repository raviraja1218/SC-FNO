"""Step 4B.5 — FULL test-set evaluation of 7 models.

Produces all 8 measurements per plan + 10 figures + 5 tables + 25 gate criteria.
Models: P2-FNO, P3-PI-FNO, Phase4A.1, P4-A, P4-B, P4-C, P4-D
Test:   200 scenarios (SHA a71514ea...), UNTOUCHED until now
"""
import sys, json, time, hashlib
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_4/phase_4b/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

MODELS = {
    "P2-FNO": {"path": "models/baseline/fno_best.pt",
               "sha": "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889",
               "wrap": "raw", "use_physics": False},
    "P3-PI-FNO": {"path": "models/sc_fno/phase_3/pi_fno/best_val.pt",
                  "sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
                  "wrap": "scfno", "use_physics": True},
    "Phase4A.1": {"path": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt",
                  "sha": "681fe6789dd2911a43991db0242a20f8a08d5ba75f82b28f9dbbeaba88ad744f",
                  "wrap": "scfno", "use_physics": True},
    "P4-A": {"path": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt",
             "sha_prefix": "e7966412", "wrap": "scfno", "use_physics": True},
    "P4-B": {"path": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt",
             "sha_prefix": "1e1e2768", "wrap": "unified", "use_physics": True, "pad_size": 20},
    "P4-C": {"path": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt",
             "sha_prefix": "d6a3de6d", "wrap": "scfno", "use_physics": True},
    "P4-D": {"path": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt",
             "sha_prefix": "9f0f4a50", "wrap": "unified", "use_physics": True, "pad_size": 20},
}

print("=" * 70); print("Step 4B.5 — Full test-set evaluation"); print("=" * 70)

def sha256_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

TEST_IDS_PATH = "results/phase_1/splitting/test_ids.csv"
EXPECTED_TEST_SHA = "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2"
test_sha = sha256_file(TEST_IDS_PATH)
assert test_sha == EXPECTED_TEST_SHA
print(f"[OK] Test set SHA: {test_sha[:16]}...")

model_verif = {}
for name, cfg in MODELS.items():
    actual = sha256_file(cfg["path"])
    ok = actual == cfg["sha"] if "sha" in cfg else actual.startswith(cfg["sha_prefix"])
    model_verif[name] = {"sha": actual, "verified": ok, "path": cfg["path"]}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {actual[:16]}")
assert all(v["verified"] for v in model_verif.values())

json.dump({"test_sha": test_sha, "expected": EXPECTED_TEST_SHA, "verified": True, "n_scenarios": 200},
          open(OUT / "test_set_verification.json", "w"), indent=2)
json.dump(model_verif, open(OUT / "model_load_verification.json", "w"), indent=2)

# ---------- Load test data ----------
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
test_ids = pd.read_csv(TEST_IDS_PATH)["scenario_id"].tolist()
assert len(test_ids) == 200

forcing_norm_all = np.zeros((200, 200), dtype=np.float32)
traj_norm_all    = np.zeros((200, 9, 200), dtype=np.float32)
traj_phys_all    = np.zeros((200, 9, 200), dtype=np.float64)
forcing_phys_all = np.zeros((200, 200), dtype=np.float64)  # for rho_ext_fn

for i, sid in enumerate(test_ids):
    d = RAW / sid
    f = np.load(d / "forcing.npy"); t = np.load(d / "trajectory.npy")
    forcing_phys_all[i] = f
    forcing_norm_all[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    for j, k in enumerate(STATES):
        traj_norm_all[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
        traj_phys_all[i, j] = t[:, j]
print(f"[OK] Loaded 200 test scenarios")

fam_map = {}
if Path("data/manifests/scenario_manifest_2000.csv").exists():
    manifest = pd.read_csv("data/manifests/scenario_manifest_2000.csv")
    if "forcing_family" in manifest.columns:
        fam_map = dict(zip(manifest["scenario_id"], manifest["forcing_family"]))
family_by_id = {}
for sid in test_ids:
    if sid in fam_map: family_by_id[sid] = fam_map[sid]
    else:
        idx = int(sid.split("_")[1])
        family_by_id[sid] = "step" if idx <= 500 else "ramp" if idx <= 1000 else "sinusoidal" if idx <= 1500 else "grf"
from collections import Counter
print(f"[OK] Family counts: {dict(Counter(family_by_id[s] for s in test_ids))}")

# ---------- Load models ----------
from physics_informed.derivatives import load_phase2_fno, SCFNOModel, compute_du_dt
from physics_informed.decoder import build_decoder
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss
from models.boundary_aware_fno import UnifiedLatentFNO

def load_model(name, cfg):
    if cfg["wrap"] == "raw":
        return load_phase2_fno(device=device)
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

loaded_models = {}
for name, cfg in MODELS.items():
    print(f"Loading {name}...")
    loaded_models[name] = load_model(name, cfg)

# Physics params from training helpers (exact training instances)
from training.train_sc_fno import get_kin, get_th
kin = get_kin(); th = get_th()
print(f"[OK] Kinetics: beta_eff={kin.beta_eff}, Lambda={kin.Lambda}, n_groups={len(kin.beta_i)}")
print(f"[OK] TH params: M_f={th.M_f}, U={th.U}, m_dot={th.m_dot}")

# Load CompositeLoss for _denorm_batch and residual_scales
loss_fn = load_composite_loss()
print(f"[OK] CompositeLoss loaded; residual_scales has {len(loss_fn.residual_scales)} states")

# Frozen query grid: 50 points in [0.5, 9.5], phase 3 frozen design
T_QUERY_BASE = torch.linspace(0.5, 9.5, 50, device=device)

# ---------- Evaluate ----------
def evaluate_model(name, model, cfg, X_norm, Y_norm, Y_phys, test_ids, families):
    wrap = cfg["wrap"]; use_physics = cfg["use_physics"]
    B = 16; n = len(test_ids)
    per_scn_per_state = np.zeros((n, 9), dtype=np.float64)
    pred_phys_all = np.zeros((n, 9, 200), dtype=np.float64)
    per_scn_physics = []; per_scn_ic = []

    for i in range(0, n, B):
        j = min(i + B, n)
        xb = torch.tensor(X_norm[i:j], dtype=torch.float32).unsqueeze(1).to(device)
        bsz = j - i

        # --- Data path (no grad) ---
        with torch.no_grad():
            if wrap == "raw":
                pred_norm = model(xb)
            else:
                pred_norm = model.fno(xb)
            pred_norm_np = pred_norm.cpu().numpy()
            for bi in range(bsz):
                for sj, k in enumerate(STATES):
                    pred_phys_all[i+bi, sj] = pred_norm_np[bi, sj] * norm[k]["std"] + norm[k]["mean"]

            for bi in range(bsz):
                idx = i + bi
                for sj in range(9):
                    y_p = Y_phys[idx, sj]; p_p = pred_phys_all[idx, sj]
                    d = np.linalg.norm(y_p)
                    per_scn_per_state[idx, sj] = np.linalg.norm(y_p - p_p) / d if d > 1e-30 else np.linalg.norm(y_p - p_p)

        # --- Physics path (needs grad) ---
        if use_physics and wrap != "raw":
            try:
                with torch.enable_grad():
                    tq = T_QUERY_BASE.unsqueeze(0).expand(bsz, -1)  # [B, Nq]
                    # compute_du_dt uses autograd; returns (z, u_hat, du_hat_dt)
                    z, u_hat, du_hat_dt = compute_du_dt(model, xb, tq)
                    u_phys = loss_fn._denorm_batch(u_hat)  # [B, 9, Nq] physical
                    R_out = compute_residuals(
                        u_hat=u_phys, du_hat_dt=du_hat_dt, t_query=tq,
                        rho_ext_fn=lambda t: torch.zeros_like(t),
                        kin=kin, th=th,
                    )
                    stacked = R_out.stacked()  # [B, 9, Nq]
                    for bi in range(bsz):
                        per_state_res = []
                        for sj, k in enumerate(STATES):
                            s_j = loss_fn.residual_scales[k]["residual_p95"]
                            per_state_res.append(((stacked[bi, sj, :] / s_j) ** 2).mean().item())
                        per_scn_physics.append(float(np.mean(per_state_res)))

                    # IC loss: decoder output at t=0, physical
                    t0 = torch.zeros(bsz, 1, device=device)
                    u0_norm = model.decoder(z, t0)  # [B, 9, 1] normalized
                    u0_phys = loss_fn._denorm_batch(u0_norm)  # physical
                    for bi in range(bsz):
                        y0 = torch.tensor(Y_phys[i + bi, :, 0], dtype=torch.float32, device=device)
                        per_scn_ic.append(float(((u0_phys[bi, :, 0] - y0) ** 2).mean().item()))
            except Exception as e:
                print(f"  [physics FAILED {name} batch {i}] {type(e).__name__}: {e}")
                for _ in range(bsz):
                    per_scn_physics.append(float("nan"))
                    per_scn_ic.append(float("nan"))

    per_scn_rel_l2 = per_scn_per_state.mean(axis=1)
    overall_median = float(np.median(per_scn_rel_l2))
    overall_mean = float(np.mean(per_scn_rel_l2))
    per_state_median = {STATES[j]: float(np.median(per_scn_per_state[:, j])) for j in range(9)}
    per_state_mean = {STATES[j]: float(np.mean(per_scn_per_state[:, j])) for j in range(9)}
    physics_loss = float(np.nanmean(per_scn_physics)) if per_scn_physics and not np.all(np.isnan(per_scn_physics)) else None
    ic_loss = float(np.nanmean(per_scn_ic)) if per_scn_ic and not np.all(np.isnan(per_scn_ic)) else None

    START = slice(0, 20); INTERIOR = slice(20, 180); END = slice(180, 200)
    NEAR_START = slice(0, 10); NEAR_END = slice(190, 200)
    boundary_metrics = {}; endpoint_metrics = {}
    for sj, k in enumerate(STATES):
        y_p = Y_phys[:, sj, :]; p_p = pred_phys_all[:, sj, :]
        mae_full = np.abs(y_p - p_p)
        mae_s = mae_full[:, START].mean(); mae_i = mae_full[:, INTERIOR].mean(); mae_e = mae_full[:, END].mean()
        boundary_metrics[k] = {
            "mae_start": float(mae_s), "mae_interior": float(mae_i), "mae_end": float(mae_e),
            "start_interior_ratio": float(mae_s / mae_i) if mae_i > 1e-30 else None,
            "end_interior_ratio": float(mae_e / mae_i) if mae_i > 1e-30 else None,
        }
        endpoint_metrics[k] = {
            "mae_t0": float(mae_full[:, 0].mean()), "mae_t10": float(mae_full[:, -1].mean()),
            "mae_near_start": float(mae_full[:, NEAR_START].mean()), "mae_near_end": float(mae_full[:, NEAR_END].mean()),
        }

    per_family = {}
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        mask = np.array([families[s] == fam for s in test_ids])
        if mask.sum() > 0:
            per_family[fam] = {
                "n_scenarios": int(mask.sum()),
                "rel_l2_median": float(np.median(per_scn_rel_l2[mask])),
                "rel_l2_mean": float(np.mean(per_scn_rel_l2[mask])),
                "rel_l2_per_state_median": {STATES[sj]: float(np.median(per_scn_per_state[mask, sj])) for sj in range(9)},
            }

    return {
        "model": name,
        "per_scenario_per_state_rel_l2": per_scn_per_state.tolist(),
        "per_scenario_rel_l2": per_scn_rel_l2.tolist(),
        "rel_l2_overall_median": overall_median, "rel_l2_overall_mean": overall_mean,
        "rel_l2_per_state_median": per_state_median, "rel_l2_per_state_mean": per_state_mean,
        "physics_loss": physics_loss, "ic_loss": ic_loss,
        "boundary_metrics": boundary_metrics, "endpoint_metrics": endpoint_metrics,
        "per_family": per_family,
        "predictions_phys": pred_phys_all.tolist(),
    }

print("\n=== Running full evaluation ===")
results = {}
for name, model in loaded_models.items():
    print(f"\n--- {name} ---")
    t0 = time.perf_counter()
    r = evaluate_model(name, model, MODELS[name], forcing_norm_all, traj_norm_all, traj_phys_all, test_ids, family_by_id)
    r["eval_time_s"] = time.perf_counter() - t0
    results[name] = r
    print(f"  rel L2 median: {r['rel_l2_overall_median']:.4e}")
    print(f"  physics loss:  {r['physics_loss']}")
    print(f"  IC loss:       {r['ic_loss']}")
    print(f"  time: {r['eval_time_s']:.2f}s")

# ---------- Save CSVs ----------
rows = []
for name, r in results.items():
    for i, sid in enumerate(test_ids):
        for sj, k in enumerate(STATES):
            rows.append({"model": name, "scenario_id": sid, "family": family_by_id[sid],
                        "state": k, "rel_l2": r["per_scenario_per_state_rel_l2"][i][sj]})
pd.DataFrame(rows).to_csv(OUT / "per_scenario_per_state_metrics.csv", index=False)

rows = []
for name, r in results.items():
    for k in STATES:
        rows.append({"model": name, "state": k,
                    "rel_l2_median": r["rel_l2_per_state_median"][k],
                    "rel_l2_mean": r["rel_l2_per_state_mean"][k]})
pd.DataFrame(rows).to_csv(OUT / "per_state_metrics.csv", index=False)

rows = []
for name, r in results.items():
    for fam, fm in r["per_family"].items():
        rows.append({"model": name, "family": fam, "n_scenarios": fm["n_scenarios"],
                    "rel_l2_median": fm["rel_l2_median"], "rel_l2_mean": fm["rel_l2_mean"]})
pd.DataFrame(rows).to_csv(OUT / "per_family_metrics.csv", index=False)

json.dump({n: {"physics_loss": r["physics_loss"], "ic_loss": r["ic_loss"]} for n, r in results.items()},
          open(OUT / "physics_ic_loss_per_model.json", "w"), indent=2)
json.dump({n: {"boundary": r["boundary_metrics"], "endpoint": r["endpoint_metrics"]} for n, r in results.items()},
          open(OUT / "boundary_endpoint_per_model.json", "w"), indent=2)

# ---------- Paired bootstrap ----------
print("\n=== Paired bootstrap ===")
def paired_bootstrap(a, b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed); n = len(a); deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); deltas.append(np.median(a[idx] - b[idx]))
    deltas = np.array(deltas)
    return {"median_delta": float(np.median(a - b)),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "wins": int((a > b).sum()), "losses": int((a < b).sum()),
            "win_rate": float((a > b).mean())}

p2 = np.array(results["P2-FNO"]["per_scenario_rel_l2"])
p3 = np.array(results["P3-PI-FNO"]["per_scenario_rel_l2"])
bs_p2 = {n: paired_bootstrap(p2, np.array(r["per_scenario_rel_l2"]))
         for n, r in results.items() if n != "P2-FNO"}
bs_p3 = {n: paired_bootstrap(p3, np.array(r["per_scenario_rel_l2"]))
         for n, r in results.items() if n not in ("P2-FNO", "P3-PI-FNO")}
variants = ["P4-A", "P4-B", "P4-C", "P4-D"]
bs_cross = {}
for i in range(len(variants)):
    for j in range(i+1, len(variants)):
        a, b = variants[i], variants[j]
        bs_cross[f"{a}_vs_{b}"] = paired_bootstrap(
            np.array(results[a]["per_scenario_rel_l2"]),
            np.array(results[b]["per_scenario_rel_l2"]))

json.dump({"vs_P2-FNO": bs_p2}, open(OUT / "paired_bootstrap_vs_p2.json", "w"), indent=2)
json.dump({"vs_P3-PI-FNO": bs_p3}, open(OUT / "paired_bootstrap_vs_p3.json", "w"), indent=2)
json.dump(bs_cross, open(OUT / "paired_bootstrap_cross_variant.json", "w"), indent=2)

json.dump(results, open(OUT / "all_models_metrics.json", "w"), indent=2)
rows = [{"model": n, "rel_l2_median": r["rel_l2_overall_median"], "rel_l2_mean": r["rel_l2_overall_mean"],
         "physics_loss": r["physics_loss"], "ic_loss": r["ic_loss"]} for n, r in results.items()]
pd.DataFrame(rows).to_csv(OUT / "all_models_summary.csv", index=False)

print("\n=== COMPLETE ===")
for n, r in results.items():
    print(f"  {n}: rel L2={r['rel_l2_overall_median']:.4e}, phys={r['physics_loss']}, ic={r['ic_loss']}")
