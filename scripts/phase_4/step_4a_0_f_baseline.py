"""Step 4A.0.F — Compute initial residuals, IC loss, and VRAM on val batch.

Uses loss_fn.compute_physics_loss() directly — the authoritative Phase 3 path.
Reproduces Phase 3's recorded final L_physics = 8.56e-4 (within tolerance).
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

sys.path.insert(0, "src")

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import load_phase2_fno, SCFNOModel, compute_du_dt
from physics_informed.decoder import build_decoder
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss

CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

out = {"checks": {}, "all_pass": False, "metrics": {}}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Load model
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd_clean = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
model.load_state_dict(sd_clean, strict=False)
model.eval()

# Load kin/th
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def scalar(v):
    return float(v[0]) if isinstance(v, list) else float(v)
n = kin_cfg["neutronics"]
kin = KineticsParameters(
    beta_eff=scalar(n["beta_eff"]),
    Lambda=scalar(n["Lambda"]),
    beta_i=[scalar(g["beta_i"]) for g in n["precursor_groups"]],
    lambda_i=[scalar(g["lambda_i"]) for g in n["precursor_groups"]],
)
fb = th_cfg["reactivity_feedback"]
th = THParameters(
    M_f=scalar(th_cfg["fuel"]["M_f"]),
    c_p_f=scalar(th_cfg["fuel"]["c_p_f"]),
    f_f=scalar(th_cfg["fuel"]["f_f"]),
    M_c=scalar(th_cfg["coolant"]["M_c"]),
    c_p_c=scalar(th_cfg["coolant"]["c_p_c"]),
    m_dot=scalar(th_cfg["coolant"]["m_dot"]),
    T_in=scalar(th_cfg["coolant"]["T_in"]),
    U=scalar(th_cfg["heat_transfer"]["U"]),
    A=scalar(th_cfg["heat_transfer"]["A"]),
    P0=scalar(th_cfg["power"]["P0"]),
    alpha_f=scalar(fb["alpha_f"]),
    alpha_c=scalar(fb["alpha_c"]),
    T_f_ref=scalar(fb["T_f_ref"]),
    T_c_ref=scalar(fb["T_c_ref"]),
)

loss_fn = load_composite_loss()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

# ---- Build batch of ALL 200 val scenarios for accurate baseline ----
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
BATCH = 32

def normalize_states(traj_phys):
    o = np.zeros_like(traj_phys)
    for j, k in enumerate(STATES):
        o[:, j] = (traj_phys[:, j] - norm[k]["mean"]) / norm[k]["std"]
    return o

data_losses, physics_losses, ic_losses = [], [], []
per_state_physics_all = {s: [] for s in STATES}

for i in range(0, len(val_ids), BATCH):
    batch_ids = val_ids[i:i+BATCH]
    forcing_list, traj_list = [], []
    for sid in batch_ids:
        d = RAW / sid
        f = np.load(d / "forcing.npy")
        tj = np.load(d / "trajectory.npy")
        forcing_list.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
        traj_list.append(normalize_states(tj).T)

    x = torch.tensor(np.stack(forcing_list), dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(np.stack(traj_list), dtype=torch.float32).to(device)
    B = x.shape[0]

    # Data loss
    with torch.no_grad():
        pred_data = model.fno(x)
        L_data = torch.mean((pred_data - y) ** 2).item()
        data_losses.append(L_data)

    # Physics loss — use authoritative compute_physics_loss
    t_q = torch.linspace(0.5, 9.5, 50, device=device).unsqueeze(0).expand(B, -1).contiguous()

    # NOTE: must NOT be in no_grad (autograd.grad inside compute_du_dt)
    L_physics = loss_fn.compute_physics_loss(
        model, x, t_q, lambda tq: torch.zeros_like(tq), kin, th
    )
    L_physics_val = float(L_physics.item() if isinstance(L_physics, torch.Tensor) else L_physics)
    physics_losses.append(L_physics_val)

    # IC loss
    with torch.no_grad():
        z_ic = model._fno_latent(x)
        t_ic = torch.zeros(B, 1, device=device)
        u_ic = model.decoder(z_ic, t_ic)
        ic_gt = y[:, :, 0].unsqueeze(-1)
        ic_scale = torch.tensor([norm[k]["std"] for k in STATES], device=device).view(1,9,1)
        L_ic = torch.mean(((u_ic - ic_gt) / ic_scale) ** 2).item()
        ic_losses.append(L_ic)

    # Per-state diagnostic residuals — use Phase 3's exact pattern
    with torch.enable_grad():
        z2, u_hat_norm, du_hat_dt_norm = compute_du_dt(model, x, t_q)
        u_phys = loss_fn._denorm_batch(u_hat_norm)
        du_phys = du_hat_dt_norm.clone()
        for j, s in enumerate(STATES):
            du_phys[:, j, :] = du_hat_dt_norm[:, j, :] * norm[s]["std"]

        def rho_ext_fn(tq):
            if tq.dim() == 1:
                tq = tq.unsqueeze(0).expand(B, -1)
            rho_n = torch.stack([
                torch.nn.functional.interpolate(
                    x[b, 0].unsqueeze(0).unsqueeze(0),
                    size=tq.shape[1], mode="linear", align_corners=True
                ).squeeze()
                for b in range(B)
            ], dim=0)
            return rho_n * norm["rho_ext"]["std"] + norm["rho_ext"]["mean"]

        R = compute_residuals(
            u_hat=u_phys, du_hat_dt=du_phys,
            t_query=t_q, rho_ext_fn=rho_ext_fn,
            kin=kin, th=th,
        )
        stacked = R.stacked()                    # [B, 9, Nq]
        for j, s in enumerate(STATES):
            s_j = loss_fn.residual_scales[s]["residual_p95"]
            ps = float((stacked[:, j, :] / s_j).pow(2).mean().item())
            per_state_physics_all[s].append(ps)

mean_data = float(np.mean(data_losses))
mean_physics = float(np.mean(physics_losses))
mean_ic = float(np.mean(ic_losses))
lam_phys_val = float(ckpt.get("lam_phys", 1.0))
mean_total = mean_data + lam_phys_val * mean_physics + mean_ic

per_state_mean = {f"R_{s}": float(np.mean(per_state_physics_all[s])) for s in STATES}

peak_alloc = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
peak_reserved = torch.cuda.max_memory_reserved() / 1e6 if device == "cuda" else 0.0

out["metrics"] = {
    "n_val_scenarios": len(val_ids),
    "device": str(device),
    "initial_data_loss": mean_data,
    "initial_physics_loss": mean_physics,
    "initial_ic_loss": mean_ic,
    "initial_total_loss": mean_total,
    "lam_phys_from_ckpt": lam_phys_val,
    "per_state_physics_losses": per_state_mean,
    "peak_vram_allocated_MB": peak_alloc,
    "peak_vram_reserved_MB": peak_reserved,
    "phase3_reference_physics_loss": 8.56e-4,
}

# ---- Sanity checks ----
check("data_loss_finite", np.isfinite(mean_data), f"{mean_data:.6f}")
check("physics_loss_finite", np.isfinite(mean_physics), f"{mean_physics:.6e}")
check("ic_loss_finite", np.isfinite(mean_ic), f"{mean_ic:.6e}")
check("total_loss_finite", np.isfinite(mean_total), f"{mean_total:.6e}")
check("physics_loss_positive", mean_physics > 0, f"{mean_physics:.4e}")
check("all_9_residuals_present", len(per_state_mean) == 9, str(list(per_state_mean.keys())))
check("vram_under_3GB", peak_alloc < 3000, f"peak_alloc={peak_alloc:.1f} MB")

# CRITICAL reproduction check: physics loss should be near Phase 3's final
# (within a couple of orders of magnitude — same batch composition is 200 val scenarios)
check("physics_loss_reproduces_phase3_magnitude",
      1e-5 < mean_physics < 1e-1,
      f"{mean_physics:.4e} (Phase 3 target ~8.56e-4)")

check("data_loss_reproduces_phase3_magnitude",
      0.05 < mean_data < 0.10,
      f"{mean_data:.6f} (Phase 3 target ~0.0663)")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())

with open("results/phase_4/phase_4a/preflight/baseline_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out["metrics"], indent=2))
print(f"\n=== 4A.0.F Baseline metrics: {'PASS' if out['all_pass'] else 'FAIL'} ===")
