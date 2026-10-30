"""Step 2.4 — Test-set evaluation of Phase 2 FNO baseline.

Loads fno_best.pt (epoch 289), runs inference on 200 untouched test scenarios,
saves per-scenario predictions (normalized + denormalized), then computes
per-state, per-family, and scenario-level metrics.
"""
import json, sys, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from neuralop.models import FNO

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from data.dataset import make_dataloader, Phase1TransientDataset, NORM_PATH, STATE_ORDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = PROJECT / "models/baseline"
OUT = PROJECT / "results/phase_2/evaluation"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"


def build_fno():
    return FNO(n_modes=(16,), hidden_channels=32, in_channels=1, out_channels=9,
               n_layers=4, positional_embedding=None, domain_padding=None)


def main():
    # ---- Load checkpoint ----
    ckpt_path = MODELS / "fno_best.pt"
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_fno().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded fno_best.pt | epoch={ckpt.get('epoch')} | val_mse={ckpt.get('val_mse'):.6e}")
    print(f"Checkpoint SHA-256: {ckpt_sha}")

    # ---- Load test dataloader (shuffle=False, deterministic) ----
    test_dl = make_dataloader("test", batch_size=16, shuffle=False)
    norm = json.load(open(NORM_PATH))["global"]
    print(f"Test scenarios: {len(test_dl.dataset)} | Batches: {len(test_dl)}")

    # ---- Predict + save per-scenario ----
    preds_dir = OUT / "test_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    all_sids = []
    all_pred_n = []  # normalized preds (N,9,200)
    all_true_n = []  # normalized truth
    all_pred_p = []  # physical preds
    all_true_p = []  # physical truth

    with torch.no_grad():
        for x, y, sids in test_dl:
            x = x.to(DEVICE)
            y_hat = model(x)  # [B,9,200]

            x_cpu = x.cpu().numpy()
            y_cpu = y.cpu().numpy()       # [B,9,200]
            yhat_cpu = y_hat.cpu().numpy()

            # Denormalize: per-state mean/std
            means = np.array([norm[v]["mean"] for v in STATE_ORDER])[:, None]  # (9,1)
            stds  = np.array([norm[v]["std"]  for v in STATE_ORDER])[:, None]
            yhat_p = yhat_cpu * stds + means
            y_p    = y_cpu * stds + means

            for i, sid in enumerate(sids):
                all_sids.append(sid)
                # store normalized
                np.save(preds_dir / f"{sid}_pred_normalized.npy", yhat_cpu[i])
                np.save(preds_dir / f"{sid}_true_normalized.npy", y_cpu[i])
                # store physical
                np.save(preds_dir / f"{sid}_pred_physical.npy", yhat_p[i])
                np.save(preds_dir / f"{sid}_true_physical.npy", y_p[i])

                all_pred_n.append(yhat_cpu[i])
                all_true_n.append(y_cpu[i])
                all_pred_p.append(yhat_p[i])
                all_true_p.append(y_p[i])

    all_pred_n = np.stack(all_pred_n)   # (200,9,200)
    all_true_n = np.stack(all_true_n)
    all_pred_p = np.stack(all_pred_p)
    all_true_p = np.stack(all_true_p)
    sids = np.array(all_sids)

    # Save consolidated
    np.savez_compressed(OUT / "predictions_summary.npz",
        sids=sids,
        pred_norm=all_pred_n, true_norm=all_true_n,
        pred_phys=all_pred_p, true_phys=all_true_p,
        state_order=np.array(STATE_ORDER),
    )
    print(f"Saved predictions_summary.npz | shape: {all_pred_p.shape}")

    # ---- Metrics: per-state, per-scenario, per-family ----
    # Family lookup from split CSV (if column absent, look up via metadata.json forcing_family)
    test_df = pd.read_csv(PROJECT / "results/phase_1/splitting/test_ids.csv")
    family_col = "forcing_family" if "forcing_family" in test_df.columns else None

    if family_col is None:
        # Build family map from metadata.json
        fam_map = {}
        for sid in sids:
            meta = json.load(open(RAW / sid / "metadata.json"))
            fam = meta.get("forcing_family") or meta.get("family") or "unknown"
            fam_map[sid] = fam
        families = np.array([fam_map[s] for s in sids])
    else:
        fm = dict(zip(test_df["scenario_id"], test_df[family_col]))
        families = np.array([fm[s] for s in sids])

    # Per-scenario, per-state metrics
    EPS = 1e-8
    per_scen_rows = []
    for i, sid in enumerate(sids):
        for j, state in enumerate(STATE_ORDER):
            y = all_true_p[i, j]      # (200,)
            yh = all_pred_p[i, j]
            err = yh - y
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            maxabs = float(np.max(np.abs(err)))
            l2 = float(np.linalg.norm(y))
            rel_l2 = float(np.linalg.norm(err) / l2) if l2 > EPS else None
            per_scen_rows.append({
                "scenario_id": sid, "family": families[i], "state": state,
                "mae": mae, "rmse": rmse, "max_abs_err": maxabs,
                "rel_l2": rel_l2, "rel_l2_defined": rel_l2 is not None,
            })

    scen_df = pd.DataFrame(per_scen_rows)
    scen_df.to_csv(OUT / "scenario_level_metrics.csv", index=False)
    print(f"scenario_level_metrics.csv: {len(scen_df)} rows")

    # Per-state aggregate (mean over scenarios, per metric)
    per_state = scen_df.groupby("state").agg(
        mae_mean=("mae", "mean"), mae_median=("mae", "median"),
        rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
        maxabs_mean=("max_abs_err", "mean"), maxabs_max=("max_abs_err", "max"),
        rel_l2_mean=("rel_l2", "mean"), rel_l2_median=("rel_l2", "median"),
    ).reset_index()
    per_state.to_csv(OUT / "metrics_per_state.csv", index=False)
    print(per_state.to_string(index=False))

    # Per-family aggregate
    per_family = scen_df.groupby("family").agg(
        mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"),
        rel_l2_median=("rel_l2", "median"), maxabs_max=("max_abs_err", "max"),
    ).reset_index()
    per_family.to_csv(OUT / "metrics_per_family.csv", index=False)

    # Family × state
    per_fs = scen_df.groupby(["family", "state"]).agg(
        mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"),
        rel_l2_median=("rel_l2", "median"),
    ).reset_index()
    per_fs.to_csv(OUT / "metrics_per_family_per_state.csv", index=False)

    # Aggregate JSON
    skipped = int(scen_df["rel_l2_defined"].eq(False).sum())
    agg = {
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": int(ckpt.get("epoch")),
        "checkpoint_val_mse": float(ckpt.get("val_mse")),
        "n_scenarios": int(len(sids)),
        "n_timesteps": 200,
        "rel_l2_skipped_count": skipped,
        "rel_l2_eps": EPS,
        "overall_mae_mean": float(scen_df["mae"].mean()),
        "overall_rmse_mean": float(scen_df["rmse"].mean()),
        "overall_rel_l2_median": float(scen_df["rel_l2"].median()),
        "per_state": per_state.to_dict(orient="records"),
        "per_family": per_family.to_dict(orient="records"),
        "state_order": STATE_ORDER,
        "families": sorted(set(families.tolist())),
    }
    json.dump(agg, open(OUT / "aggregate_metrics.json", "w"), indent=2)
    print(f"\nAggregate: MAE {agg['overall_mae_mean']:.4e} | RMSE {agg['overall_rmse_mean']:.4e} | "
          f"RelL2 median {agg['overall_rel_l2_median']:.4e} | skipped {skipped}")

    # ---- Worst-case (aggregate across 9 states) ----
    scen_agg = scen_df[scen_df["rel_l2_defined"]].groupby("scenario_id").agg(
        rel_l2_mean=("rel_l2", "mean")
    ).reset_index()
    worst = scen_agg.sort_values("rel_l2_mean", ascending=False).iloc[0]
    worst_sid = worst["scenario_id"]
    worst_family = scen_df[scen_df["scenario_id"] == worst_sid]["family"].iloc[0]
    worst_states = scen_df[scen_df["scenario_id"] == worst_sid].to_dict(orient="records")
    json.dump({
        "scenario_id": worst_sid,
        "family": worst_family,
        "aggregate_rel_l2_mean": float(worst["rel_l2_mean"]),
        "per_state_metrics": worst_states,
    }, open(OUT / "worst_case_aggregate.json", "w"), indent=2)
    print(f"Worst aggregate scenario: {worst_sid} ({worst_family}) | relL2={worst['rel_l2_mean']:.4e}")

    # ---- Worst-case per state ----
    per_state_worst = {}
    for state in STATE_ORDER:
        sub = scen_df[(scen_df["state"] == state) & scen_df["rel_l2_defined"]]
        if len(sub) == 0: continue
        row = sub.loc[sub["rel_l2"].idxmax()]
        per_state_worst[state] = {
            "scenario_id": row["scenario_id"],
            "family": row["family"],
            "rel_l2": float(row["rel_l2"]),
            "mae": float(row["mae"]),
            "max_abs_err": float(row["max_abs_err"]),
        }
    json.dump(per_state_worst, open(OUT / "worst_case_per_state.json", "w"), indent=2)
    print(f"Per-state worst-cases recorded: {len(per_state_worst)}")

    print("\n=== 2.4.A/B/C/D DONE ===")


if __name__ == "__main__":
    main()
