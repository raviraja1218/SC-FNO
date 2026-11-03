"""Step 5.D.B — Zero-shot inference on locked test set (200) + unseen (100)."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW_P1 = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
RAW_P5 = Path("data/raw/phase_5_unseen")
OUT = Path("results/phase_5/step_5_D/inference")
OUT.mkdir(parents=True, exist_ok=True)
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

test_ids = pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()
unseen_manifest = pd.read_csv("results/phase_5/step_5_D/unseen_generation/unseen_manifest.csv")
unseen_ids = unseen_manifest["id"].tolist()

def load_scenario(sid, base):
    d = base / sid
    f = np.load(d / "forcing.npy")
    t = np.load(d / "trajectory.npy")
    fn = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    tn = np.zeros_like(t)
    for j, k in enumerate(STATES):
        tn[:, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
    return fn.astype(np.float32), tn.T.astype(np.float32), t

from training.adaptive_optimizer import load_phase3_pi_fno_model
from phase_5_corrected.model_v2 import SCFNOModelV2
from phase_5_corrected.decoder_v2_causal import build_causal_decoder

def load_5b_seed42():
    m3, _ = load_phase3_pi_fno_model("models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
    raw = m3.fno.fno if hasattr(m3.fno, "fno") else m3.fno
    m = SCFNOModelV2(raw, m3.decoder, pad_size=0).to(device)
    ck = torch.load("models/sc_fno/phase_5/step_5_B/seed_42_best.pt",
                    map_location=device, weights_only=False)
    sd = {k:v for k,v in ck["model_state_dict"].items() if k != "_metadata"}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def load_5c_seed456():
    m3, _ = load_phase3_pi_fno_model("models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
    raw = m3.fno.fno if hasattr(m3.fno, "fno") else m3.fno
    dec = build_causal_decoder(seed=42).to(device)
    m = SCFNOModelV2(raw, dec, pad_size=0).to(device)
    ck = torch.load("models/sc_fno/phase_5/step_5_C/seed_456_causal_best.pt",
                    map_location=device, weights_only=False)
    sd = {k:v for k,v in ck["model_state_dict"].items() if k != "_metadata"}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def predict_data_path(model, fn_batch):
    xb = torch.tensor(fn_batch, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model.forward_data(xb)
    return pred.cpu().numpy()

def evaluate_split(model, ids, base, label):
    B = 16
    per_scn_per_state = np.zeros((len(ids), 9), dtype=np.float64)
    boundary_start = {k: 0.0 for k in STATES}
    boundary_interior = {k: 0.0 for k in STATES}
    boundary_end = {k: 0.0 for k in STATES}
    n = 0
    for i in range(0, len(ids), B):
        batch_ids = ids[i:i+B]
        fn_b, traj_phys = [], []
        for sid in batch_ids:
            fn, tn, tp = load_scenario(sid, base)
            fn_b.append(fn); traj_phys.append(tp)
        fn_b = np.stack(fn_b)
        pred_n = predict_data_path(model, fn_b)
        for bi in range(len(batch_ids)):
            for sj, k in enumerate(STATES):
                gt_phys = traj_phys[bi][:, sj]
                pred_phys = pred_n[bi, sj] * norm[k]["std"] + norm[k]["mean"]
                d = np.linalg.norm(gt_phys)
                if d > 1e-30:
                    per_scn_per_state[i+bi, sj] = np.linalg.norm(gt_phys - pred_phys) / d
                else:
                    per_scn_per_state[i+bi, sj] = np.linalg.norm(gt_phys - pred_phys)
                diff = np.abs(gt_phys - pred_phys)
                boundary_start[k] += diff[:20].mean()
                boundary_interior[k] += diff[20:180].mean()
                boundary_end[k] += diff[180:].mean()
            n += 1
    per_scn_mean = per_scn_per_state.mean(axis=1)
    result = {
        "label": label,
        "n_scenarios": len(ids),
        "overall_median": float(np.median(per_scn_mean)),
        "overall_mean": float(np.mean(per_scn_mean)),
        "per_state_median": {k: float(np.median(per_scn_per_state[:, j]))
                              for j, k in enumerate(STATES)},
        "boundary_ratios": {},
    }
    for k in STATES:
        s = boundary_start[k] / max(n, 1)
        i = boundary_interior[k] / max(n, 1)
        e = boundary_end[k] / max(n, 1)
        result["boundary_ratios"][k] = {
            "start_interior": s / i if i > 1e-30 else None,
        }
    return result

def main():
    models = {}
    try:
        models["5B_noncausal_s42"] = load_5b_seed42()
        print("[OK] Loaded 5.B seed 42")
    except Exception as e:
        print(f"[FAIL] 5.B load: {e}")
    try:
        models["5C_causal_s456"] = load_5c_seed456()
        print("[OK] Loaded 5.C seed 456")
    except Exception as e:
        print(f"[FAIL] 5.C load: {e}")

    results = {}
    in_dist_ids = unseen_manifest[unseen_manifest["in_distribution"] == True]["id"].tolist()
    ood_ids = unseen_manifest[unseen_manifest["in_distribution"] == False]["id"].tolist()

    for mname, model in models.items():
        print(f"\n=== {mname} ===")
        r_test = evaluate_split(model, test_ids, RAW_P1, f"{mname}_test")
        print(f"  test 200: median = {r_test['overall_median']:.4e}, B_n = {r_test['boundary_ratios']['n']['start_interior']:.2f}×")
        results[f"{mname}_test"] = r_test

        r_unseen = evaluate_split(model, unseen_ids, RAW_P5, f"{mname}_unseen")
        print(f"  unseen 100: median = {r_unseen['overall_median']:.4e}, B_n = {r_unseen['boundary_ratios']['n']['start_interior']:.2f}×")
        results[f"{mname}_unseen"] = r_unseen

        r_in = evaluate_split(model, in_dist_ids, RAW_P5, f"{mname}_unseen_indist")
        r_ood = evaluate_split(model, ood_ids, RAW_P5, f"{mname}_unseen_ood")
        print(f"  in-dist (50): median = {r_in['overall_median']:.4e}")
        print(f"  OOD (50):     median = {r_ood['overall_median']:.4e}")
        results[f"{mname}_unseen_indist"] = r_in
        results[f"{mname}_unseen_ood"] = r_ood

    json.dump(results, open(OUT / "inference_results.json", "w"), indent=2)
    print(f"\nSaved: {OUT}/inference_results.json")

if __name__ == "__main__":
    main()
