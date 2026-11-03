"""Step 5.B — Train SCFNOModelV2 with 4-term composite loss. One seed per run."""
import sys, json, time, argparse
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from training.adaptive_optimizer import load_phase3_pi_fno_model
from physics_informed.losses import load_composite_loss
from training.train_sc_fno import get_kin, get_th
from phase_5_corrected.model_v2 import SCFNOModelV2
from phase_5_corrected.losses_v2 import compute_composite_loss_v2
from phase_5_corrected.residuals_v2 import make_rho_ext_fn
from phase_5_corrected.derivatives_v2 import compute_du_dt_per_channel

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

def load_data():
    norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
    tr = pd.read_csv("results/phase_1/splitting/train_ids.csv")["scenario_id"].tolist()
    va = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
    def load(ids):
        X = np.zeros((len(ids), 200), dtype=np.float32)
        Y = np.zeros((len(ids), 9, 200), dtype=np.float32)
        for i, sid in enumerate(ids):
            d = RAW / sid
            f = np.load(d / "forcing.npy"); t = np.load(d / "trajectory.npy")
            X[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
            for j, k in enumerate(STATES):
                Y[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
        return torch.tensor(X), torch.tensor(Y)
    Xtr, Ytr = load(tr)
    Xva, Yva = load(va)
    return Xtr, Ytr, Xva, Yva

def build_model(seed):
    torch.manual_seed(seed)
    model, ckpt = load_phase3_pi_fno_model(
        "models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
    from phase_5_corrected.decoder_v2 import build_decoder_v2
    decoder = build_decoder_v2(latent_dim=32, hidden_dim=64, n_freqs=16, T=10.0, seed=seed).to(device)
    model = SCFNOModelV2(model.fno, decoder, pad_size=0).to(device)
    return model

def fixed_lambda(epoch, init=1e-4, target=1.0, ramp=100):
    if epoch >= ramp: return target
    return init * (10.0 ** (epoch / ramp * np.log10(target / init)))

def train_seed(seed):
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Xva, Yva = load_data()
    model = build_model(seed)
    kin, th = get_kin(), get_th()
    loss_fn = load_composite_loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=0.0)
    TQ = torch.linspace(0.5, 9.5, 50, device=device)
    out_dir = Path(f"results/phase_5/step_5_B/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("models/sc_fno/phase_5/step_5_B")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history, lam_hist = [], []
    best_val, best_ep, since_best = float("inf"), 0, 0
    peak_vram = 0.0
    stop_ep, stop_reason = None, None
    t0 = time.time()

    for epoch in range(300):
        model.train()
        lam = fixed_lambda(epoch)
        perm = torch.randperm(Xtr.shape[0])
        Xp, Yp = Xtr[perm], Ytr[perm]
        ep = {k:0.0 for k in ["L_data","L_phys","L_ic","L_data_decoder","L_total"]}
        nb = 0
        for bi in range(0, Xtr.shape[0], 16):
            b_end = min(bi+16, Xtr.shape[0])
            xb = Xp[bi:b_end].unsqueeze(1).to(device)
            yb = Yp[bi:b_end].to(device)
            B = xb.shape[0]
            tq = TQ.unsqueeze(0).expand(B, -1)
            norm_rho = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]["rho_ext"]
            rho_fn = make_rho_ext_fn(xb[:, 0, :], norm_rho["mean"], norm_rho["std"])
            opt.zero_grad()
            losses = compute_composite_loss_v2(
                model, xb, yb, tq, rho_fn, kin, th, loss_fn, lambda_phys=lam)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in ep: ep[k] += float(losses[k].detach())
            nb += 1
        sched.step()
        for k in ep: ep[k] /= max(nb,1)

        # Validation data loss (projection MLP path)
        model.eval()
        vloss = 0.0; nv = 0
        with torch.no_grad():
            for vi in range(0, Xva.shape[0], 16):
                ve = min(vi+16, Xva.shape[0])
                xb = Xva[vi:ve].unsqueeze(1).to(device)
                yb = Yva[vi:ve].to(device)
                pred = model.forward_data(xb)
                vloss += float(((pred - yb)**2).mean())
                nv += 1
        vloss /= max(nv,1)

        vram = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
        peak_vram = max(peak_vram, vram)
        row = {"epoch":epoch, "lambda_phys":lam, "val_data_loss":vloss,
               "peak_vram_MB":vram, **{k:ep[k] for k in ep}}
        history.append(row); lam_hist.append({"epoch":epoch,"lambda":lam})

        if epoch % 10 == 0:
            print(f"  ep{epoch:3d}: data={ep['L_data']:.5f} phys={ep['L_phys']:.4f} "
                  f"dec={ep['L_data_decoder']:.5f} lam={lam:.4f} val={vloss:.5f} vram={vram:.0f}MB")

        if vloss < best_val:
            best_val = vloss; best_ep = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss":vloss, "epoch":epoch, "seed":seed},
                       ckpt_dir / f"seed_{seed}_best.pt")

        if epoch >= 100:
            since_best = since_best+1 if vloss >= best_val - 1e-6 else 0
            if since_best >= 100:
                stop_ep, stop_reason = epoch, "patience=100 after epoch 100"
                print(f"  EARLY STOP: {stop_reason}")
                break

    wall = time.time() - t0
    torch.save({"model_state_dict": model.state_dict(), "val_loss":best_val,
                "epoch":stop_ep or 300, "seed":seed},
               ckpt_dir / f"seed_{seed}_final.pt")

    # Post-training decoder health: du_dt std on val data
    print("  Post-training decoder health check...")
    model.eval()
    with torch.enable_grad():
        xb = Xva[:16].unsqueeze(1).to(device)
        tq = TQ.unsqueeze(0).expand(16, -1)
        u_hat, du_dt = compute_du_dt_per_channel(model, xb, tq)
        du_dt_std = float(du_dt.std())
        u_hat_var = float(u_hat.var(dim=-1).mean())
    print(f"  du_dt std: {du_dt_std:.4f} | u_hat var: {u_hat_var:.4f}")

    pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame(lam_hist).to_csv(out_dir / "lambda_history.csv", index=False)

    summary = {
        "seed": seed, "best_val_data_loss": best_val, "best_epoch": best_ep,
        "final_epoch": stop_ep or 300, "stop_reason": stop_reason or "max_epochs",
        "final_lambda": lam, "wall_time_s": wall, "peak_vram_MB": peak_vram,
        "post_train_du_dt_std": du_dt_std, "post_train_u_hat_var": u_hat_var,
        "final_train_data_loss": history[-1]["L_data"],
        "final_train_phys_loss": history[-1]["L_phys"],
        "final_train_dec_loss": history[-1]["L_data_decoder"],
        "final_val_data_loss": history[-1]["val_data_loss"],
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\n  SEED {seed} COMPLETE: best_val={best_val:.6f} @ ep{best_ep}, "
          f"du_dt_std={du_dt_std:.3f}, wall={wall/60:.1f}min, vram={peak_vram:.0f}MB")
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    train_seed(args.seed)
