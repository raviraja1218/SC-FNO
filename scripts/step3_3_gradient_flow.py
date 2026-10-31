"""Step 3.3.E — Per-loss gradient norms + conflict cos on shared params."""
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, "src")
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.losses import load_composite_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device).train()
loss_fn = load_composite_loss()

kin = KineticsParameters(beta_eff=0.006502, Lambda=1e-4,
    beta_i=[0.000215,0.001424,0.001274,0.002568,0.000748,0.000273],
    lambda_i=[0.0124,0.0305,0.1110,0.3010,1.1400,3.0100])
th = THParameters(M_f=27000., c_p_f=300., f_f=0.97, M_c=4000., c_p_c=5500.,
    m_dot=13636.36, T_in=563., U=3060., A=3000., P0=3e9,
    alpha_f=-3e-5, alpha_c=-5e-4, T_f_ref=900., T_c_ref=583.)

B, Nq = 4, 20
rho = torch.randn(B, 1, 200, device=device) * 1e-3
u_gt = torch.randn(B, 9, 200, device=device) * 0.1
t_q = torch.linspace(0.5, 9.5, Nq, device=device).unsqueeze(0).expand(B, -1).contiguous()
def rho_fn(tq): return torch.zeros_like(tq)

def shared_params():
    """List of shared parameter tensors (lifting + fno_blocks)."""
    ps = []
    for p in model.fno.lifting.parameters():
        ps.append(p)
    for p in model.fno.fno_blocks.parameters():
        ps.append(p)
    return ps

def grad_vector(loss_fn_call):
    """Compute loss, backward, return flat grad vector over shared params."""
    model.zero_grad()
    loss = loss_fn_call()
    loss.backward(retain_graph=True)
    vecs = []
    for p in shared_params():
        if p.grad is not None:
            vecs.append(p.grad.detach().flatten())
        else:
            vecs.append(torch.zeros(p.numel(), device=p.device))
    return loss.detach(), torch.cat(vecs)

# Per-loss gradient norms on shared params
g_data, vec_data = grad_vector(lambda: loss_fn.compute_data_loss(model, rho, u_gt))
g_phys, vec_phys = grad_vector(lambda: loss_fn.compute_physics_loss(model, rho, t_q, rho_fn, kin, th))
g_ic,   vec_ic   = grad_vector(lambda: loss_fn.compute_ic_loss(model, rho, u_gt[:,:,0]))

def cos(a, b):
    """Cosine similarity — normalize in FP64 to avoid FP32 overflow on large vectors."""
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na == 0.0 or nb == 0.0:
        return 0.0
    # Normalize each to unit length first, then sum of elementwise products
    # (equivalent to dot/(na*nb), but avoids intermediate overflow)
    a_hat = (a / na).to(torch.float64)
    b_hat = (b / nb).to(torch.float64)
    return float(torch.sum(a_hat * b_hat).item())

out = {
    "batch_size": B, "n_query": Nq,
    "shared_params": "lifting + fno_blocks (36 tensors)",
    "loss_values": {
        "data":    float(g_data.item()),
        "physics": float(g_phys.item()),
        "ic":      float(g_ic.item()),
    },
    "shared_grad_norms": {
        "data":    float(vec_data.norm().item()),
        "physics": float(vec_phys.norm().item()),
        "ic":      float(vec_ic.norm().item()),
    },
    "cos_shared": {
        "data_vs_physics": cos(vec_data, vec_phys),
        "data_vs_ic":      cos(vec_data, vec_ic),
        "physics_vs_ic":   cos(vec_phys, vec_ic),
    },
    "note": (
        "Informational only. cos>0 = reinforce, cos<0 = conflict. "
        "Recorded for Step 3.4 weight tuning — NOT a pass/fail criterion."
    ),
}
Path("results/phase_3/loss_integration/gradient_norms_sanity_batch.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
