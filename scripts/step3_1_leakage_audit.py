"""Step 3.1.F — Verify that rho_feedback uses PREDICTED Tf, Tc (not ground truth).

If feedback read ground truth, changing the predicted Tf in u_hat would NOT
change rho_fb. We verify it does.
"""
import sys
sys.path.insert(0, "src")
import dataclasses, json, numpy as np, torch
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.residuals import compute_residuals

def _kin():
    fields = {f.name for f in dataclasses.fields(KineticsParameters)}
    kw = {"beta_eff": 0.006502, "Lambda": 1e-4,
          "beta_i": np.array([0.000215,0.001424,0.001274,0.002568,0.000748,0.000273]),
          "lambda_i": np.array([0.0124,0.0305,0.1110,0.3010,1.1400,3.0100])}
    return KineticsParameters(**{k:v for k,v in kw.items() if k in fields})

def _th():
    fields = {f.name for f in dataclasses.fields(THParameters)}
    kw = dict(M_f=27000., c_p_f=300., f_f=0.97, M_c=4000., c_p_c=5500.,
              m_dot=13636.36, T_in=563., U=3060., A=3000., P0=3e9,
              alpha_f=-3e-5, alpha_c=-5e-4, T_f_ref=900., T_c_ref=583.)
    return THParameters(**{k:v for k,v in kw.items() if k in fields})

kin, th = _kin(), _th()

B, N = 1, 20
t = torch.linspace(0, 1, N).unsqueeze(0)
n = torch.ones(B, N)
C = torch.zeros(B, 6, N)
for i in range(6):
    C[:, i, :] = kin.beta_i[i] / (kin.Lambda * kin.lambda_i[i])

Tf_lo = torch.full((B, N), th.T_f_ref)
Tf_hi = torch.full((B, N), th.T_f_ref + 100.0)
Tc = torch.full((B, N), th.T_c_ref)

u_lo = torch.cat([n.unsqueeze(1), C, Tf_lo.unsqueeze(1), Tc.unsqueeze(1)], dim=1)
u_hi = torch.cat([n.unsqueeze(1), C, Tf_hi.unsqueeze(1), Tc.unsqueeze(1)], dim=1)
z = torch.zeros_like(u_lo)

o_lo = compute_residuals(u_lo, z, t, lambda tq: torch.zeros_like(tq), kin, th)
o_hi = compute_residuals(u_hi, z, t, lambda tq: torch.zeros_like(tq), kin, th)

delta_fb = float((o_hi.rho_fb - o_lo.rho_fb).abs().mean())
delta_Rn = float((o_hi.R_n - o_lo.R_n).abs().mean())
delta_RTf = float((o_hi.R_Tf - o_lo.R_Tf).abs().mean())

# Expected: rho_fb changes by alpha_f * 100 = -3e-5 * 100 = -3e-3
expected_delta_fb = abs(th.alpha_f * 100.0)

checks = {
    "feedback_responds_to_predicted_Tf": delta_fb > 1e-6,
    "R_n_responds_to_predicted_Tf":      delta_Rn > 1e-6,
    "R_Tf_responds_to_predicted_Tf":     delta_RTf > 1e-6,
    "feedback_delta_matches_alpha_f_100K":
        abs(delta_fb - expected_delta_fb) / expected_delta_fb < 1e-3,
}
all_pass = all(checks.values())

out = {
    "test": "no ground-truth leakage audit",
    "setup": "Two residuals computed with identical n, C, Tc; Tf differs by 100 K",
    "delta_rho_fb": delta_fb,
    "delta_R_n":   delta_Rn,
    "delta_R_Tf":  delta_RTf,
    "expected_delta_rho_fb": expected_delta_fb,
    "checks": checks,
    "all_pass": all_pass,
    "interpretation": (
        "If feedback used ground truth, rho_fb would NOT change when predicted "
        "Tf is offset. Confirmed rho_fb responds to predicted Tf -> no leakage."
    ),
}
with open("results/phase_3/residuals/no_leakage_verification.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\n=== No-leakage audit:", "PASS" if all_pass else "FAIL", "===")
