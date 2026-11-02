"""Phase 5 Corrected Architecture — SCFNOModelV2.

Fresh implementation that fixes Bug J (decoder temporal collapse) via
supervised decoder data loss. Does NOT modify Phase 3/4 code.

Key fixes:
  - Bug B: true external forcing at query times (not zero)
  - Bug E: per-channel derivative (not per-sum)
  - Bug J: L_data_decoder anchors decoder output to ground truth

Layout:
  - decoder_v2.py       : temporal-attention decoder (Phase 3 arch, new training)
  - model_v2.py         : SCFNOModelV2 wrapper
  - losses_v2.py        : CompositeLossV2 with L_data_decoder
  - derivatives_v2.py   : compute_du_dt_per_channel
  - residuals_v2.py     : 9 residuals with true forcing
  - trainer_v2.py       : training loop with decoder supervision
  - verification/       : health checks before any real training
"""
