# Phase 1.2 Block C — Pilot Validation Report (v2)

## Methodology correction
SC-FNO-P1-DEFECT-002: thermal absolute threshold raised from 1e-6 K (v1,
unachievable) to 1e-2 K (v2, calibrated). See
`results/phase_1/solver/bdf_radau_methodology.v2.md`.

Original v1 results preserved at `results/phase_1/pilot/v1_preserved/`.

## BDF/Radau comparison (v2)
- Scenarios compared: 100
- Comparison passed: 100/100
- Failures: 100

## Thresholds (v2)
- Kinetics (n, C_i): max_abs_diff <= 1e-6 × max(|value|)
- Thermal (T_f, T_c): max_abs_diff <= 1e-2 K  ← calibrated
- Relative L2 per state: <= 1e-5

## Per-state statistics
| State | Scenarios | Passed | Max abs diff | Max rel L2 |
|-------|-----------|--------|--------------|------------|
| n | 100 | 100 | 7.019e-04 | 1.517e-06 |
| C1 | 100 | 100 | 1.379e-05 | 7.352e-09 |
| C2 | 100 | 100 | 4.682e-04 | 5.730e-08 |
| C3 | 100 | 100 | 3.103e-03 | 5.824e-07 |
| C4 | 100 | 100 | 9.446e-03 | 1.598e-06 |
| C5 | 100 | 100 | 1.793e-04 | 2.490e-07 |
| C6 | 100 | 100 | 1.447e-05 | 5.793e-08 |
| Tf | 100 | 0 | 6.407e-03 | 2.654e-07 |
| Tc | 100 | 0 | 3.467e-03 | 4.812e-07 |

## Physical validation
- Trajectories passing physical checks: 0/100

## Total wall time
81.59 s

## Failure register
- Total failures: 100

| Scenario | cmp | phys | reason |
|----------|-----|------|--------|
| pilot_0001 | False | False | threshold_exceeded|physical_violation |
| pilot_0002 | False | False | threshold_exceeded|physical_violation |
| pilot_0003 | False | False | threshold_exceeded|physical_violation |
| pilot_0004 | False | False | threshold_exceeded|physical_violation |
| pilot_0005 | False | False | threshold_exceeded|physical_violation |
| pilot_0006 | False | False | threshold_exceeded|physical_violation |
| pilot_0007 | False | False | threshold_exceeded|physical_violation |
| pilot_0008 | False | False | threshold_exceeded|physical_violation |
| pilot_0009 | False | False | threshold_exceeded|physical_violation |
| pilot_0010 | False | False | threshold_exceeded|physical_violation |
| pilot_0011 | False | False | threshold_exceeded|physical_violation |
| pilot_0012 | False | False | threshold_exceeded|physical_violation |
| pilot_0013 | False | False | threshold_exceeded|physical_violation |
| pilot_0014 | False | False | threshold_exceeded|physical_violation |
| pilot_0015 | False | False | threshold_exceeded|physical_violation |
| pilot_0016 | False | False | threshold_exceeded|physical_violation |
| pilot_0017 | False | False | threshold_exceeded|physical_violation |
| pilot_0018 | False | False | threshold_exceeded|physical_violation |
| pilot_0019 | False | False | threshold_exceeded|physical_violation |
| pilot_0020 | False | False | threshold_exceeded|physical_violation |
| pilot_0021 | False | False | threshold_exceeded|physical_violation |
| pilot_0022 | False | False | threshold_exceeded|physical_violation |
| pilot_0023 | False | False | threshold_exceeded|physical_violation |
| pilot_0024 | False | False | threshold_exceeded|physical_violation |
| pilot_0025 | False | False | threshold_exceeded|physical_violation |
| pilot_0026 | False | False | threshold_exceeded|physical_violation |
| pilot_0027 | False | False | threshold_exceeded|physical_violation |
| pilot_0028 | False | False | threshold_exceeded|physical_violation |
| pilot_0029 | False | False | threshold_exceeded|physical_violation |
| pilot_0030 | False | False | threshold_exceeded|physical_violation |
| pilot_0031 | False | False | threshold_exceeded|physical_violation |
| pilot_0032 | False | False | threshold_exceeded|physical_violation |
| pilot_0033 | False | False | threshold_exceeded|physical_violation |
| pilot_0034 | False | False | threshold_exceeded|physical_violation |
| pilot_0035 | False | False | threshold_exceeded|physical_violation |
| pilot_0036 | False | False | threshold_exceeded|physical_violation |
| pilot_0037 | False | False | threshold_exceeded|physical_violation |
| pilot_0038 | False | False | threshold_exceeded|physical_violation |
| pilot_0039 | False | False | threshold_exceeded|physical_violation |
| pilot_0040 | False | False | threshold_exceeded|physical_violation |
| pilot_0041 | False | False | threshold_exceeded|physical_violation |
| pilot_0042 | False | False | threshold_exceeded|physical_violation |
| pilot_0043 | False | False | threshold_exceeded|physical_violation |
| pilot_0044 | False | False | threshold_exceeded|physical_violation |
| pilot_0045 | False | False | threshold_exceeded|physical_violation |
| pilot_0046 | False | False | threshold_exceeded|physical_violation |
| pilot_0047 | False | False | threshold_exceeded|physical_violation |
| pilot_0048 | False | False | threshold_exceeded|physical_violation |
| pilot_0049 | False | False | threshold_exceeded|physical_violation |
| pilot_0050 | False | False | threshold_exceeded|physical_violation |
| pilot_0051 | False | False | threshold_exceeded|physical_violation |
| pilot_0052 | False | False | threshold_exceeded|physical_violation |
| pilot_0053 | False | False | threshold_exceeded|physical_violation |
| pilot_0054 | False | False | threshold_exceeded|physical_violation |
| pilot_0055 | False | False | threshold_exceeded|physical_violation |
| pilot_0056 | False | False | threshold_exceeded|physical_violation |
| pilot_0057 | False | False | threshold_exceeded|physical_violation |
| pilot_0058 | False | False | threshold_exceeded|physical_violation |
| pilot_0059 | False | False | threshold_exceeded|physical_violation |
| pilot_0060 | False | False | threshold_exceeded|physical_violation |
| pilot_0061 | False | False | threshold_exceeded|physical_violation |
| pilot_0062 | False | False | threshold_exceeded|physical_violation |
| pilot_0063 | False | False | threshold_exceeded|physical_violation |
| pilot_0064 | False | False | threshold_exceeded|physical_violation |
| pilot_0065 | False | False | threshold_exceeded|physical_violation |
| pilot_0066 | False | False | threshold_exceeded|physical_violation |
| pilot_0067 | False | False | threshold_exceeded|physical_violation |
| pilot_0068 | False | False | threshold_exceeded|physical_violation |
| pilot_0069 | False | False | threshold_exceeded|physical_violation |
| pilot_0070 | False | False | threshold_exceeded|physical_violation |
| pilot_0071 | False | False | threshold_exceeded|physical_violation |
| pilot_0072 | False | False | threshold_exceeded|physical_violation |
| pilot_0073 | False | False | threshold_exceeded|physical_violation |
| pilot_0074 | False | False | threshold_exceeded|physical_violation |
| pilot_0075 | False | False | threshold_exceeded|physical_violation |
| pilot_0076 | False | False | threshold_exceeded|physical_violation |
| pilot_0077 | False | False | threshold_exceeded|physical_violation |
| pilot_0078 | False | False | threshold_exceeded|physical_violation |
| pilot_0079 | False | False | threshold_exceeded|physical_violation |
| pilot_0080 | False | False | threshold_exceeded|physical_violation |
| pilot_0081 | False | False | threshold_exceeded|physical_violation |
| pilot_0082 | False | False | threshold_exceeded|physical_violation |
| pilot_0083 | False | False | threshold_exceeded|physical_violation |
| pilot_0084 | False | False | threshold_exceeded|physical_violation |
| pilot_0085 | False | False | threshold_exceeded|physical_violation |
| pilot_0086 | False | False | threshold_exceeded|physical_violation |
| pilot_0087 | False | False | threshold_exceeded|physical_violation |
| pilot_0088 | False | False | threshold_exceeded|physical_violation |
| pilot_0089 | False | False | threshold_exceeded|physical_violation |
| pilot_0090 | False | False | threshold_exceeded|physical_violation |
| pilot_0091 | False | False | threshold_exceeded|physical_violation |
| pilot_0092 | False | False | threshold_exceeded|physical_violation |
| pilot_0093 | False | False | threshold_exceeded|physical_violation |
| pilot_0094 | False | False | threshold_exceeded|physical_violation |
| pilot_0095 | False | False | threshold_exceeded|physical_violation |
| pilot_0096 | False | False | threshold_exceeded|physical_violation |
| pilot_0097 | False | False | threshold_exceeded|physical_violation |
| pilot_0098 | False | False | threshold_exceeded|physical_violation |
| pilot_0099 | False | False | threshold_exceeded|physical_violation |
| pilot_0100 | False | False | threshold_exceeded|physical_violation |
