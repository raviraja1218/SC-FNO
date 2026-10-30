# Step 1.5 Block B — Independent Checksum Verification Report

## Purpose
Independently reproduce the aggregate SHA-256 recorded in Step 1.4 Block F,
addressing the reviewer's concern that the aggregate checksum was only
computed by its original generation script and not independently verified.

## 1. Aggregate SHA-256

**Algorithm (reimplemented from scratch):**
1. Read `dataset_checksums.csv` (2,000 rows)
2. Sort rows by `scenario_id` (lexicographic)
3. Build string per row: `scenario_id|family|split|trajectory_sha256|forcing_sha256`
4. Join rows with `\n`
5. SHA-256 of UTF-8 encoded text

| | Value |
|---|---|
| Recorded (Step 1.4 Block F) | `9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6` |
| Independently recomputed | `9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6` |
| Match | **PASS** |

## 2. Split file checksums

| File | Match |
|------|-------|
| train_ids.csv | PASS |
| validation_ids.csv | PASS |
| test_ids.csv | PASS |

Overall: **PASS** (3/3)

## 3. Sample of 50 random per-scenario hashes

Seed: `verify_v1` (deterministic)

| Field | Matched |
|-------|---------|
| Trajectory SHA-256 | 50/50 |
| Forcing SHA-256 | 50/50 |
| Time SHA-256 | 50/50 |

Overall: **PASS**

## 4. Structure of aggregate_checksums.json

- Required fields present: PASS
- Counts (2000/1600/200/200): PASS
- 11 split file checksums present: PASS

## Conclusion

**All checks PASS.**
The aggregate SHA-256 `9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6` is independently reproducible from the
2,000 per-scenario hashes recorded in `dataset_checksums.csv`, using the
documented algorithm. The reviewer's outstanding concern is resolved.

## Wall time
0.01 s
