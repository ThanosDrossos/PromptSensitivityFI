# Part-B analyses — three-axes defense (generated)

## B2 counterexamples (all invariant checks passed)
```json
{
 "ex1_rho_f_gap": true,
 "ex1_accuracy_pinned": true,
 "ex2_hsem_gap": true,
 "ex3_accuracy_gap": true
}
```

Ex1: accuracy pinned at 0.5, rho_F 0.00 vs 1.00 (the FI_in curves differ — the distribution information rho_F reads and the accuracy scalar discards). Ex2: accuracy pinned, normalized H_sem 1.00 vs 2.00. Ex3: H_sem pinned (single mode), accuracy 1.0 vs 0.0.

## B3a factor structure
- top-3 eigenvalues [5.772, 2.444, 1.595] explain **70%** of the metric-space variance
- varimax factor assignment (0/1/2 = the three axes):
```json
{
 "accuracy": 1,
 "AUFI (graded)": 1,
 "rho_F  [M1]": 2,
 "FI premium  [M2]": 1,
 "spread (Cao)": 2,
 "H_sem": 0,
 "FI_out_fixed": 0,
 "Var[FI_out]  [M4]": 0,
 "TVD-sens  [M4]": 0,
 "S_tau (Errica)": 0,
 "variation ratio": 0,
 "|A_q| observed": 0,
 "rho_u (Cox)": 2,
 "ESS_in": 2
}
```

## B3b octant occupancy (per model x level)

     model_key  spec_level  octants_populated
  llama_3_1_8b           0                  8
  llama_3_1_8b           1                  8
mistral_7b_v03           0                  8
mistral_7b_v03           1                  8
   qwen_2_5_7b           0                  8
   qwen_2_5_7b           1                  8

Full table with example questions: data/b3_octants.parquet (506 questions across cells).

## B4 rho_F bootstrap CIs
- 506 cells with CIs; median 95% width 0.18; defined-resample floor 1226/2000