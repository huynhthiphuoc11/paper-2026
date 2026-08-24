"""
Post-Fix Stage 1 Rerun: Option A (drop lf_exp) + lf_loc remote fix
====================================================================
Reports EXACTLY 3 tables:
  Table 1 — Joint Coverage with 4-LF set
  Table 2 — Pairwise Conflict Rate (4 LFs)
  Table 3 — Cramér's V (4 LFs)
Plus: lf_loc fix impact quantification BEFORE declaring fix complete.
Plus: Fleiss' κ with dual-scenario interpretation.

Does NOT shortcut any verification step.
"""

import io, os, sys, json, warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.data import load_dataset, CVJobDatasetLoader
from src.weak import AspectLabelingFunctions

REPORT_DIR = "reports"
SEP = "-" * 68

# After Option A: lf_exp removed from the active LF set
LF_COLS_4 = ["lf_skill", "lf_sem", "lf_role", "lf_loc"]

REMOTE_KEYWORDS = [
    "remote", "work from home", "toan quoc", "to\u00e0n qu\u1ed1c",
    "online", "t\u1eeb xa", "tu xa", "nationwide", "anywhere",
    "l\u00e0m vi\u1ec7c t\u1eeb xa", "lam viec tu xa", "kh\u00f4ng y\u00eau c\u1ea7u v\u0103n ph\u00f2ng",
]


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# Step 0 — Load data and produce LF matrix
# ══════════════════════════════════════════════════════════════
def load_lf_matrix():
    df = load_dataset(data_dir="data", random_seed=42)
    loader = CVJobDatasetLoader(random_seed=42)
    df_train, _, _ = loader.get_job_disjoint_splits(df)

    lf = AspectLabelingFunctions(pos_percentile=75, neg_percentile=25)
    lf.fit(df_train)
    df_lf = lf.transform(df_train)
    return df_train, df_lf


# ══════════════════════════════════════════════════════════════
# Step 1 — Quantify lf_loc fix impact BEFORE proceeding
# ══════════════════════════════════════════════════════════════
def quantify_lfloc_fix(df_train: pd.DataFrame, df_lf: pd.DataFrame) -> dict:
    section("STEP 1 — lf_loc Fix Impact: Quantify Before Declaring Fixed")

    # Before fix: lf_loc as currently computed (already in df_lf after lf_definitions transform)
    lf_loc_before = df_lf["lf_loc"].values.copy()

    # Detect remote/ambiguous via job_title keyword check
    if "job_title" not in df_train.columns:
        print("  [WARNING] 'job_title' column not found — remote keyword fix cannot run.")
        print("            lf_loc remains pure binary. No ABSTAIN zone added.")
        n_remote = 0
        remote_mask = pd.Series(False, index=df_train.index)
    else:
        title_lower = df_train["job_title"].fillna("").str.lower()
        remote_mask = title_lower.str.contains("|".join(REMOTE_KEYWORDS), regex=True, na=False)
        n_remote = int(remote_mask.sum())

    N = len(df_train)
    print(f"\n  Remote keyword detection on 'job_title' column:")
    print(f"    Total pairs          : {N:>6,}")
    print(f"    Remote/ambiguous jobs: {n_remote:>6,}  ({n_remote/N:.1%} of all pairs)")

    # Show sample of detected remote job titles (up to 10)
    if n_remote > 0 and "job_title" in df_train.columns:
        remote_titles = df_train.loc[remote_mask, "job_title"].unique()[:10]
        print(f"\n  Sample remote job titles detected:")
        for t in remote_titles:
            print(f"    - {t}")
    else:
        print("\n  No remote jobs detected via keyword search.")

    # Before distribution
    n_pos_b  = int((lf_loc_before ==  1).sum())
    n_neg_b  = int((lf_loc_before == -1).sum())
    n_abs_b  = int((lf_loc_before ==  0).sum())
    print(f"\n  lf_loc BEFORE fix:")
    print(f"    +1 (match)    : {n_pos_b:>6,}  ({n_pos_b/N:.1%})")
    print(f"     0 (ABSTAIN)  : {n_abs_b:>6,}  ({n_abs_b/N:.1%})")
    print(f"    -1 (no match) : {n_neg_b:>6,}  ({n_neg_b/N:.1%})")

    # Apply fix
    lf_loc_after = lf_loc_before.copy()
    if n_remote > 0:
        remote_idx = df_train.index[remote_mask]
        # Map df_train index to positional index in df_lf
        pos_idx = [df_train.index.get_loc(i) for i in remote_idx if i in df_train.index]
        lf_loc_after[pos_idx] = 0  # force ABSTAIN

    # After distribution
    n_pos_a  = int((lf_loc_after ==  1).sum())
    n_neg_a  = int((lf_loc_after == -1).sum())
    n_abs_a  = int((lf_loc_after ==  0).sum())
    print(f"\n  lf_loc AFTER fix:")
    print(f"    +1 (match)    : {n_pos_a:>6,}  ({n_pos_a/N:.1%})")
    print(f"     0 (ABSTAIN)  : {n_abs_a:>6,}  ({n_abs_a/N:.1%})")
    print(f"    -1 (no match) : {n_neg_a:>6,}  ({n_neg_a/N:.1%})")

    delta_abs = n_abs_a - n_abs_b
    delta_neg = n_neg_a - n_neg_b
    print(f"\n  Change: ABSTAIN +{delta_abs} pairs, Negative {delta_neg:+d} pairs")

    # Verdict on fix effectiveness
    if n_remote == 0:
        verdict = ("INEFFECTIVE: No remote jobs found via keywords. "
                   "lf_loc remains 100% coverage with 74% negative. "
                   "The 74% negative rate appears to reflect genuine geographic diversity "
                   "in the dataset (dataset has wide location spread), not a parse artifact. "
                   "Document this as a dataset characteristic, not a LF design flaw.")
        structural_issue_remains = True
    elif n_abs_a / N < 0.05:
        verdict = (f"MINOR IMPACT: Only {n_abs_a/N:.1%} of pairs moved to ABSTAIN. "
                   "lf_loc is still near-100% coverage and near-74% negative. "
                   "The fix is correct in principle but insufficient to change the structural "
                   "characteristic of this LF. Accept this and document.")
        structural_issue_remains = True
    else:
        verdict = (f"MEANINGFUL IMPACT: {n_abs_a/N:.1%} ABSTAIN zone created. "
                   "lf_loc now has a non-trivial ABSTAIN region. Fix is effective.")
        structural_issue_remains = False

    print(f"\n  VERDICT: {verdict}")

    # Persist fix into df_lf
    df_lf["lf_loc"] = lf_loc_after

    return {
        "n_remote_detected"    : n_remote,
        "pct_remote"           : round(n_remote / N, 4),
        "lf_loc_before"        : {"pos": n_pos_b, "neg": n_neg_b, "abs": n_abs_b},
        "lf_loc_after"         : {"pos": n_pos_a, "neg": n_neg_a, "abs": n_abs_a},
        "fix_effective"        : not structural_issue_remains,
        "structural_issue_remains": structural_issue_remains,
        "verdict"              : verdict,
    }


# ══════════════════════════════════════════════════════════════
# Step 2 — Drop lf_exp: confirm Option A
# ══════════════════════════════════════════════════════════════
def apply_option_a(df_lf: pd.DataFrame) -> pd.DataFrame:
    section("STEP 2 — Option A: Remove lf_exp from Active LF Set")

    before = df_lf[["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]]
    print(f"\n  Before: 5 LFs = {list(before.columns)}")
    print(f"  Removing lf_exp (data artifact: 93.6% pos_rate due to exp_score mass-point at 1.0)")
    print(f"  Reason: lf_exp cannot be fixed by threshold tuning — it is a feature design issue.")
    print(f"  exp_score will be used as a CONTINUOUS FEATURE in downstream RankNet instead.")
    print(f"\n  After: 4 LFs = {LF_COLS_4}")
    print(f"  NOTE: K=4 >= 3 (minimum identifiability bound under CI assumptions) — valid.")
    return df_lf


# ══════════════════════════════════════════════════════════════
# TABLE 1 — Joint Coverage with 4-LF set
# ══════════════════════════════════════════════════════════════
def table1_joint_coverage(df_lf: pd.DataFrame) -> dict:
    section("TABLE 1 — Joint Coverage (4 LFs: skill, sem, role, loc)")

    active_count = (df_lf[LF_COLS_4] != 0).sum(axis=1)
    N = len(df_lf)

    dist = active_count.value_counts(normalize=True).sort_index()

    interpretations = {
        0: "NO supervision — excluded from Label Model input",
        1: "SINGLE-SOURCE ONLY — degenerates to copying that one LF",
        2: "Minimal multi-source (thin but valid)",
        3: "Moderate multi-source (good for EM)",
        4: "Full coverage (all 4 LFs active)",
    }

    print(f"\n  {'n_active':>10}  {'fraction':>10}  {'count':>8}  interpretation")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*40}")

    result_rows = []
    for n in range(5):
        frac  = float(dist.get(n, 0.0))
        count = int(round(frac * N))
        interp = interpretations.get(n, "")
        flag = " <-- PROBLEM" if n <= 1 and frac > 0.05 else ""
        print(f"  {n:>10}  {frac:>10.4f}  {count:>8,}  {interp}{flag}")
        result_rows.append({"n_active": n, "fraction": round(frac, 4),
                             "count": count, "note": interp})

    frac_0  = float(dist.get(0, 0.0))
    frac_1  = float(dist.get(1, 0.0))
    frac_ge2 = 1.0 - frac_0 - frac_1
    frac_ge3 = float(dist.get(3, 0.0)) + float(dist.get(4, 0.0))

    print(f"\n  Summary:")
    print(f"    Pairs with 0 active LFs : {frac_0:.2%}  {'[OK]' if frac_0 < 0.01 else '[WARNING]'}")
    print(f"    Pairs with 1 active LF  : {frac_1:.2%}  {'[OK]' if frac_1 < 0.05 else '[WARNING: single-source]'}")
    print(f"    Pairs with >=2 active   : {frac_ge2:.2%}  (genuine multi-source)")
    print(f"    Pairs with >=3 active   : {frac_ge3:.2%}  (strong multi-source)")

    # Which LF is the sole active LF when n_active == 1?
    single_mask = (active_count == 1)
    if single_mask.sum() > 0:
        print(f"\n  When n_active == 1, sole LF breakdown:")
        for col in LF_COLS_4:
            sole_count = int(
                (((df_lf[LF_COLS_4] != 0).sum(axis=1) == 1) & (df_lf[col] != 0)).sum()
            )
            if sole_count > 0:
                print(f"    {col:<12}: {sole_count:>5,} pairs ({sole_count/N:.1%}) "
                      f"— these pairs have only {col} deciding label")
    else:
        print(f"\n  No pairs with n_active == 1. [GOOD]")

    # Decision on 0/1 active pairs
    problem_frac = frac_0 + frac_1
    print(f"\n  ACTION NEEDED ({problem_frac:.1%} pairs with 0-1 active LFs):")
    if problem_frac < 0.05:
        print("    -> Small fraction. OK to keep in Label Model — note in Limitations.")
    elif problem_frac < 0.20:
        print("    -> Moderate fraction. Recommend EXCLUDING from Label Model train set.")
        print("       These pairs can still be labelled by the selected model's predict().")
    else:
        print("    -> Large fraction. Strongly recommend exclusion from Label Model.")
        print("       Reconsider LF coverage design.")

    return {
        "distribution"    : result_rows,
        "pct_0_active"    : round(frac_0, 4),
        "pct_1_active"    : round(frac_1, 4),
        "pct_ge2_active"  : round(frac_ge2, 4),
        "pct_ge3_active"  : round(frac_ge3, 4),
        "problem_pct"     : round(frac_0 + frac_1, 4),
    }


# ══════════════════════════════════════════════════════════════
# TABLE 2 — Pairwise Conflict Rate (4 LFs)
# ══════════════════════════════════════════════════════════════
def table2_conflict_rate(df_lf: pd.DataFrame) -> dict:
    section("TABLE 2 — Pairwise Conflict Rate (4 LFs)")

    print(f"\n  Conflict(Li, Lj) = P(Li != Lj | Li != 0 AND Lj != 0)\n")
    print(f"  Interpretation guide:")
    print(f"    ~50% conflict  = near-random (signals uncorrelated, may include noise)")
    print(f"    20-45% conflict = genuine divergence (signals see different things)")
    print(f"    <15% conflict  = high agreement (possible redundancy)")
    print()
    print(f"  {'Pair':<22}  {'n_both_active':>14}  {'conflict_rate':>14}  interpretation")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*30}")

    rows = []
    for c1, c2 in combinations(LF_COLS_4, 2):
        mask = (df_lf[c1] != 0) & (df_lf[c2] != 0)
        n_both = int(mask.sum())
        if n_both == 0:
            conflict = float("nan")
            interp = "No overlap — cannot assess"
        else:
            n_conflict = int(((df_lf[c1] != df_lf[c2]) & mask).sum())
            conflict = round(n_conflict / n_both, 4)
            if conflict > 0.45:
                interp = "Near-random / possible noise in one LF"
            elif conflict > 0.20:
                interp = "Genuine divergence (good for weak supervision)"
            elif conflict > 0.10:
                interp = "Some agreement but distinct"
            else:
                interp = "High agreement / possible redundancy"

        print(f"  {c1+'<->'+c2:<22}  {n_both:>14,}  {conflict:>14.4f}  {interp}")
        rows.append({"lf_a": c1, "lf_b": c2, "n_both_active": n_both,
                     "conflict_rate": conflict if not np.isnan(conflict) else None,
                     "interpretation": interp})

    return {"conflict_rates": rows}


# ══════════════════════════════════════════════════════════════
# Cramér's V helper
# ══════════════════════════════════════════════════════════════
def cramers_v_corrected(x: np.ndarray, y: np.ndarray) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n  = ct.values.sum()
    r, c = ct.shape
    phi2_corr = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(round(np.sqrt(phi2_corr / denom), 4)) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# TABLE 3 — Cramér's V (4 LFs)
# ══════════════════════════════════════════════════════════════
def table3_cramers_v(df_lf: pd.DataFrame) -> dict:
    section("TABLE 3 — Cramér's V Matrix (Bias-Corrected, 4 LFs)")

    print(f"\n  V = 0: statistically independent | V > 0.30: meaningful association")
    print(f"  V > 0.50: strong dependency (CI assumption in Label Model likely violated)\n")

    mat = {}
    for c1 in LF_COLS_4:
        mat[c1] = {}
        for c2 in LF_COLS_4:
            mat[c1][c2] = 1.0 if c1 == c2 else cramers_v_corrected(
                df_lf[c1].values, df_lf[c2].values
            )

    # Print matrix
    header = f"  {'':12}" + "".join(f"  {c:>10}" for c in LF_COLS_4)
    print(header)
    print(f"  {'-'*12}" + "".join(f"  {'-'*10}" for _ in LF_COLS_4))
    for c1 in LF_COLS_4:
        row_str = f"  {c1:<12}"
        for c2 in LF_COLS_4:
            v = mat[c1][c2]
            flag = " *" if c1 != c2 and v > 0.30 else "  "
            row_str += f"  {v:>9.3f}{flag}"
        print(row_str)

    print(f"\n  (* = V > 0.30 — potential dependency, review for Model D justification)\n")

    # Pairwise summary
    pairs = []
    for c1, c2 in combinations(LF_COLS_4, 2):
        v = mat[c1][c2]
        dep = "DEPENDENT" if v > 0.30 else ("BORDERLINE" if v > 0.15 else "INDEPENDENT")
        pairs.append({"pair": f"{c1}<->{c2}", "V": v, "assessment": dep})
        print(f"  {c1+'<->'+c2:<25}: V={v:.3f}  [{dep}]")

    # Model D justification update
    any_dependent = any(p["V"] > 0.30 for p in pairs if p["pair"].split("<->")[0] != p["pair"].split("<->")[1])
    print(f"\n  Model D (dependency-aware LM) justified: {any_dependent}")
    if not any_dependent:
        print("    -> All pairs V < 0.30. Model C (CI-assuming LM) is appropriate.")
        print("    -> Do NOT add dependency edges based on domain assumption alone.")

    return {"cramers_v": mat, "pairwise_summary": pairs,
            "model_d_justified": any_dependent}


# ══════════════════════════════════════════════════════════════
# Fleiss' κ with dual-scenario interpretation
# ══════════════════════════════════════════════════════════════
def compute_fleiss_kappa_4lf(df_lf: pd.DataFrame) -> dict:
    section("Fleiss' kappa (4 LFs) — Dual-Scenario Interpretation")

    print(f"\n  Recall from Stage 1 (5 LFs): kappa = -0.003 (near-zero)")
    print(f"  Root cause identified: lf_exp systematic bias toward +1")
    print(f"  Expected after removing lf_exp: kappa should change toward genuine signal\n")

    df_bin = df_lf[LF_COLS_4].replace({1: 1, -1: 0, 0: np.nan})
    n_active = df_bin.notna().sum(axis=1)
    df_bin = df_bin[n_active >= 2].copy()
    N = len(df_bin)
    K = len(LF_COLS_4)

    if N == 0:
        print("  ERROR: No pairs with >=2 active LFs.")
        return {"fleiss_kappa": None}

    df_filled = df_bin.apply(lambda row: row.fillna(row.mean()), axis=1)

    p_pos = df_filled.mean().mean()
    p_neg = 1.0 - p_pos

    n_pos_row = df_filled.sum(axis=1)
    n_neg_row = K - n_pos_row
    p_i = (n_pos_row * (n_pos_row - 1) + n_neg_row * (n_neg_row - 1)) / (K * (K - 1))
    p_bar = p_i.mean()
    p_e   = p_pos ** 2 + p_neg ** 2

    kappa = round(float((p_bar - p_e) / (1.0 - p_e + 1e-10)), 4)

    print(f"  Fleiss' kappa (4-LF): {kappa:.4f}")
    print(f"  p_bar (observed agreement): {p_bar:.4f}")
    print(f"  p_e   (expected by chance): {p_e:.4f}")
    print(f"  N pairs used (>=2 active LFs): {N:,}")

    print(f"\n  Dual-scenario interpretation:")
    if kappa < 0.10 and kappa > -0.10:
        print(f"  Scenario A: 'near-zero kappa due to genuinely complementary signals'")
        print(f"    -> 4 LFs diverge in a meaningful, non-systematic way.")
        print(f"    -> This is DESIRABLE for weak supervision: each LF sees a different aspect.")
        print(f"    -> Check Conflict Rate (Table 2) — if conflicts are 20-45%, this is scenario A.")
        print(f"  Scenario B: 'near-zero kappa because one or more LFs is still noisy'")
        print(f"    -> A remaining LF fires randomly, pushing kappa toward 0.")
        print(f"    -> Check which LF has conflict ~50% with ALL others (Table 2).")
        print(f"    -> If lf_loc still shows ~50% conflict with skill/sem/role -> Scenario B.")
    elif kappa >= 0.10:
        print(f"  Kappa > 0.10 — LFs show some positive agreement. Check for redundancy.")
        print(f"    -> If V < 0.15 for all pairs (Table 3), agreement is weak/chance-level.")
    else:
        print(f"  Kappa < -0.10 — systematic disagreement. Review LF design.")

    print(f"\n  REMINDER: kappa is DIAGNOSTIC ONLY. Model selection remains empirical (Stage 4).")

    return {
        "fleiss_kappa": kappa,
        "p_bar"       : round(float(p_bar), 4),
        "p_e"         : round(float(p_e), 4),
        "n_used"      : N,
    }


# ══════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════
def print_final_summary(loc_result, t1, t2, t3, kappa_result):
    section("FINAL SUMMARY — Post-Fix Stage 1 Audit")

    print()
    print("  [lf_loc fix]")
    n_abs_after = loc_result["lf_loc_after"]["abs"]
    N_total = sum(loc_result["lf_loc_after"].values())
    if loc_result["fix_effective"]:
        print(f"  EFFECTIVE: {n_abs_after/N_total:.1%} ABSTAIN zone created.")
    else:
        print(f"  LIMITED IMPACT: {n_abs_after/N_total:.1%} ABSTAIN (structural binary persists).")
        print(f"  -> Accept as dataset characteristic. Document in paper.")

    print()
    print("  [lf_exp removal — Option A]")
    print("  CONFIRMED: lf_exp removed from LF set.")
    print("  exp_score retained as continuous feature for downstream RankNet.")

    print()
    print("  [Table 1 — Joint Coverage (4 LFs)]")
    p0 = t1["pct_0_active"]
    p1 = t1["pct_1_active"]
    pge2 = t1["pct_ge2_active"]
    print(f"  0 active  : {p0:.2%}  {'[OK]' if p0 < 0.01 else '[WARN]'}")
    print(f"  1 active  : {p1:.2%}  {'[OK]' if p1 < 0.05 else '[WARN: single-source pairs]'}")
    print(f"  >=2 active: {pge2:.2%}  (multi-source supervision)")
    if p0 + p1 > 0.05:
        print(f"  -> EXCLUDE pairs with <=1 active LF from Label Model training.")

    print()
    print("  [Table 2 — Conflict Rate]")
    for row in t2["conflict_rates"]:
        cr = row.get("conflict_rate")
        cr_str = f"{cr:.4f}" if cr is not None else "nan"
        print(f"  {row['lf_a']+'<->'+row['lf_b']:<25}: {cr_str}  {row['interpretation']}")

    print()
    print("  [Table 3 — Cramér's V]")
    for p in t3["pairwise_summary"]:
        print(f"  {p['pair']:<25}: V={p['V']:.3f}  [{p['assessment']}]")
    print(f"  Model D (dependency-aware) justified: {t3['model_d_justified']}")

    print()
    print("  [Fleiss' kappa (4 LFs)]")
    print(f"  kappa = {kappa_result['fleiss_kappa']:.4f}")
    print(f"  -> Read kappa jointly with Conflict Rate to distinguish:")
    print(f"     'genuine divergence' (good) vs 'residual noise in one LF' (needs fix).")

    print()
    print("  NEXT STEPS (conditional on results above):")
    print("  IF pct_ge2 >= 80% AND no V > 0.30:")
    print("    -> Proceed to Gold-A + Gold-B annotation design (Stage 2)")
    print("  IF pct_ge2 < 80% OR any pair with V > 0.50:")
    print("    -> Reconsider LF coverage / add a new LF before annotation")
    print()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 68)
    print("  Post-Fix Stage 1 Rerun: Option A + lf_loc Fix Verification")
    print("=" * 68)

    # Load
    df_train, df_lf = load_lf_matrix()
    print(f"\n  Loaded {len(df_train):,} train pairs.")

    # Step 1: quantify lf_loc fix
    loc_result = quantify_lfloc_fix(df_train, df_lf)

    # Step 2: confirm Option A (lf_exp removed from active set)
    apply_option_a(df_lf)  # informational only — LF_COLS_4 already excludes lf_exp

    # Table 1
    t1 = table1_joint_coverage(df_lf)

    # Table 2
    t2 = table2_conflict_rate(df_lf)

    # Table 3
    t3 = table3_cramers_v(df_lf)

    # Fleiss' kappa
    kappa_result = compute_fleiss_kappa_4lf(df_lf)

    # Final summary
    print_final_summary(loc_result, t1, t2, t3, kappa_result)

    # Save
    report = {
        "lf_set"       : LF_COLS_4,
        "lf_exp_status": "REMOVED (Option A): retained as continuous feature for RankNet",
        "lf_loc_fix"   : loc_result,
        "table1_joint_coverage": t1,
        "table2_conflict_rate" : t2,
        "table3_cramers_v"     : t3,
        "fleiss_kappa"         : kappa_result,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "post_fix_stage1_rerun.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Done] Report saved to '{out_path}'")
    print("=" * 68)


if __name__ == "__main__":
    main()
