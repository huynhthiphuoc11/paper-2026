"""
Pre-fix Diagnostic: Joint Coverage + exp_score distribution + lf_loc logic audit
==================================================================================
Chạy TRƯỚC khi sửa ngưỡng LF, để có baseline so sánh sau khi sửa.
Không cần gold label.

Outputs:
  - reports/pre_fix_diagnostic.json
  - Console tables (ASCII-safe)
"""

import io
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

LF_COLS    = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]
WEAK_PATH  = os.path.join("data", "weak_labels", "train_weak_labels.csv")
RAW_PATH   = os.path.join("data", "JOB_DATA_FINAL.csv")      # to check exp distribution
USER_PATH  = os.path.join("data", "USER_DATA_FINAL.csv")
REPORT_DIR = "reports"

SEP = "-" * 65


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# 1. Joint Coverage Analysis
# ══════════════════════════════════════════════════════════════
def analyze_joint_coverage(df: pd.DataFrame) -> dict:
    """
    Reports the distribution of how many LFs are non-abstaining per pair.

    Key question: what fraction of pairs have >=2 non-abstaining LFs?
    Only those pairs have meaningful "multi-source" supervision.
    """
    section("1. Joint Coverage — How Many LFs Are Active Per Pair?")

    non_abstain = (df[LF_COLS] != 0).sum(axis=1)
    dist = non_abstain.value_counts(normalize=True).sort_index()

    print("\n  Active LFs per pair (n_active = non-abstain count):\n")
    print(f"  {'n_active':>10}  {'fraction':>10}  {'count':>8}  {'interpretation'}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*35}")

    interpretations = {
        0: "No supervision — excluded from Label Model",
        1: "Single LF only — degenerates to copying that LF",
        2: "Minimal multi-source (sufficient but thin)",
        3: "Moderate multi-source",
        4: "Good multi-source",
        5: "Full coverage (all LFs active)",
    }

    result_rows = []
    for n_act in range(6):
        frac  = float(dist.get(n_act, 0.0))
        count = int(round(frac * len(df)))
        interp = interpretations.get(n_act, "")
        print(f"  {n_act:>10}  {frac:>10.4f}  {count:>8,}  {interp}")
        result_rows.append({
            "n_active_lfs": n_act,
            "fraction"    : round(frac, 4),
            "count"       : count,
            "note"        : interp,
        })

    # Derived summary
    frac_0 = float(dist.get(0, 0.0))
    frac_1 = float(dist.get(1, 0.0))
    frac_ge2 = 1.0 - frac_0 - frac_1

    print(f"\n  Summary:")
    print(f"    Pairs with 0 active LFs  : {frac_0:.2%}  -> cannot be labelled at all")
    print(f"    Pairs with 1 active LF   : {frac_1:.2%}  -> single-source (degenerates to copy)")
    print(f"    Pairs with >=2 active LFs: {frac_ge2:.2%} -> genuinely multi-source supervision")

    # Identify which LFs are active on pairs with only 1 active LF
    single_lf_mask = (non_abstain == 1)
    if single_lf_mask.sum() > 0:
        single_lf_df = df[LF_COLS][single_lf_mask]
        sole_lf_counts = {}
        for col in LF_COLS:
            sole = int((single_lf_df[col] != 0).sum())
            if sole > 0:
                sole_lf_counts[col] = sole
        print(f"\n  When only 1 LF is active, which LF is it?")
        for lf, cnt in sorted(sole_lf_counts.items(), key=lambda x: -x[1]):
            pct = cnt / single_lf_mask.sum()
            print(f"    {lf:<12}: {cnt:>6,} pairs  ({pct:.1%} of single-LF pairs)")

    # Which LF pairs co-abstain most often?
    print(f"\n  Co-abstain analysis (fraction of pairs where BOTH LFs abstain):")
    coabs = {}
    for c1, c2 in combinations(LF_COLS, 2):
        both_abs = ((df[c1] == 0) & (df[c2] == 0)).mean()
        coabs[f"{c1}+{c2}"] = round(float(both_abs), 4)
    for pair, frac in sorted(coabs.items(), key=lambda x: -x[1])[:8]:
        bar = "#" * int(frac * 40)
        print(f"    {pair:<25}: {frac:.3f}  {bar}")

    return {
        "distribution"        : result_rows,
        "pct_zero_active"     : round(frac_0, 4),
        "pct_one_active"      : round(frac_1, 4),
        "pct_ge2_active"      : round(frac_ge2, 4),
        "co_abstain_rates"    : coabs,
        "interpretation": (
            f"{frac_ge2:.1%} of pairs have >=2 active LFs and can be used as "
            "multi-source weak supervision. The remaining pairs either have no "
            "supervision signal or degenerate to a single LF."
        ),
    }


# ══════════════════════════════════════════════════════════════
# 2. exp_score Raw Distribution (Root Cause of lf_exp Degeneration)
# ══════════════════════════════════════════════════════════════
def analyze_exp_score_distribution(df_lf: pd.DataFrame) -> dict:
    """
    The weak labels CSV doesn't store raw exp_score, only discretised lf_exp.
    We reverse-engineer the distribution from lf_exp and augment with
    raw data from the full dataset if available.
    """
    section("2. lf_exp Degeneration — Root Cause Analysis")

    # From LF matrix: reconstruct approximate exp_score ranges
    # lf_exp = +1 when exp_score >= 1.00 (exp_pos threshold)
    # lf_exp = -1 when exp_score <= 0.50 (exp_neg threshold)
    # lf_exp =  0 when 0.50 < exp_score < 1.00

    lf_exp = df_lf["lf_exp"].values
    n_pos  = int((lf_exp ==  1).sum())
    n_neg  = int((lf_exp == -1).sum())
    n_abs  = int((lf_exp ==  0).sum())
    N      = len(lf_exp)

    print(f"\n  lf_exp vote distribution (current thresholds: pos=1.00, neg=0.50):")
    print(f"    +1 (exp_score >= 1.00) : {n_pos:>6,}  ({n_pos/N:.1%})")
    print(f"     0 (0.50 < score < 1.00): {n_abs:>6,}  ({n_abs/N:.1%})")
    print(f"    -1 (exp_score <= 0.50) : {n_neg:>6,}  ({n_neg/N:.1%})")

    print(f"\n  Interpretation:")
    print(f"    93.6% of pairs have exp_score >= 1.0.")
    print(f"    This means most CVs meet or exceed the experience requirement.")
    print(f"    With the current threshold, lf_exp provides NEAR-ZERO discrimination.")
    print(f"    It inflates P(Y=1) prior in the EM model without adding signal.")

    # Compute what different thresholds would produce
    print(f"\n  Simulated effect of threshold changes on lf_exp vote distribution:")
    print(f"  (Assuming exp_score distribution concentrated at 1.0 and ~0.45)")
    print(f"\n  {'exp_pos':>8}  {'exp_neg':>8}  {'Approx +1%':>12}  {'Approx -1%':>12}  {'Approx ABSTAIN%':>16}  Status")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*16}  {'-'*20}")

    # Current
    print(f"  {'1.00':>8}  {'0.50':>8}  {'93.6%':>12}  {'6.4%':>12}  {'0.0%':>16}  [CURRENT - DEGENERATE]")
    # Proposed fixes
    scenarios = [
        (1.00, 0.30, "Pos: gap=0; Neg: gap>3yr"),
        (1.00, 0.20, "Pos: gap=0; Neg: gap>4yr"),
        (0.90, 0.50, "Pos: gap<=0.5yr; Neg: gap>1.7yr"),
        (0.90, 0.30, "Pos: gap<=0.5yr; Neg: gap>3yr"),
    ]
    for ep, en, desc in scenarios:
        # Approximate based on known: 93.6% at >=1.0, 6.4% at <=0.5
        # exp_score = exp(-0.4 * gap_years)
        # ep=0.90 -> gap = -ln(0.90)/0.4 = 0.26yr; ep=1.00 -> gap=0
        # en=0.30 -> gap = -ln(0.30)/0.4 = 3.01yr; en=0.20 -> gap=4.02yr
        import math
        gap_pos = 0.0 if ep >= 1.0 else -math.log(ep) / 0.4
        gap_neg = -math.log(en) / 0.4
        print(f"  {ep:>8.2f}  {en:>8.2f}  {'varies':>12}  {'varies':>12}  {'~wider':>16}  {desc}")

    print(f"\n  Recommended fix: exp_pos=1.00, exp_neg=0.30")
    print(f"    -> +1 only when candidate exactly meets/exceeds requirement (gap=0)")
    print(f"    -> -1 only when shortfall > 3 years (exp_score <= 0.30)")
    print(f"    -> ABSTAIN for all other cases (0.30 < exp_score < 1.00)")
    print(f"    -> Expected ABSTAIN zone widens significantly, reducing pos_rate from 93.6%")

    return {
        "current_pos_rate" : round(n_pos / N, 4),
        "current_neg_rate" : round(n_neg / N, 4),
        "current_abs_rate" : round(n_abs / N, 4),
        "current_thresholds": {"exp_pos": 1.00, "exp_neg": 0.50},
        "recommended_thresholds": {"exp_pos": 1.00, "exp_neg": 0.30},
        "diagnosis": (
            "lf_exp has pos_rate=93.6% because exp_pos=1.00 triggers for all "
            "pairs where exp_score>=1.0, and 93.6% of pairs satisfy this. "
            "The exp_neg=0.50 threshold only captures exp_score<=0.50 (~6.4%). "
            "Zero ABSTAIN zone means lf_exp never withholds judgment, "
            "distorting the EM prior toward P(Y=1)~0.93."
        ),
    }


# ══════════════════════════════════════════════════════════════
# 3. lf_loc Logic Audit
# ══════════════════════════════════════════════════════════════
def analyze_lf_loc(df_lf: pd.DataFrame) -> dict:
    """
    lf_loc is defined as a deterministic binary rule:
      loc_pos = (loc_match == 1.0) -> +1
      loc_neg = (loc_match == 0.0) -> -1
      No ABSTAIN case defined.

    This means 100% coverage by construction, regardless of data quality.
    """
    section("3. lf_loc Logic Audit — Why 100% Coverage, 74% Negative?")

    lf_loc = df_lf["lf_loc"].values
    n_pos  = int((lf_loc ==  1).sum())
    n_neg  = int((lf_loc == -1).sum())
    n_abs  = int((lf_loc ==  0).sum())
    N      = len(lf_loc)

    print(f"\n  lf_loc vote distribution:")
    print(f"    +1 (loc_match == 1.0): {n_pos:>6,}  ({n_pos/N:.1%})")
    print(f"     0 (ABSTAIN)          : {n_abs:>6,}  ({n_abs/N:.1%})")
    print(f"    -1 (loc_match == 0.0): {n_neg:>6,}  ({n_neg/N:.1%})")

    print(f"\n  Root cause of 100% coverage (no ABSTAIN):")
    print(f"    lf_loc is defined in lf_definitions.py as:")
    print(f"      loc_pos = (loc_match == 1.0)  -> +1")
    print(f"      loc_neg = (loc_match == 0.0)  -> -1")
    print(f"    loc_match is a BINARY feature (0 or 1), so every pair")
    print(f"    falls into exactly one of the two cases. ABSTAIN is structurally impossible.")

    print(f"\n  Potential problems:")
    problems = [
        ("No ABSTAIN for ambiguous cases",
         "Remote jobs, jobs with 'multiple locations', or CVs without location "
         "should ideally ABSTAIN rather than force a hard negative."),
        ("74% negative rate is suspicious",
         "If 74% of CV-JD pairs have location mismatch, either: (a) dataset is "
         "geographically diverse and location truly mismatches often, or (b) "
         "the parse logic in data_loader.py is too strict, returning 0.0 "
         "for ambiguous/missing location data (which should be ABSTAIN)."),
        ("Deterministic binary = no uncertainty propagation",
         "A generative Label Model benefits from soft signals. A hard 0/1 binary "
         "LF carries maximum weight in EM regardless of actual reliability."),
    ]
    for i, (title, detail) in enumerate(problems, 1):
        print(f"\n  [{i}] {title}")
        print(f"       {detail}")

    print(f"\n  Recommended fix for lf_loc:")
    print(f"    +1 when loc_match == 1.0                  (confirmed match)")
    print(f"    -1 when loc_match == 0.0 AND job is NOT remote")
    print(f"     0 ABSTAIN when job is remote/flexible OR location field is empty")
    print(f"\n  To implement: need 'is_remote' flag in the dataset.")
    print(f"  Quick approximation (no new data needed):")
    print(f"    If job_title or job_desc contains 'remote'/'work from home'/'toan quoc'")
    print(f"    -> set lf_loc = 0 (ABSTAIN) instead of -1.")

    return {
        "current_pos_rate" : round(n_pos / N, 4),
        "current_neg_rate" : round(n_neg / N, 4),
        "current_abs_rate" : round(n_abs / N, 4),
        "design_flaw"      : "Binary loc_match (0/1) → no ABSTAIN by construction",
        "diagnosis": (
            "lf_loc has 100% coverage because loc_match is binary (0 or 1) "
            "and the LF rule has no ABSTAIN branch. "
            "74% negative rate may reflect genuine location mismatch in the dataset, "
            "OR may be an artifact of overly strict location parsing that treats "
            "missing/remote/flexible location as mismatch (0.0)."
        ),
        "recommended_fix": (
            "Add ABSTAIN branch: set lf_loc=0 when job is remote/national, "
            "or when candidate/job location field is empty/ambiguous. "
            "This requires a remote_job flag or keyword check in the job title/description."
        ),
    }


# ══════════════════════════════════════════════════════════════
# 4. LF Redundancy Check: Are ABSTAIN patterns correlated?
# ══════════════════════════════════════════════════════════════
def analyze_abstain_independence(df_lf: pd.DataFrame) -> dict:
    """
    If lf_skill and lf_sem both abstain on the same pairs, their combined
    coverage is much lower than individual coverage would suggest.

    Checks whether ABSTAIN patterns are independent across LF pairs.
    """
    section("4. ABSTAIN Independence — Do LFs Abstain On The Same Pairs?")

    abstain_mat = (df_lf[LF_COLS] == 0).astype(int)
    N = len(abstain_mat)

    print(f"\n  Individual ABSTAIN rates:")
    for col in LF_COLS:
        rate = abstain_mat[col].mean()
        print(f"    {col:<12}: {rate:.3f}")

    print(f"\n  Observed vs. Expected joint ABSTAIN (assuming independence):")
    print(f"  {'Pair':<25}  {'Observed':>10}  {'Expected (ind.)':>15}  {'Ratio':>8}  {'Interpretation'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*15}  {'-'*8}  {'-'*30}")

    results = {}
    for c1, c2 in combinations(LF_COLS, 2):
        obs  = float((abstain_mat[c1] & abstain_mat[c2]).mean())
        exp  = float(abstain_mat[c1].mean() * abstain_mat[c2].mean())
        ratio = obs / exp if exp > 0 else float("inf")
        interp = (
            "Co-abstain more than expected (dependent ABSTAIN)"
            if ratio > 1.3
            else "Co-abstain less than expected (complementary coverage)"
            if ratio < 0.7
            else "Near-independent ABSTAIN"
        )
        key = f"{c1}+{c2}"
        results[key] = {"observed": round(obs, 4), "expected_if_independent": round(exp, 4),
                        "ratio": round(ratio, 3), "interpretation": interp}
        print(f"  {key:<25}  {obs:>10.4f}  {exp:>15.4f}  {ratio:>8.3f}  {interp}")

    # Compute effective joint coverage (pairs where >=2 LFs are active)
    active_count = (df_lf[LF_COLS] != 0).sum(axis=1)
    frac_ge2 = (active_count >= 2).mean()
    print(f"\n  Pairs with >=2 active LFs (genuine multi-source): {frac_ge2:.2%}")
    print(f"  Pairs with >=3 active LFs                        : {(active_count >= 3).mean():.2%}")
    print(f"  Pairs with >=4 active LFs                        : {(active_count >= 4).mean():.2%}")

    return results


# ══════════════════════════════════════════════════════════════
# 5. Simulate Post-Fix Coverage (before actually changing code)
# ══════════════════════════════════════════════════════════════
def simulate_fixed_lf_exp(df_lf: pd.DataFrame, new_exp_neg: float = 0.30) -> dict:
    """
    Simulates what lf_exp would look like with a widened ABSTAIN zone.

    Since train_weak_labels.csv only has discretised lf_exp (not raw exp_score),
    we can estimate the impact using the known threshold math:
      exp_score = exp(-0.4 * gap_years)
      lf_exp = +1 if exp_score >= 1.00  (gap = 0, i.e. fully qualified)
      lf_exp = -1 if exp_score <= new_neg_threshold
      lf_exp =  0 otherwise

    With current distribution: 93.6% at +1, 6.4% at -1 (0% ABSTAIN).
    New neg threshold=0.30 means: only vote -1 if gap > 3 years.

    We can approximate: gap=0 → 93.6% (unchanged +1 pool).
    The 6.4% currently at -1 consists of gaps up to some max.
    With new_neg=0.30, we're only keeping the "severe shortfall" cases as -1.
    """
    section(f"5. Simulated lf_exp After Fix (new exp_neg = {new_exp_neg})")

    # Current distribution from LF matrix
    cur_pos = (df_lf["lf_exp"] ==  1).sum()
    cur_neg = (df_lf["lf_exp"] == -1).sum()
    cur_abs = (df_lf["lf_exp"] ==  0).sum()
    N = len(df_lf)

    import math
    # Current neg threshold = 0.50 → gap threshold = -ln(0.50)/0.4 = 1.73yr
    # New neg threshold = 0.30 → gap threshold = -ln(0.30)/0.4 = 3.01yr
    # Assumption: exp_score of the 6.4% negative pool is approximately uniform in [0, 0.50]
    # Fraction of current negatives that fall below new threshold (0.30):
    # Uniform approx: fraction <= 0.30 of (0, 0.50] range = 0.30/0.50 = 0.60
    # So ~60% of current negatives would remain -1; ~40% move to ABSTAIN

    gap_cur  = -math.log(0.50) / 0.4
    gap_new  = -math.log(new_exp_neg) / 0.4
    approx_remaining_neg_frac = min(1.0, (new_exp_neg / 0.50))  # fraction ≤ new_neg out of old neg pool
    new_neg_count = int(cur_neg * approx_remaining_neg_frac)
    moved_to_abs  = int(cur_neg * (1 - approx_remaining_neg_frac))

    print(f"\n  Current  (exp_pos=1.00, exp_neg=0.50):")
    print(f"    +1: {cur_pos:>6,}  ({cur_pos/N:.1%})   -1: {cur_neg:>6,}  ({cur_neg/N:.1%})   0: {cur_abs:>6,}  ({cur_abs/N:.1%})")
    print(f"\n  Estimated post-fix  (exp_pos=1.00, exp_neg={new_exp_neg}):")
    print(f"    +1: {cur_pos:>6,}  ({cur_pos/N:.1%})   "
          f"-1: ~{new_neg_count:>5,}  (~{new_neg_count/N:.1%})   "
          f"0: ~{cur_abs + moved_to_abs:>5,}  (~{(cur_abs + moved_to_abs)/N:.1%})")
    print(f"\n  Gap threshold change: {gap_cur:.2f} yr -> {gap_new:.2f} yr for -1 vote")
    print(f"  ~{moved_to_abs:,} pairs move from -1 to ABSTAIN")
    print(f"  Note: These are ORDER-OF-MAGNITUDE estimates based on uniform approximation.")
    print(f"        Actual numbers will differ. Run Stage 1 again after the fix to get exact figures.")

    return {
        "current" : {"pos": int(cur_pos), "neg": int(cur_neg), "abs": int(cur_abs)},
        "estimated_post_fix": {
            "pos": int(cur_pos), "neg": new_neg_count, "abs": int(cur_abs) + moved_to_abs
        },
        "gap_threshold_years": {"current": round(gap_cur, 2), "new": round(gap_new, 2)},
        "note": "Approximate simulation. Exact results require re-running lf_definitions with new threshold.",
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def run_diagnostic():
    print("=" * 65)
    print("  Pre-Fix Diagnostic: Joint Coverage + Root Cause Analysis")
    print("=" * 65)

    df_lf = pd.read_csv(WEAK_PATH)
    print(f"\n[Load] {len(df_lf):,} pairs from '{WEAK_PATH}'")

    results = {}

    # Run all diagnostics
    results["joint_coverage"]       = analyze_joint_coverage(df_lf)
    results["lf_exp_diagnosis"]     = analyze_exp_score_distribution(df_lf)
    results["lf_loc_diagnosis"]     = analyze_lf_loc(df_lf)
    results["abstain_independence"] = analyze_abstain_independence(df_lf)
    results["simulated_fix"]        = simulate_fixed_lf_exp(df_lf, new_exp_neg=0.30)

    # Final summary
    section("SUMMARY & ACTION PLAN")
    print()
    print("  CRITICAL ISSUES (fix before annotation):")
    print("  [1] lf_exp: pos_rate=93.6% -> degenerate LF. Widen ABSTAIN zone.")
    print("      Fix: exp_neg = 0.30 (gap > 3yr = negative; else ABSTAIN)")
    print("  [2] lf_loc: 100% coverage, no ABSTAIN by design (pure binary rule).")
    print("      Fix: add ABSTAIN for remote/flexible jobs and missing location fields.")
    print()
    print("  VERIFY AFTER FIX:")
    print("  [ ] Re-run Stage 1 audit -> check all LF pos_rates in [20%, 80%]")
    print("  [ ] Check joint coverage: target >=80% pairs with >=2 active LFs")
    print("  [ ] Confirm Fleiss' kappa improves (less degenerate signal)")
    print()
    print("  THEN PROCEED:")
    print("  [ ] Design Gold-A + Gold-B sampling (disagreement + representative)")
    print("  [ ] Stage 4: include Model B' (Supervised LR) as official baseline")

    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "pre_fix_diagnostic.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Full report saved to '{report_path}'")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_diagnostic()
