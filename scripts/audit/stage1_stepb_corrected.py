"""
Step B (corrected): Genuinely new analysis — 3+4-active group composition
==========================================================================
The prior "100% of 2-active pairs have lf_loc" is a mathematical inevitability
from Coverage(lf_loc)=100%, not a new empirical finding.

The genuinely new question: in the 3-active + 4-active groups (62% of data),
how many pairs have skill+sem+role ALL co-active simultaneously (with or without
lf_loc)? This is the fraction of data with evidence from 3 truly independent
signals, not contaminated by lf_loc's structural binary property.

Also computes the lf_loc gating bucket: pairs where lf_loc=-1 but skill+sem+role
all positive — used to test "gating vs evidence" in Gold-A annotation design.
"""

import io, os, sys, json, warnings
import numpy as np
import pandas as pd
from itertools import combinations

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.data import load_dataset, CVJobDatasetLoader
from src.weak import AspectLabelingFunctions

LF_COLS_4   = ["lf_skill", "lf_sem", "lf_role", "lf_loc"]
LF_SOFT     = ["lf_skill", "lf_sem", "lf_role"]    # "informative" LFs (can ABSTAIN)
SEP = "-" * 68

def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")


def load_4lf():
    df = load_dataset(data_dir="data", random_seed=42)
    loader = CVJobDatasetLoader(random_seed=42)
    df_train, _, _ = loader.get_job_disjoint_splits(df)
    lf = AspectLabelingFunctions(pos_percentile=75, neg_percentile=25)
    lf.fit(df_train)
    return df_train, lf.transform(df_train)


# ══════════════════════════════════════════════════════════════
# STEP B (corrected) — 3+4-active group: skill+sem+role composition
# ══════════════════════════════════════════════════════════════
def step_b_corrected(df_lf: pd.DataFrame) -> dict:
    section("STEP B (corrected) — 3+4-Active Group: skill+sem+role Co-Active Analysis")

    N = len(df_lf)
    active = (df_lf[LF_COLS_4] != 0)
    active_count = active.sum(axis=1)

    # Contextualise: prior result was a tautology
    print("""
  Correction to prior Step B framing:
    "100% of 2-active pairs have lf_loc" is a mathematical tautology:
    Coverage(lf_loc) = 100% → lf_loc is always active → any pair with
    exactly 2 active LFs must include lf_loc. This required no computation.

  The genuinely new question: in the 3-active + 4-active groups,
  how often do skill+sem+role appear together WITHOUT needing lf_loc
  as the "third wheel"? This fraction represents data with evidence
  from 3 independently-abstaining, non-structural signals.
    """)

    # Groups of interest
    mask_ge3 = (active_count >= 3)
    n_ge3 = int(mask_ge3.sum())
    sub_ge3 = df_lf[mask_ge3]

    print(f"  Pairs with >=3 active LFs: {n_ge3:,}  ({n_ge3/N:.1%} of total)")

    # Key metric: all 3 soft LFs active simultaneously (skill+sem+role all non-zero)
    soft_all_active = (
        (df_lf["lf_skill"] != 0) &
        (df_lf["lf_sem"]   != 0) &
        (df_lf["lf_role"]  != 0)
    )
    n_soft_all = int(soft_all_active.sum())
    print(f"\n  Pairs where skill+sem+role ALL active (with or without lf_loc):")
    print(f"    Count: {n_soft_all:,}  ({n_soft_all/N:.1%} of total)")
    print(f"    (These pairs have evidence from 3 independently-designed, abstaining signals)")

    # Among soft_all_active: how many also have lf_loc active?
    # (always yes, since lf_loc always active — but state explicitly for clarity)
    n_soft_all_with_loc = int((soft_all_active & (df_lf["lf_loc"] != 0)).sum())
    print(f"    Of these, also with lf_loc: {n_soft_all_with_loc:,} ({n_soft_all_with_loc/n_soft_all:.0%})")
    print(f"    Without lf_loc: 0 (Coverage=100% → tautological)")

    # Pairwise co-active rates among soft LFs
    print(f"\n  Pairwise co-active rates (fraction of all pairs):")
    for c1, c2 in combinations(LF_SOFT, 2):
        both = ((df_lf[c1] != 0) & (df_lf[c2] != 0)).mean()
        print(f"    {c1+'&'+c2:<22}: {both:.3f}  ({both:.1%})")

    # Breakdown of 3-active group by which soft LF is absent
    mask_3 = (active_count == 3)
    sub_3  = df_lf[mask_3]
    n_3    = len(sub_3)
    print(f"\n  3-active pairs ({n_3:,} pairs, {n_3/N:.1%}): which soft LF is absent?")
    for col in LF_SOFT:
        # In 3-active group, lf_loc is always present; the absent LF is always soft
        absent_count = int((sub_3[col] == 0).sum())
        print(f"    {col} ABSENT: {absent_count:>5,}  ({absent_count/n_3:.1%}) "
              f"→ pair uses {{{'skill,sem,role,loc'.replace(col.split('_')[1]+',','').replace(','+col.split('_')[1],'')}}}")

    # Breakdown of 4-active group
    mask_4 = (active_count == 4)
    n_4    = int(mask_4.sum())
    print(f"\n  4-active pairs ({n_4:,} pairs, {n_4/N:.1%}): all 4 LFs active")
    print(f"    These pairs have full evidence from all 3 soft LFs + lf_loc.")

    # Summary of "high-quality" supervision
    print(f"""
  Summary:
    "High-quality" supervision (skill+sem+role all active):
      {n_soft_all:,} pairs  ({n_soft_all/N:.1%} of total train pool)
      These are the pairs where Label Model inference is most reliable.
      lf_loc is always present too, but as a supplementary not primary signal.

    "lf_loc-dependent" supervision (only 1 soft LF active + lf_loc):
      {N - n_ge3:,} pairs (0-active = 0; 1-active = {int((active_count==1).sum())}) already excluded
      + {n_3:,} pairs with 3-active = lf_loc + only 2 soft LFs
      Of these 3-active, the pairs where lf_loc is the "swing vote" between
      2 soft LFs disagreeing are the highest-risk for Label Model reliability.
    """)

    return {
        "n_ge3_active": n_ge3,
        "pct_ge3_active": round(n_ge3 / N, 4),
        "n_skill_sem_role_all_active": n_soft_all,
        "pct_skill_sem_role_all_active": round(n_soft_all / N, 4),
        "n_4_active": n_4,
        "pct_4_active": round(n_4 / N, 4),
    }


# ══════════════════════════════════════════════════════════════
# lf_loc Gating Bucket — for Gold-A sampling design
# ══════════════════════════════════════════════════════════════
def compute_lfloc_gating_bucket(df_train: pd.DataFrame, df_lf: pd.DataFrame) -> dict:
    section("lf_loc Gating Bucket — Gold-A Diagnostic Bucket (NEW)")

    N = len(df_lf)

    # Gating bucket: lf_loc = -1 AND at least 2 soft LFs positive
    # This is the critical test for "gating vs evidence"
    loc_neg = (df_lf["lf_loc"] == -1)

    # Three flavours of "soft LFs positive"
    any_soft_pos  = (
        (df_lf["lf_skill"] ==  1) |
        (df_lf["lf_sem"]   ==  1) |
        (df_lf["lf_role"]  ==  1)
    )
    two_soft_pos  = (
        ((df_lf["lf_skill"] ==  1).astype(int) +
         (df_lf["lf_sem"]   ==  1).astype(int) +
         (df_lf["lf_role"]  ==  1).astype(int)) >= 2
    )
    all_soft_pos  = (
        (df_lf["lf_skill"] ==  1) &
        (df_lf["lf_sem"]   ==  1) &
        (df_lf["lf_role"]  ==  1)
    )

    bucket_any  = loc_neg & any_soft_pos
    bucket_2pos = loc_neg & two_soft_pos
    bucket_all  = loc_neg & all_soft_pos

    n_any  = int(bucket_any.sum())
    n_2pos = int(bucket_2pos.sum())
    n_all  = int(bucket_all.sum())

    print(f"""
  The key design question unanswered by current data:
    Is lf_loc a "soft evidence vote" (EM/vote alongside skill/sem/role)
    OR a "hard gate/filter" (location mismatch -> exclude regardless of skill)?

  Gold-A can answer this: when annotators see pairs where lf_loc=-1 but
  skill/sem/role are strongly positive, do they label "relevant" or "not relevant"?
    If "relevant" despite loc=-1  -> lf_loc is soft evidence, keep in EM
    If "not relevant" consistently -> lf_loc is a gate; treat like lf_exp
                                       (remove from Label Model, add as feature to RankNet)
    """)

    print(f"  Candidate gating bucket sizes:")
    print(f"    lf_loc=-1 AND >= 1 soft LF positive: {n_any:>5,}  ({n_any/N:.1%})")
    print(f"    lf_loc=-1 AND >= 2 soft LFs positive: {n_2pos:>5,}  ({n_2pos/N:.1%})  <- recommended")
    print(f"    lf_loc=-1 AND ALL 3 soft LFs positive: {n_all:>5,}  ({n_all/N:.1%})  <- most discriminative")

    print(f"""
  Recommended bucket for Gold-A:
    "Bucket E — lf_loc Gating Test":
      Criterion: lf_loc = -1 AND (lf_skill = +1 OR lf_sem = +1 OR lf_role = +1)
                 i.e., location mismatch despite at least one strong positive signal
      Size: {n_any:,} candidate pairs
      Annotation target: sample ~40-50 pairs from this bucket
      Schema addition: annotators should note "Would location be a dealbreaker?" (Y/N)

    Why a separate bucket (not absorbed into Bucket 3/4 minority conflict):
      Bucket 3/4 are defined by LF-level agreement/disagreement patterns generally.
      This bucket targets a SPECIFIC CONCEPTUAL QUESTION about lf_loc's role.
      Without oversampling this exact configuration, the gating question may be
      answered implicitly (by random chance) or not at all.
    """)

    # Show a few example pairs from this bucket
    gating_pairs = df_train[bucket_any].head(5)
    if len(gating_pairs) > 0 and "job_id" in gating_pairs.columns:
        print("  Example pairs in gating bucket (lf_loc=-1, some soft LF positive):")
        show_cols = ["job_id", "cand_id"] + [c for c in ["lf_skill","lf_sem","lf_role","lf_loc"]
                                               if c in df_lf.columns]
        subset = df_lf[bucket_any][show_cols[:6] if len(show_cols)>=6 else show_cols].head(5)
        print(subset.to_string(index=False))

    return {
        "gating_bucket_1soft_pos" : {"n": n_any,  "pct": round(n_any/N, 4)},
        "gating_bucket_2soft_pos" : {"n": n_2pos, "pct": round(n_2pos/N, 4)},
        "gating_bucket_all_soft_pos": {"n": n_all,  "pct": round(n_all/N, 4)},
        "annotation_target"       : "~40-50 pairs sampled from lf_loc=-1 & >=1 soft_pos",
        "decision_criteria": {
            "if_mostly_relevant"    : "lf_loc is soft evidence -> keep in Label Model",
            "if_mostly_irrelevant"  : "lf_loc is hard gate -> remove from LM, use as RankNet feature",
        }
    }


def main():
    print("=" * 68)
    print("  Step B (corrected) + lf_loc Gating Bucket Design")
    print("=" * 68)

    df_train, df_lf = load_4lf()
    print(f"\n  Loaded {len(df_train):,} train pairs.")

    result_b  = step_b_corrected(df_lf)
    result_gate = compute_lfloc_gating_bucket(df_train, df_lf)

    out = {"step_b_corrected": result_b, "lfloc_gating_bucket": result_gate}
    os.makedirs("reports", exist_ok=True)
    with open("reports/stage1_stepb_corrected.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Saved to 'reports/stage1_stepb_corrected.json'")
    print("=" * 68)


if __name__ == "__main__":
    main()
