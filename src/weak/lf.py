import numpy as np
import pandas as pd

class AspectLabelingFunctions:
    """
    5 Aspect Labeling Functions with explicit ABSTAIN Policy (+1, -1, 0).
    All thresholds MUST be estimated/fitted ONLY on TRAIN data and frozen.
    
    LF Values:
      +1 : Positive Signal
       0 : ABSTAIN (Uncertain / Delegate)
      -1 : Negative Signal
    """
    def __init__(self, pos_percentile=75, neg_percentile=25):
        self.pos_percentile = pos_percentile
        self.neg_percentile = neg_percentile
        self.is_fitted = False
        self.thresholds = {}

    def fit(self, df_train):
        """
        Estimates feature thresholds ONLY from df_train.
        Never touches Dev or Test distributions.
        """
        # Skill IoU thresholds
        skill_pos = float(np.percentile(df_train['skill_iou'], self.pos_percentile))
        skill_neg = float(np.percentile(df_train['skill_iou'], self.neg_percentile))
        
        if skill_pos == 0.0:
            # 96% sparsity: force positive on ANY overlap, negative on zero
            skill_pos = 0.001
            skill_neg = 0.0
        elif skill_pos <= skill_neg:
            skill_pos, skill_neg = 0.20, 0.05
            
        # Description Semantic Similarity thresholds
        sem_pos = float(np.percentile(df_train['desc_sem_sim'], self.pos_percentile))
        sem_neg = float(np.percentile(df_train['desc_sem_sim'], self.neg_percentile))
        if sem_pos <= sem_neg:
            sem_pos, sem_neg = 0.25, 0.10
            
        # Experience Match thresholds
        exp_pos = float(np.percentile(df_train['exp_score'], self.pos_percentile))
        exp_neg = float(np.percentile(df_train['exp_score'], self.neg_percentile))
        if exp_pos <= exp_neg:
            exp_pos, exp_neg = 0.80, 0.40

        # Role Match thresholds
        role_pos = float(np.percentile(df_train['role_match'], self.pos_percentile))
        role_neg = float(np.percentile(df_train['role_match'], self.neg_percentile))
        if role_pos <= role_neg:
            role_pos, role_neg = 0.15, 0.00

        self.thresholds = {
            'skill_pos': skill_pos,
            'skill_neg': skill_neg,
            'sem_pos': sem_pos,
            'sem_neg': sem_neg,
            'exp_pos': exp_pos,
            'exp_neg': exp_neg,
            'role_pos': role_pos,
            'role_neg': role_neg,
        }
        self.is_fitted = True
        print(f"[LF AUDIT] Fitted thresholds on TRAIN set ({len(df_train)} pairs):")
        for k, v in self.thresholds.items():
            print(f"  - {k:12s}: {v:.4f}")
        return self

    def transform(self, df):
        """
        Applies frozen thresholds to produce discrete LF matrix (-1, 0, +1).
        Guarantees Positive cap Negative = empty set.
        """
        if not self.is_fitted:
            raise ValueError("AspectLabelingFunctions must be fitted on TRAIN before transform!")
            
        df_out = df.copy()
        
        # 1. LF_skill
        skill_pos = df_out['skill_iou'] >= self.thresholds['skill_pos']
        skill_neg = df_out['skill_iou'] <= self.thresholds['skill_neg']
        df_out['lf_skill'] = np.where(skill_pos, 1, np.where(skill_neg, -1, 0))
        
        # 2. LF_sem
        sem_pos = df_out['desc_sem_sim'] >= self.thresholds['sem_pos']
        sem_neg = df_out['desc_sem_sim'] <= self.thresholds['sem_neg']
        df_out['lf_sem'] = np.where(sem_pos, 1, np.where(sem_neg, -1, 0))
        
        # 3. LF_exp
        exp_pos = df_out['exp_score'] >= self.thresholds['exp_pos']
        exp_neg = df_out['exp_score'] <= self.thresholds['exp_neg']
        df_out['lf_exp'] = np.where(exp_pos, 1, np.where(exp_neg, -1, 0))

        # 4. LF_role
        role_pos = df_out['role_match'] >= self.thresholds['role_pos']
        role_neg = df_out['role_match'] <= self.thresholds['role_neg']
        df_out['lf_role'] = np.where(role_pos, 1, np.where(role_neg, -1, 0))

        # 5. LF_loc — FIX: add ABSTAIN branch for ambiguous/missing location.
        # Previous design: pure binary (loc_match in {0.0, 1.0}) — no ABSTAIN possible.
        # This caused 100% coverage and distorted EM parameter estimation.
        #
        # New logic:
        #   +1  when loc_match == 1.0  (confirmed geographic match)
        #   -1  when loc_match == 0.0  AND location fields are present for both sides
        #    0  ABSTAIN when location data is missing/empty on either side,
        #        or when job title/desc suggests remote/nationwide work.
        #
        # Heuristic for remote detection (no external flag needed):
        # Checks 'job_title' column if present; falls back to pure binary otherwise.
        REMOTE_KEYWORDS = [
            'remote', 'work from home', 'toan quoc', 'toàn quốc',
            'online', 'từ xa', 'tu xa', 'nationwide', 'anywhere',
        ]

        def is_remote_or_ambiguous(row) -> bool:
            """Returns True if job/CV location data is ambiguous or job is remote."""
            # Check for empty/NaN location on candidate side
            job_loc_val = str(row.get('job_loc', row.get('loc_match', ''))).lower()
            # Check job title for remote keywords if available
            title_val = str(row.get('job_title', '')).lower()
            if any(kw in title_val for kw in REMOTE_KEYWORDS):
                return True
            # Empty / unknown location strings
            if job_loc_val in ('', 'nan', 'none', 'unknown', 'n/a'):
                return True
            return False

        if 'job_title' in df_out.columns:
            remote_mask = df_out.apply(is_remote_or_ambiguous, axis=1)
        else:
            remote_mask = pd.Series(False, index=df_out.index)

        loc_pos  = (df_out['loc_match'] == 1.0)
        loc_neg  = (df_out['loc_match'] == 0.0) & (~remote_mask)
        loc_abs  = remote_mask | df_out['loc_match'].isna()  # ABSTAIN for remote/ambiguous/missing

        df_out['lf_loc'] = np.where(loc_pos, 1, np.where(loc_neg, -1, 0))
        # Override: remote/ambiguous → ABSTAIN regardless of loc_match value
        df_out.loc[loc_abs, 'lf_loc'] = 0
        
        return df_out

    def compute_lf_coverage_stats(self, df_lfs):
        """
        Computes Coverage, Positive Rate, Negative Rate, and Abstain Rate for each LF.
        """
        lf_cols = ['lf_skill', 'lf_sem', 'lf_exp', 'lf_role', 'lf_loc']
        stats = []
        n_samples = len(df_lfs)
        if n_samples == 0:
            return pd.DataFrame(columns=[
                'lf_name', 'coverage_rate', 'positive_rate',
                'negative_rate', 'abstain_rate'
            ])

        for col in lf_cols:
            vals = df_lfs[col].values
            pos_cnt = np.sum(vals == 1)
            neg_cnt = np.sum(vals == -1)
            abs_cnt = np.sum(vals == 0)
            cov_cnt = pos_cnt + neg_cnt
            
            stats.append({
                'lf_name': col,
                'coverage_rate': float(cov_cnt / n_samples),
                'positive_rate': float(pos_cnt / n_samples),
                'negative_rate': float(neg_cnt / n_samples),
                'abstain_rate': float(abs_cnt / n_samples)
            })
            
        return pd.DataFrame(stats)
