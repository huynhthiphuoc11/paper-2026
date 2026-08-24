import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.skill_normalization import extract_normalized_skills

class RealKaggleDatasetAdapter:
    """
    Adapter for processing real Kaggle Job & User Datasets:
    - JOB_DATA_FINAL.csv (14,634 jobs)
    - USER_DATA_FINAL.csv (3,983 candidates)
    
    Source: https://www.kaggle.com/datasets/phamtheds/job-dataset-for-recommendation
    
    Parses real Vietnamese job fields & user profiles to construct candidate-job matching pairs:
    1. loc_match (location string overlap)
    2. skill_iou (Jaccard IoU between user skills and job requirements)
    3. exp_score (asymmetric experience gap decay)
    4. role_match (TF-IDF similarity between desired job and job title)
    5. desc_sem_sim (TF-IDF cosine similarity between full candidate profile and job description)
    """
    def __init__(self, data_dir='data', random_seed=42, df_threshold=None):
        self.data_dir = data_dir
        self.job_path = os.path.join(data_dir, 'JOB_DATA_FINAL.csv')
        self.user_path = os.path.join(data_dir, 'USER_DATA_FINAL.csv')
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        self.high_df_skills = set()
        if df_threshold is not None:
            df_path = os.path.join(data_dir, 'global_skill_df.json')
            if os.path.exists(df_path):
                import json
                with open(df_path, 'r', encoding='utf-8') as f:
                    global_df = json.load(f)
                self.high_df_skills = {k for k, v in global_df.items() if v['df_all'] > df_threshold}
                print(f"[DATA LOADER] Loaded global DF. Filtering {len(self.high_df_skills)} skills with DF > {df_threshold}")
        
    def exists(self):
        return os.path.exists(self.job_path) and os.path.exists(self.user_path)

    def prepare_feature_space(self, n_jobs=80, n_candidates=120, force=False):
        """
        Sample jobs/users + fit TF-IDF once. Cached on the adapter so train subsample
        and gold pair features share the same vocabulary / IDF.
        """
        key = (int(n_jobs), int(n_candidates))
        if (
            not force
            and getattr(self, "_space", None) is not None
            and getattr(self, "_space_key", None) == key
        ):
            return self._space

        print(f"Loading real Kaggle dataset from '{self.data_dir}'...")
        df_j = pd.read_csv(self.job_path)
        df_u = pd.read_csv(self.user_path)
        df_jobs = (
            df_j.dropna(subset=["Job Title", "Job Requirements"])
            .assign(_source_index=lambda frame: frame.index)
            .sample(n=min(len(df_j), n_jobs), random_state=self.random_seed)
            .reset_index(drop=True)
        )
        df_users = (
            df_u.dropna(subset=["Desired Job", "Skills"])
            .assign(_source_index=lambda frame: frame.index)
            .sample(n=min(len(df_u), n_candidates), random_state=self.random_seed)
            .reset_index(drop=True)
        )

        job_titles = df_jobs["Job Title"].fillna("").tolist()
        user_desired_jobs = df_users["Desired Job"].fillna("").tolist()
        tfidf_role = TfidfVectorizer(max_features=2000)
        tfidf_role.fit(job_titles + user_desired_jobs)
        role_sim_matrix = cosine_similarity(
            tfidf_role.transform(user_desired_jobs), tfidf_role.transform(job_titles)
        )

        job_descs = (
            df_jobs["Job Description"].fillna("")
            + " "
            + df_jobs["Job Requirements"].fillna("")
        ).tolist()
        user_profiles = (
            df_users["Target"].fillna("")
            + " "
            + df_users["Skills"].fillna("")
            + " "
            + df_users["Work Experience"].fillna("")
        ).tolist()
        tfidf_desc = TfidfVectorizer(max_features=5000)
        tfidf_desc.fit(job_descs + user_profiles)
        desc_sim_matrix = cosine_similarity(
            tfidf_desc.transform(user_profiles), tfidf_desc.transform(job_descs)
        )

        self._space = {
            "df_jobs": df_jobs,
            "df_users": df_users,
            "tfidf_role": tfidf_role,
            "tfidf_desc": tfidf_desc,
            "role_sim_matrix": role_sim_matrix,
            "desc_sim_matrix": desc_sim_matrix,
            "n_jobs": n_jobs,
            "n_candidates": n_candidates,
        }
        self._space_key = key
        return self._space

    def _features_for_indices(self, job_idx, user_idx, space):
        df_jobs = space["df_jobs"]
        df_users = space["df_users"]
        job_row = df_jobs.iloc[job_idx]
        user_row = df_users.iloc[user_idx]
        job_skills = self._extract_skills(str(job_row.get("Job Requirements", "")))
        user_skills = self._extract_skills(str(user_row.get("Skills", "")))
        job_exp_years = self._parse_experience(str(job_row.get("Years of Experience", "1")))
        user_exp_years = self._parse_experience(str(user_row.get("Work Experience", "1")))
        job_loc = str(job_row.get("Job Address", "Hanoi")).lower()
        user_loc = str(user_row.get("Workplace Desired", "")).lower()
        loc_overlap = self._check_loc_overlap(user_loc, job_loc)
        loc_match = np.nan if pd.isna(loc_overlap) else (1.0 if loc_overlap else 0.0)
        union = max(1, len(job_skills.union(user_skills)))
        skill_iou = float(np.clip(len(job_skills.intersection(user_skills)) / union, 0.0, 1.0))
        experience_gap = float(abs(user_exp_years - job_exp_years))
        exp_score = float(np.exp(-experience_gap / max(1.0, job_exp_years)))
        required_skill_match_ratio = float(len(job_skills & user_skills) / max(1, len(job_skills)))
        missing_required_skill_ratio = float(len(job_skills - user_skills) / max(1, len(job_skills)))
        role_match = float(np.clip(space["role_sim_matrix"][user_idx, job_idx] * 1.5, 0.0, 1.0))
        desc_sem_sim = float(np.clip(space["desc_sim_matrix"][user_idx, job_idx] * 1.3, 0.0, 1.0))
        heuristic_score = (
            0.30 * loc_match
            + 0.25 * skill_iou
            + 0.20 * exp_score
            + 0.15 * role_match
            + 0.10 * desc_sem_sim
        )
        return {
            "job_title": str(job_row["Job Title"]),
            "loc_match": loc_match,
            "skill_iou": skill_iou,
            "exp_score": exp_score,
            "experience_gap": experience_gap,
            "job_required_years": float(job_exp_years),
            "cv_years": float(user_exp_years),
            "required_skill_match_ratio": required_skill_match_ratio,
            "missing_required_skill_ratio": missing_required_skill_ratio,
            "role_match": role_match,
            "desc_sem_sim": desc_sem_sim,
            "heuristic_score": heuristic_score,
            "heuristic_label": 1 if heuristic_score >= 0.45 else 0,
            "job_skills": " ".join(sorted(job_skills)),
            "user_skills": " ".join(sorted(user_skills)),
        }

    @staticmethod
    def parse_id_index(entity_id):
        return int(str(entity_id).split("_")[1])

    def features_for_pairs(self, pair_df, cand_col="cand_id", n_jobs=80, n_candidates=120):
        """Feature rows for raw-index IDs using the cached TF-IDF transformers."""
        space = self.prepare_feature_space(n_jobs=n_jobs, n_candidates=n_candidates)
        df_jobs_all = pd.read_csv(self.job_path)
        df_users_all = pd.read_csv(self.user_path)
        rows = []
        for _, r in pair_df.iterrows():
            job_idx = self.parse_id_index(r["job_id"])
            user_idx = self.parse_id_index(r[cand_col])
            if job_idx >= len(df_jobs_all) or user_idx >= len(df_users_all):
                continue
            job_row = df_jobs_all.iloc[job_idx]
            user_row = df_users_all.iloc[user_idx]
            role_match = float(np.clip(
                cosine_similarity(
                    space["tfidf_role"].transform([str(user_row.get("Desired Job", ""))]),
                    space["tfidf_role"].transform([str(job_row.get("Job Title", ""))]),
                )[0, 0] * 1.5,
                0.0,
                1.0,
            ))
            job_desc = str(job_row.get("Job Description", "")) + " " + str(
                job_row.get("Job Requirements", "")
            )
            user_profile = (
                str(user_row.get("Target", ""))
                + " "
                + str(user_row.get("Skills", ""))
                + " "
                + str(user_row.get("Work Experience", ""))
            )
            desc_sem_sim = float(np.clip(
                cosine_similarity(
                    space["tfidf_desc"].transform([user_profile]),
                    space["tfidf_desc"].transform([job_desc]),
                )[0, 0] * 1.3,
                0.0,
                1.0,
            ))
            pair_space = {
                **space,
                "df_jobs": pd.DataFrame([job_row]),
                "df_users": pd.DataFrame([user_row]),
                "role_sim_matrix": np.array([[role_match / 1.5]]),
                "desc_sim_matrix": np.array([[desc_sem_sim / 1.3]]),
            }
            feats = self._features_for_indices(0, 0, pair_space)
            rows.append({"job_id": r["job_id"], cand_col: r[cand_col], **feats})
        return pd.DataFrame(rows)

    def load_and_preprocess(
        self, n_jobs=80, n_candidates=120, candidates_per_job=70
    ):
        space = self.prepare_feature_space(n_jobs=n_jobs, n_candidates=n_candidates)
        df_jobs = space["df_jobs"]
        df_users = space["df_users"]

        records = []
        pair_id = 0
        # Fresh RNG stream for per-job CV subsample (same seed as historical loader)
        rng = np.random.RandomState(self.random_seed)

        for job_idx in range(len(df_jobs)):
            job_id = f"JOB_{int(df_jobs.iloc[job_idx]['_source_index']):02d}"
            selected_user_indices = rng.choice(
                len(df_users),
                size=min(len(df_users), candidates_per_job),
                replace=False,
            )
            for user_idx in selected_user_indices:
                cand_id = f"CV_{int(df_users.iloc[user_idx]['_source_index']):03d}"
                feats = self._features_for_indices(job_idx, user_idx, space)
                loc_match = feats["loc_match"]
                skill_iou = feats["skill_iou"]
                exp_score = feats["exp_score"]
                role_match = feats["role_match"]
                desc_sem_sim = feats["desc_sem_sim"]

                legacy_composite_quality = (
                    0.35 * skill_iou
                    + 0.35 * desc_sem_sim
                    + 0.15 * exp_score
                    + 0.15 * role_match
                )
                if legacy_composite_quality >= 0.40 and loc_match > 0:
                    legacy_gold_relevance = 2
                elif legacy_composite_quality >= 0.25:
                    legacy_gold_relevance = 1
                else:
                    legacy_gold_relevance = 0

                records.append(
                    {
                        "pair_id": pair_id,
                        "job_id": job_id,
                        "cand_id": cand_id,
                        "job_title": feats["job_title"],
                        "loc_match": loc_match,
                        "skill_iou": skill_iou,
                        "exp_score": exp_score,
                        "experience_gap": feats["experience_gap"],
                        "job_required_years": feats["job_required_years"],
                        "cv_years": feats["cv_years"],
                        "required_skill_match_ratio": feats["required_skill_match_ratio"],
                        "missing_required_skill_ratio": feats["missing_required_skill_ratio"],
                        "role_match": role_match,
                        "desc_sem_sim": desc_sem_sim,
                        "heuristic_score": feats["heuristic_score"],
                        "heuristic_label": feats["heuristic_label"],
                        "job_skills": feats["job_skills"],
                        "user_skills": feats["user_skills"],
                        "legacy_composite_quality": legacy_composite_quality,
                        "legacy_gold_relevance": legacy_gold_relevance,
                        "legacy_gold_label_binary": 1 if legacy_gold_relevance >= 1 else 0,
                        "gold_relevance": legacy_gold_relevance,
                        "gold_label_binary": 1 if legacy_gold_relevance >= 1 else 0,
                    }
                )
                pair_id += 1

        df_out = pd.DataFrame(records)
        print(f"Constructed {len(df_out)} real candidate-job matching evaluation pairs.")
        return df_out

    def _extract_skills(self, text):
        skills = extract_normalized_skills(text)
        if hasattr(self, 'high_df_skills'):
            skills = skills - self.high_df_skills
        return skills
        if not text or str(text).lower() == "nan":
            return set()
        
        # Use underthesea for proper Vietnamese compound segmentation
        import underthesea
        try:
            segmented = underthesea.word_tokenize(text.lower(), format='text')
            tokens = segmented.split()
        except Exception:
            # Fallback to naive if underthesea fails
            import re
            tokens = re.split(r'[,|/;:\n\-]+', text.lower())
            
        stopwords = {'và', 'của', 'các', 'có', 'những', 'cho', 'với', 'trong', 'về', 'là', 'để', 'các', 'được'}
        skills = {t.strip() for t in tokens if len(t.strip()) > 1 and t.strip() not in stopwords}
        
        if hasattr(self, 'high_df_skills'):
            skills = skills - self.high_df_skills
            
        return skills

    def _parse_experience(self, text):
        nums = re.findall(r'\d+', str(text))
        if nums:
            return float(nums[0])
        return 1.0

    def _check_loc_overlap(self, loc1, loc2):
        if not loc1 or not loc2 or loc1 == 'nan' or loc2 == 'nan':
            return np.nan
        common_cities = ['hà nội', 'hồ chí minh', 'tphcm', 'đà nẵng', 'cần thơ', 'hải phòng', 'bình dương']
        for c in common_cities:
            if c in loc1 and c in loc2:
                return True
                
        # Fallback to simple substring match (e.g. 'bắc ninh' in 'bắc giang, bắc ninh')
        # Avoid naive split intersection which falsely matches 'bắc ninh' and 'bắc giang' on 'bắc'
        return (loc1 in loc2) or (loc2 in loc1)


class CVJobDatasetLoader:
    """
    Data Loader and Simulator for Vietnamese CV-Job Ranking Experiment.
    Simulates 80 Jobs, 117 Candidates, yielding ~5,817 Job-CV pairs with 5 core features.
    """
    def __init__(self, n_jobs=80, n_candidates=117, data_dir='data', random_seed=42):
        self.n_jobs = n_jobs
        self.n_candidates = n_candidates
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
    def generate_simulated_data(self):
        records = []
        pair_id = 0
        
        for job_id in range(self.n_jobs):
            candidates_for_job = self.rng.choice(self.n_candidates, size=self.rng.randint(70, 75), replace=False)
            job_skill_strictness = self.rng.uniform(0.4, 0.9)
            job_exp_req = self.rng.uniform(0.3, 0.8)
            
            for cand_id in candidates_for_job:
                true_quality = self.rng.beta(2, 5) if self.rng.rand() > 0.3 else self.rng.beta(5, 2)
                
                loc_match = 1.0 if self.rng.rand() < (0.7 if true_quality > 0.5 else 0.4) else 0.0
                skill_iou = np.clip(true_quality * job_skill_strictness + self.rng.normal(0, 0.1), 0.0, 1.0)
                exp_score = np.clip(true_quality * job_exp_req + self.rng.normal(0, 0.1), 0.0, 1.0)
                role_match = np.clip(true_quality + self.rng.normal(0, 0.15), 0.0, 1.0)
                desc_sem_sim = np.clip(0.6 * true_quality + 0.4 * skill_iou + self.rng.normal(0, 0.08), 0.0, 1.0)
                
                if true_quality >= 0.70:
                    gold_relevance = 2
                elif true_quality >= 0.45:
                    gold_relevance = 1
                else:
                    gold_relevance = 0
                    
                heuristic_score = (0.30 * loc_match + 
                                   0.25 * skill_iou + 
                                   0.20 * exp_score + 
                                   0.15 * role_match + 
                                   0.10 * desc_sem_sim)
                
                heuristic_label = 1 if heuristic_score >= 0.45 else 0
                
                records.append({
                    'pair_id': pair_id,
                    'job_id': f"JOB_{job_id:02d}",
                    'cand_id': f"CV_{cand_id:03d}",
                    'loc_match': loc_match,
                    'skill_iou': skill_iou,
                    'exp_score': exp_score,
                    'role_match': role_match,
                    'desc_sem_sim': desc_sem_sim,
                    'heuristic_score': heuristic_score,
                    'heuristic_label': heuristic_label,
                    'gold_relevance': gold_relevance,
                    'gold_label_binary': 1 if gold_relevance >= 1 else 0
                })
                pair_id += 1
                
        df = pd.DataFrame(records)
        return df

    def get_job_disjoint_splits(self, df):
        manifest_path = os.path.join(self.data_dir, 'splits', 'split_manifest.csv')
        if os.path.exists(manifest_path):
            print(f"[DATA LOADER] Loading deterministic job-disjoint split manifest from '{manifest_path}'...")
            df_manifest = pd.read_csv(manifest_path)
            train_jobs = df_manifest[df_manifest['split'] == 'train']['job_id'].values
            dev_jobs = df_manifest[df_manifest['split'] == 'dev']['job_id'].values
            test_jobs = df_manifest[df_manifest['split'] == 'test']['job_id'].values
        else:
            print(f"[DATA LOADER] Note: Manifest file not found. Performing dynamic seed-based shuffle split...")
            all_jobs = df['job_id'].unique()
            self.rng.shuffle(all_jobs)
            
            n_total = len(all_jobs)
            n_train = max(1, int(0.8 * n_total))
            n_dev = max(1, int(0.06 * n_total))
            
            train_jobs = all_jobs[:n_train]
            dev_jobs = all_jobs[n_train:n_train+n_dev]
            test_jobs = all_jobs[n_train+n_dev:]
        
        df_train = df[df['job_id'].isin(train_jobs)].copy()
        df_dev = df[df['job_id'].isin(dev_jobs)].copy()
        df_test = df[df['job_id'].isin(test_jobs)].copy()
        
        return df_train, df_dev, df_test


def load_dataset(data_dir='data', random_seed=42, df_threshold=0.15):
    """
    Unified dataset loader. Automatically detects real Kaggle JOB_DATA_FINAL.csv & USER_DATA_FINAL.csv.
    Frozen df_threshold=0.15 based on sensitivity analysis.
    """
    adapter = RealKaggleDatasetAdapter(data_dir=data_dir, random_seed=random_seed, df_threshold=df_threshold)
    if adapter.exists():
        print(f"[DATA LOADER] Found real Kaggle dataset files in '{data_dir}'. Processing real dataset...")
        df = adapter.load_and_preprocess()
    else:
        print(f"[DATA LOADER] Note: Real dataset files not found in '{data_dir}'.")
        print(f" -> Falling back to high-fidelity synthetic benchmark generator...")
        loader = CVJobDatasetLoader(n_jobs=80, n_candidates=117, random_seed=random_seed)
        df = loader.generate_simulated_data()
        
    return df

FEATURE_COLS = ['loc_match', 'skill_iou', 'exp_score', 'role_match', 'desc_sem_sim']
