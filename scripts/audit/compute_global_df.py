import pandas as pd
import json
import underthesea
import re
from collections import Counter
import os

def get_tokens(text):
    if not text or str(text).lower() == 'nan': return set()
    stopwords = {'và', 'của', 'các', 'có', 'những', 'cho', 'với', 'trong', 'về', 'là', 'để', 'các', 'được'}
    try:
        segmented = underthesea.word_tokenize(str(text).lower(), format='text')
        tokens = segmented.split()
    except Exception:
        tokens = re.split(r'[,|/;:\n\-]+', str(text).lower())
    return {t.strip() for t in tokens if len(t.strip()) > 1 and t.strip() not in stopwords}

def compute_global_df():
    print("Computing global Document Frequency for skills...")
    df_j = pd.read_csv('data/JOB_DATA_FINAL.csv')
    df_u = pd.read_csv('data/USER_DATA_FINAL.csv')
    
    # Remove obvious duplicates
    df_j = df_j.drop_duplicates(subset=['Job Requirements'])
    df_u = df_u.drop_duplicates(subset=['Skills'])
    
    jd_texts = df_j['Job Requirements'].dropna().tolist()
    cv_texts = df_u['Skills'].dropna().tolist()
    
    N_JD = len(jd_texts)
    N_CV = len(cv_texts)
    N_TOTAL = N_JD + N_CV
    
    print(f"Total Unique JDs: {N_JD}")
    print(f"Total Unique CVs: {N_CV}")
    
    jd_counter = Counter()
    cv_counter = Counter()
    total_counter = Counter()
    
    for txt in jd_texts:
        tokens = get_tokens(txt)
        jd_counter.update(tokens)
        total_counter.update(tokens)
        
    for txt in cv_texts:
        tokens = get_tokens(txt)
        cv_counter.update(tokens)
        total_counter.update(tokens)
        
    global_df = {}
    for token, count in total_counter.items():
        jd_count = jd_counter.get(token, 0)
        cv_count = cv_counter.get(token, 0)
        global_df[token] = {
            'count_all': count,
            'df_all': count / N_TOTAL,
            'count_jd': jd_count,
            'df_jd': jd_count / N_JD if N_JD > 0 else 0,
            'count_cv': cv_count,
            'df_cv': cv_count / N_CV if N_CV > 0 else 0
        }
        
    output_path = 'data/global_skill_df.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(global_df, f, ensure_ascii=False, indent=2)
        
    print(f"Saved global DF to {output_path}")

if __name__ == "__main__":
    compute_global_df()
