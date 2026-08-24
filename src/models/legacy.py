import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge

from src.data.loader import FEATURE_COLS

# ---------------------------------------------------------
# 1. Model H: Heuristic Direct Ranking (Huynh et al., 2025)
# ---------------------------------------------------------
class ModelH_Heuristic:
    """Model H: Direct Heuristic Score Sorting without training."""
    def predict(self, df):
        scores = (0.30 * df['loc_match'] + 
                  0.25 * df['skill_iou'] + 
                  0.20 * df['exp_score'] + 
                  0.15 * df['role_match'] + 
                  0.10 * df['desc_sem_sim'])
        return scores.values

# ---------------------------------------------------------
# 2. Model A: Fixed Heuristic Feature + Binary BCE (Baseline)
# ---------------------------------------------------------
class ModelA_FixedBCE:
    """Model A: Baseline - Fixed Heuristic Score mapping trained with BCE."""
    def __init__(self):
        self.clf = LogisticRegression()
        
    def fit(self, df_train):
        # Uses single 1D feature: Heuristic score
        X = df_train['heuristic_score'].values.reshape(-1, 1)
        y = df_train['heuristic_label'].values
        self.clf.fit(X, y)
        
    def predict(self, df):
        X = df['heuristic_score'].values.reshape(-1, 1)
        return self.clf.predict_proba(X)[:, 1]

# ---------------------------------------------------------
# 3. Model B: Learned Weights + Pointwise BCE
# ---------------------------------------------------------
class ModelB_LearnedBCE:
    """Model B: Learned Feature Weights with BCE on Hard Heuristic Labels."""
    def __init__(self):
        self.clf = LogisticRegression(C=1.0)
        
    def fit(self, df_train):
        X = df_train[FEATURE_COLS].values
        y = df_train['heuristic_label'].values
        self.clf.fit(X, y)
        
    def predict(self, df):
        X = df[FEATURE_COLS].values
        return self.clf.predict_proba(X)[:, 1]

# ---------------------------------------------------------
# 4. Model B+: Learned Weights + Soft BCE on Probabilistic Labels
# ---------------------------------------------------------
class ModelB_Plus_SoftBCE:
    """Model B+: Learned Feature Weights with BCE on Probabilistic Labels y_tilde."""
    def __init__(self):
        self.reg = Ridge(alpha=1.0)
        
    def fit(self, df_train):
        X = df_train[FEATURE_COLS].values
        y_prob = df_train['y_prob'].values
        self.reg.fit(X, y_prob)
        
    def predict(self, df):
        X = df[FEATURE_COLS].values
        return self.reg.predict(X)

# ---------------------------------------------------------
# PyTorch Neural Scoring Network for LTR
# ---------------------------------------------------------
class RankScoringNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super(RankScoringNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------
# 5. Model C: Fixed Heuristic + Pairwise RankNet
# ---------------------------------------------------------
class ModelC_FixedRankNet:
    """Model C: Fixed Heuristic Score mapping trained with Pairwise RankNet."""
    def __init__(self, epochs=30, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.net = nn.Linear(1, 1)
        
    def fit(self, df_train):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # Build pairs per job
        pairs_x_i, pairs_x_j = [], []
        for job_id, group in df_train.groupby('job_id'):
            scores = group['heuristic_score'].values
            pos_idx = np.where(scores >= 0.45)[0]
            neg_idx = np.where(scores < 0.45)[0]
            
            for i in pos_idx:
                for j in neg_idx:
                    pairs_x_i.append([scores[i]])
                    pairs_x_j.append([scores[j]])
                    
        if len(pairs_x_i) == 0:
            return
            
        X_i = torch.tensor(pairs_x_i, dtype=torch.float32)
        X_j = torch.tensor(pairs_x_j, dtype=torch.float32)
        
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            s_i = self.net(X_i).squeeze(-1)
            s_j = self.net(X_j).squeeze(-1)
            loss = F.softplus(-(s_i - s_j)).mean()
            loss.backward()
            optimizer.step()
            
    def predict(self, df):
        X = torch.tensor(df[['heuristic_score']].values, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            scores = self.net(X).squeeze(-1).numpy()
        return scores

# ---------------------------------------------------------
# 6. Model D: Learned Weights + Pairwise RankNet
# ---------------------------------------------------------
class ModelD_LearnedRankNet:
    """Model D: Learned Feature Weights trained with Pairwise RankNet on Hard Pairs."""
    def __init__(self, epochs=30, lr=0.005):
        self.epochs = epochs
        self.lr = lr
        self.net = RankScoringNet(input_dim=5)
        
    def fit(self, df_train):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        pairs_x_i, pairs_x_j = [], []
        for job_id, group in df_train.groupby('job_id'):
            feat = group[FEATURE_COLS].values
            y_hard = group['heuristic_label'].values
            
            pos_idx = np.where(y_hard == 1)[0]
            neg_idx = np.where(y_hard == 0)[0]
            
            for i in pos_idx:
                for j in neg_idx:
                    pairs_x_i.append(feat[i])
                    pairs_x_j.append(feat[j])
                    
        if len(pairs_x_i) == 0:
            return
            
        X_i = torch.tensor(np.array(pairs_x_i), dtype=torch.float32)
        X_j = torch.tensor(np.array(pairs_x_j), dtype=torch.float32)
        
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            s_i = self.net(X_i)
            s_j = self.net(X_j)
            loss = F.softplus(-(s_i - s_j)).mean()
            loss.backward()
            optimizer.step()
            
    def predict(self, df):
        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            scores = self.net(X).numpy()
        return scores

# ---------------------------------------------------------
# 7. Model D+ (Proposed): Multi-Aspect Soft-RankNet
# ---------------------------------------------------------
class ModelD_Plus_ProposedSoftRankNet:
    """
    Model D+ (Proposed): Multi-Aspect Neural Network trained with 
    Confidence-Weighted Soft-RankNet Loss using Margin Weighting c_ij = y_i * (1 - y_j).
    """
    def __init__(self, epochs=40, lr=0.005, weight_decay=1e-4):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.net = RankScoringNet(input_dim=5, hidden_dim=24)
        
    def fit(self, df_train):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        pairs_x_i, pairs_x_j, pairs_weights = [], [], []
        
        for job_id, group in df_train.groupby('job_id'):
            feat = group[FEATURE_COLS].values
            y_probs = group['y_prob'].values
            N_g = len(group)
            
            # Form all directed pairs where y_prob_i > y_prob_j
            for i in range(N_g):
                for j in range(N_g):
                    if y_probs[i] > y_probs[j] + 0.10: # Minimum margin
                        c_ij = y_probs[i] * (1.0 - y_probs[j])
                        if c_ij > 0.05:
                            pairs_x_i.append(feat[i])
                            pairs_x_j.append(feat[j])
                            pairs_weights.append(c_ij)
                            
        if len(pairs_x_i) == 0:
            return
            
        X_i = torch.tensor(np.array(pairs_x_i), dtype=torch.float32)
        X_j = torch.tensor(np.array(pairs_x_j), dtype=torch.float32)
        W_ij = torch.tensor(np.array(pairs_weights), dtype=torch.float32)
        
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            s_i = self.net(X_i)
            s_j = self.net(X_j)
            
            # Numerically stable Confidence-Weighted Soft-RankNet Loss
            raw_loss = F.softplus(-(s_i - s_j))
            weighted_loss = (W_ij * raw_loss).sum() / (W_ij.sum() + 1e-8)
            
            weighted_loss.backward()
            optimizer.step()
            
    def predict(self, df):
        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            scores = self.net(X).numpy()
        return scores
