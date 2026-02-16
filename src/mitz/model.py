from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


# =========================
# Utilities
# =========================
def ensure_float_matrix(A):
    """
    Convert input to a contiguous float32 2D matrix.

    Handles:
    - sparse matrices
    - object dtype arrays
    - strings like "[0.23]" or "3e-1"
    - nested arrays or lists

    This function ensures compatibility with:
    XGBoost / PyTorch / SHAP.
    """
    import re

    if hasattr(A, "toarray"):
        A = A.toarray()

    A = np.asarray(A)

    if A.ndim == 1:
        A = A.reshape(-1, 1)

    # Handle string arrays
    if A.dtype.kind in ("U", "S"):
        flat = A.astype(str)
        flat = np.char.strip(flat)
        flat = np.char.replace(flat, "[", "")
        flat = np.char.replace(flat, "]", "")

        def clean_one(s):
            s = re.sub(r"[^0-9eE\+\-\.]", "", s)
            return s if s != "" else "0"

        flat2 = np.vectorize(clean_one, otypes=[str])(flat)
        A = flat2.astype(np.float32)

    # Handle object arrays
    elif A.dtype == object:

        def to_float(v):
            if v is None:
                return 0.0
            if isinstance(v, (list, tuple, np.ndarray)):
                vv = np.asarray(v).ravel()
                v = vv[0] if vv.size > 0 else 0.0
            if isinstance(v, str):
                s = v.strip().replace("[", "").replace("]", "")
                s = re.sub(r"[^0-9eE\+\-\.]", "", s)
                return float(s) if s != "" else 0.0
            return float(v)

        A = np.array([[to_float(v) for v in row] for row in A], dtype=np.float32)

    else:
        A = np.asarray(A, dtype=np.float32)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(A)


def calculate_ensemble(score1, score2, method="rank_mean", weight=0.7):
    """
    Combine two scores into a single ensemble score.

    Default: rank_mean (robust to scale differences)
    Alternatives:
        - weighted average
        - simple mean
    """

    def minmax(x):
        x = np.asarray(x, dtype=np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    s1, s2 = minmax(score1), minmax(score2)

    if method == "rank_mean":
        r1 = pd.Series(s1).rank(method="average").to_numpy() / len(s1)
        r2 = pd.Series(s2).rank(method="average").to_numpy() / len(s2)
        return (r1 + r2) / 2.0

    if method == "weighted":
        return weight * s1 + (1.0 - weight) * s2

    return (s1 + s2) / 2.0


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Triplet backbone
# =========================
class TripletNet(nn.Module):
    """
    Simple MLP encoder used for triplet embedding learning.
    Produces a low-dimensional latent representation.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def forward(self, x):
        return self.encoder(x)


# =========================
# Threshold container
# =========================
@dataclass
class GMMThreshold:
    """
    Stores parameters of the fitted 2-component Gaussian Mixture Model.
    """
    threshold: float
    hijack_component: int
    means: np.ndarray
    weights: np.ndarray
    std: np.ndarray
    gmm: GaussianMixture


# =========================
# Main model
# =========================
class MitzModel:
    """
    Unified model combining:

    1. XGBoost internal-reference scoring
    2. Triplet embedding similarity scoring
    3. Ensemble score fusion
    4. Unsupervised 2-GMM threshold estimation

    Expected labels:
        T-cells = 1 (reference)
        Cancer  = 0 (target population)
    """

    def __init__(
        self,
        ensemble_method: str = "rank_mean",
        ensemble_weight: float = 0.7,
        seed: int = 42,
        xgb_params: Optional[Dict[str, Any]] = None,
        xgb_cv_splits: int = 5,
        triplet_epochs: int = 80,
        triplet_batch_size: int = 512,
        triplet_lr: float = 5e-4,
        triplet_clip: float = 50.0,
    ):
        self.ensemble_method = ensemble_method
        self.ensemble_weight = ensemble_weight
        self.seed = seed

        self.xgb_cv_splits = xgb_cv_splits
        self.triplet_epochs = triplet_epochs
        self.triplet_batch_size = triplet_batch_size
        self.triplet_lr = triplet_lr
        self.triplet_clip = triplet_clip

        # Default XGBoost config
        self.xgb_params = xgb_params or dict(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            verbosity=0,
        )

        # Learned components
        self.xgb_model_: Optional[xgb.XGBClassifier] = None
        self.triplet_model_: Optional[TripletNet] = None
        self.t_center_: Optional[np.ndarray] = None
        self.gmm_pack_: Optional[GMMThreshold] = None

        self.input_dim_: Optional[int] = None

    # ============================================================
    # XGBoost internal-reference scoring
    # ============================================================
    def _fit_xgb_internal(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, xgb.XGBClassifier]:
        """
        Train XGBoost using T vs Cancer labels.

        Returns:
            - OOF scores for cancer cells only
            - Final model trained on all data
        """
        seed_everything(self.seed)

        X = ensure_float_matrix(X)
        y = np.asarray(y).astype(int)

        skf = StratifiedKFold(n_splits=self.xgb_cv_splits, shuffle=True, random_state=self.seed)

        cancer_mask = (y == 0)
        cancer_scores_full = np.full(X.shape[0], np.nan, dtype=np.float32)

        # Out-of-fold scoring
        for train_idx, val_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**self.xgb_params, random_state=self.seed)
            model.fit(X[train_idx], y[train_idx])

            probs = model.predict_proba(X[val_idx])[:, 1].astype(np.float32)
            val_cancer = (y[val_idx] == 0)
            cancer_scores_full[val_idx[val_cancer]] = probs[val_cancer]

        # Final model
        rep_model = xgb.XGBClassifier(**self.xgb_params, random_state=self.seed)
        rep_model.fit(X, y)

        # Fill missing OOF entries if needed
        if np.isnan(cancer_scores_full[cancer_mask]).any():
            fill_probs = rep_model.predict_proba(X[cancer_mask])[:, 1].astype(np.float32)
            cancer_scores_full[cancer_mask] = np.where(
                np.isnan(cancer_scores_full[cancer_mask]),
                fill_probs,
                cancer_scores_full[cancer_mask],
            )

        return cancer_scores_full[cancer_mask], rep_model

