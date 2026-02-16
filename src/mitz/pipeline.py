from __future__ import annotations

from typing import Dict, Any

import numpy as np
import scanpy as sc
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from .model import MitzModel, ensure_float_matrix


# ============================================================
# AnnData pipeline wrapper
# ============================================================
def run_mitz_invitro_full(
    adata: sc.AnnData,
    target_column: str,
    donor_labels,
    recipient_labels,
    gt_column: str,
    pos_label,
    ensemble_method: str = "rank_mean",
    seed: int = 42,
    xgb_cv_splits: int = 5,
    triplet_epochs: int = 80,
) -> Dict[str, Any]:
    """
    Full Mitz pipeline operating on AnnData.

    Steps:
    1. Extract mitochondrial genes
    2. Build internal reference (T vs Cancer)
    3. Fit MitzModel
    4. Write scores back into adata.obs
    5. Evaluate using GT (optional, invitro only)

    Returns dict with model, metrics, and scores.
    """

    print("\n========== Running Mitz invitro pipeline ==========")

    # --------------------------------------------------
    # 1. Select mitochondrial genes
    # --------------------------------------------------
    mito_mask = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
    mito_genes = adata.var_names[mito_mask]

    if len(mito_genes) == 0:
        raise ValueError("No mitochondrial genes found.")

    print(f"[1] Using {len(mito_genes)} mitochondrial genes")

    X = ensure_float_matrix(adata[:, mito_genes].X)

    # --------------------------------------------------
    # 2. Build internal reference
    # --------------------------------------------------
    is_t = adata.obs[target_column].isin(donor_labels).to_numpy()
    is_cancer = adata.obs[target_column].isin(recipient_labels).to_numpy()
    mask = is_t | is_cancer

    X_data = X[mask]
    y_internal = is_t[mask].astype(int)  # T=1, Cancer=0

    if len(np.unique(y_internal)) < 2:
        raise ValueError("Internal reference must contain both classes.")

    print(f"[2] T-cells={y_internal.sum()} | Cancer={(y_internal==0).sum()}")

    # --------------------------------------------------
    # 3. Fit model
    # --------------------------------------------------
    print("[3] Fitting MitzModel...")

    model = MitzModel(
        ensemble_method=ensemble_method,
        seed=seed,
        xgb_cv_splits=xgb_cv_splits,
        triplet_epochs=triplet_epochs,
    )
    model.fit(X_data, y_internal)

    scores = model.score_cancer_only(X_data, y_internal)
    ens_scores = scores["ensemble_scores_cancer"]
    pred_binary = model.predict_cancer_binary()
    thr = model.get_threshold()

    print(f"[4] Threshold = {thr:.4f}")

    # --------------------------------------------------
    # 4. Write results back to AnnData
    # --------------------------------------------------
    cancer_full_idx = np.where(is_cancer)[0]

    adata.obs["XGB_Score"] = np.nan
    adata.obs["Triplet_Score"] = np.nan
    adata.obs["Ensemble_Score"] = np.nan

    adata.obs.iloc[cancer_full_idx, adata.obs.columns.get_loc("XGB_Score")] = scores["xgb_scores_cancer"]
    adata.obs.iloc[cancer_full_idx, adata.obs.columns.get_loc("Triplet_Score")] = scores["triplet_scores_cancer"]
    adata.obs.iloc[cancer_full_idx, adata.obs.columns.get_loc("Ensemble_Score")] = ens_scores

    # Prediction label
    adata.obs["Prediction"] = "Other"
    adata.obs.loc[is_t, "Prediction"] = "T-cell"
    adata.obs.loc[is_cancer, "Prediction"] = "Non-hijacking"

    adata.obs.iloc[cancer_full_idx, adata.obs.columns.get_loc("Prediction")] = np.where(
        pred_binary == 1, "Hijacking", "Non-hijacking"
    )

    # --------------------------------------------------
    # 5. Evaluation (invitro only)
    # --------------------------------------------------
    print("[5] Evaluating with ground truth (optional)")

    true_labels = (adata.obs.iloc[cancer_full_idx][gt_column].to_numpy() == pos_label).astype(int)

    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, ens_scores)
    else:
        auc = np.nan

    acc = accuracy_score(true_labels, pred_binary)
    cm = confusion_matrix(true_labels, pred_binary)

    print(f"  Accuracy = {acc:.4f}")
    print(f"  ROC-AUC  = {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC unavailable")

    return {
        "adata": adata,
        "model": model,
        "mito_genes": list(mito_genes),
        "scores": scores,
        "threshold": thr,
        "metrics": {
            "auc": auc,
            "accuracy": acc,
            "confusion_matrix": cm,
        },
        "mask": mask,
        "y_internal": y_internal,
    }

