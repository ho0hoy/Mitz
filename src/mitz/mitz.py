import scanpy as sc
from mitz.pipeline import run_mitz_invitro_full

adata = sc.read_h5ad("bench1_scRNAseq.h5ad")

out = run_mitz_invitro_full(
    adata=adata,
    target_column="label",
    donor_labels=["MC_T_cell"],
    recipient_labels=["CC_cancer_cell", "MC_cancer_cell"],
    gt_column="label",
    pos_label="CC_cancer_cell",
    triplet_epochs=500,
)

model = out["model"]   # trained MitzModel
