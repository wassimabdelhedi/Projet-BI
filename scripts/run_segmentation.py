"""Script de generation des livrables de la Phase 5 (sans notebook Jupyter).
Execute le meme code que le notebook 04_segmentation.ipynb mais en script Python.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve, load_table
from src.segmentation import (
    SegmentationPreprocessor, evaluate_k_range, suggest_k,
    fit_kmeans, kmeans_metrics, compare_algorithms,
    rfm_segments, name_segments, bootstrap_stability,
    build_customer_segments, save_artifacts,
)

cfg = load_config()
seg_cfg = cfg.get("segmentation", {})
FIG = resolve(cfg, "figures_dir")
MODELS = ROOT / seg_cfg.get("models_dir", "models")
MODELS.mkdir(parents=True, exist_ok=True)

K = seg_cfg.get("k", 5)
K_RANGE = seg_cfg.get("k_range", [2, 3, 4, 5, 6, 7, 8])
SEED = cfg.get("seed", 42)

print("Chargement des donnees...")
df = load_table("customer_features", cfg)
print(f"  {len(df):,} clients | {df.shape[1]} variables")

COUNTRY_COLS = ["country_group", "is_uk"]
df_country = df[["CustomerID"] + COUNTRY_COLS].copy()
df_clust = df.drop(columns=COUNTRY_COLS)

log_vars = seg_cfg.get("log_vars", None)
winsor_q = seg_cfg.get("winsor_quantile", 0.99)

print("Preprocessing (log + winsor + scaler)...")
preprocessor = SegmentationPreprocessor(log_vars=log_vars, winsor_quantile=winsor_q)
X = preprocessor.fit_transform(df_clust)
print(f"  Matrice {X.shape} | {preprocessor.n_features} features")

print(f"\nChoix de K sur {K_RANGE}...")
metrics_df = evaluate_k_range(X, K_RANGE, random_state=SEED)
k_auto = suggest_k(metrics_df)
print(metrics_df.to_string(index=False))
print(f"  K suggere auto : {k_auto} | K retenu (config) : {K}")

# Figure Elbow/Silhouette/DB
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics_df.plot(x="k", y="inertia", marker="o", ax=axes[0], title="Elbow (inertie)", legend=False)
metrics_df.plot(x="k", y="silhouette", marker="o", ax=axes[1], title="Silhouette", legend=False, color="green")
axes[1].axhline(0.30, color="red", ls="--", label="objectif 0.30")
axes[1].axvline(K, color="orange", ls="--", label=f"K={K}")
axes[1].legend()
metrics_df.plot(x="k", y="davies_bouldin", marker="o", ax=axes[2], title="Davies-Bouldin", legend=False, color="red")
plt.tight_layout()
plt.savefig(FIG / "choix_k.png", dpi=150)
plt.close()

print(f"\nK-Means avec K={K}...")
km = fit_kmeans(X, k=K, random_state=SEED)
labels = km.predict(X)
metrics = kmeans_metrics(X, labels)
print(f"  Silhouette : {metrics['silhouette']} | Davies-Bouldin : {metrics['davies_bouldin']}")

df_seg = df_clust.copy()
df_seg["segment_id"] = labels
SEGMENT_NAMES = name_segments(df_seg)
df_seg["segment_name"] = df_seg["segment_id"].map(SEGMENT_NAMES)

print("\nRepartition :")
for sid, sname in sorted(SEGMENT_NAMES.items()):
    n = (labels == sid).sum()
    mon = df_seg.loc[df_seg["segment_id"] == sid, "monetary"].median()
    rec = df_seg.loc[df_seg["segment_id"] == sid, "recency"].median()
    print(f"  {sid} {sname:30s} {n:5d} clients | recency med={rec:.0f}j | monetary med={mon:,.0f}")

print("\nComparaison algorithmes...")
comp = compare_algorithms(X, k=K, random_state=SEED)
print(comp.to_string(index=False))

print("\nSegmentation RFM par quantiles...")
rfm = rfm_segments(df)
print(rfm["rfm_segment"].value_counts())

print("\nPCA 2D...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=SEED)
coords_pca = pca.fit_transform(X)
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(10, 7))
for sid, sname in sorted(SEGMENT_NAMES.items()):
    mask = labels == sid
    ax.scatter(coords_pca[mask, 0], coords_pca[mask, 1],
               label=f"{sname} ({mask.sum():,})", alpha=0.5, s=8,
               color=colors[sid % len(colors)])
ax.legend(markerscale=3, fontsize=9)
ax.set_title("Segments K-Means — espace PCA 2D")
plt.tight_layout()
plt.savefig(FIG / "segments_pca.png", dpi=150)
plt.close()

print("\nt-SNE 2D...")
from sklearn.manifold import TSNE
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
coords_low = np.zeros((len(X), 2))
coords_low[idx] = TSNE(n_components=2, random_state=SEED).fit_transform(X[idx])
fig_name = "segments_tsne.png"
title = "Segments K-Means — t-SNE 2D"
print("  t-SNE utilise")

fig, ax = plt.subplots(figsize=(10, 7))
for sid, sname in sorted(SEGMENT_NAMES.items()):
    mask = labels == sid
    ax.scatter(coords_low[mask, 0], coords_low[mask, 1],
               label=f"{sname} ({mask.sum():,})", alpha=0.5, s=8,
               color=colors[sid % len(colors)])
ax.legend(markerscale=3, fontsize=9)
ax.set_title(title)
plt.tight_layout()
plt.savefig(FIG / fig_name, dpi=150)
plt.close()

print("\nProfils des segments...")
PROFILE_VARS = ["recency", "frequency", "monetary", "avg_basket", "n_unique_products", "return_rate"]
profile = df_seg.groupby("segment_name")[PROFILE_VARS].median().round(1)
print(profile.to_string())

n_boot = seg_cfg.get("n_bootstrap", 50)
boot_frac = seg_cfg.get("bootstrap_sample_frac", 0.80)
print(f"\nStabilite bootstrap ({n_boot} tirages)...")
stability = bootstrap_stability(df_clust, preprocessor, k=K,
                                n_bootstrap=n_boot, sample_frac=boot_frac,
                                random_state=SEED)
print(f"  ARI moyen = {stability['ari_mean']} +/- {stability['ari_std']}")
print(f"  Objectif > 0.80 : {'ATTEINT' if stability['ari_mean'] >= 0.80 else 'NON ATTEINT'}")
print("  Jaccard par segment :")
for seg, jac in stability["jaccard_per_segment"].items():
    print(f"    Segment {seg} ({SEGMENT_NAMES.get(seg,'')}) : {jac:.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(stability["ari_values"], bins=20, edgecolor="white")
ax.axvline(stability["ari_mean"], color="red", ls="--", label=f"moyenne = {stability['ari_mean']}")
ax.axvline(0.80, color="green", ls=":", label="objectif = 0.80")
ax.set_title(f"Distribution ARI bootstrap (n={n_boot})")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "bootstrap_ari.png", dpi=150)
plt.close()

print("\nExport customer_segments.csv...")
seg_table = build_customer_segments(df, preprocessor, km)
out_csv = ROOT / "data/processed/customer_segments.csv"
seg_table.to_csv(out_csv, index=False)
print(f"  -> {out_csv} ({len(seg_table):,} lignes)")

print("\nSauvegarde des modeles...")
artifact_paths = save_artifacts(preprocessor, km, SEGMENT_NAMES, metrics, cfg)
for name, path in artifact_paths.items():
    print(f"  {name:10s} -> {path}")

print("\n=== RESUME PHASE 5 ===")
print(f"K = {K} | Silhouette = {metrics['silhouette']} | Davies-Bouldin = {metrics['davies_bouldin']}")
print(f"ARI bootstrap moyen = {stability['ari_mean']}")
summary = df_seg.merge(df_country, on="CustomerID").groupby("segment_name").agg(
    n_clients=("CustomerID", "count"),
    pct_clients=("CustomerID", lambda x: round(100 * len(x) / len(df_seg), 1)),
    pct_ca=("monetary", lambda x: round(100 * x.sum() / df_seg["monetary"].sum(), 1)),
    recency_med=("recency", "median"),
    freq_med=("frequency", "median"),
).sort_values("pct_ca", ascending=False)
print(summary.to_string())
print("\nPhase 5 terminee.")

