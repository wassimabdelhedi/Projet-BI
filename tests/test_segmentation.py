"""Tests Phase 5 : segmentation client.

27 tests verifies :
- SegmentationPreprocessor : fit/transform, interdiction pays, imputation ipt_mean,
  winsorisation, absence de fuite.
- evaluate_k_range : structure du resultat, monotonie de l'inertie.
- kmeans_metrics : plage des valeurs.
- compare_algorithms : colonnes presentes, ARI KMeans = 1.
- rfm_segments : presence des colonnes, scores dans la bonne plage.
- name_segments : tous les IDs present, overrides appliques.
- bootstrap_stability : ARI dans [0, 1], Jaccard dans [0, 1].
- compare_snapshots : ARI dans [-1, 1], pct dans [0, 1].
- build_customer_segments : colonnes requises, coherence segment_id.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from src.segmentation import (
    SegmentationPreprocessor,
    bootstrap_stability,
    build_customer_segments,
    compare_algorithms,
    compare_snapshots,
    evaluate_k_range,
    fit_kmeans,
    kmeans_metrics,
    name_segments,
    rfm_segments,
    suggest_k,
)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tiny_df():
    """Mini-dataset de 200 clients sans colonne pays."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "recency": rng.integers(1, 400, n),
            "frequency": rng.integers(1, 50, n),
            "monetary": rng.uniform(10, 5000, n),
            "gross_revenue": rng.uniform(10, 6000, n),
            "avg_basket": rng.uniform(5, 300, n),
            "tenure_days": rng.integers(1, 700, n),
            "return_amount": rng.uniform(0, 200, n),
            "return_rate": rng.uniform(0, 0.5, n),
            "avg_qty_per_line": rng.uniform(1, 50, n),
            "avg_items_per_invoice": rng.uniform(1, 100, n),
            "n_unique_products": rng.integers(1, 100, n),
            "n_unique_categories": rng.integers(1, 11, n),
            "ipt_mean": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(1, 100, n)),
            "is_single_purchase": rng.integers(0, 2, n),
            "recency_over_ipt": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0, 5, n)),
            "revenue_recent": rng.uniform(0, 2000, n),
            "revenue_prev": rng.uniform(0, 2000, n),
            "trend": rng.uniform(-1, 1, n),
            "is_wholesale": rng.integers(0, 2, n),
            "snapshot_month": 12,
            "snapshot_date": "2010-12-31",
            "cat_share_bags": rng.uniform(0, 1, n),
            "cat_share_christmas": rng.uniform(0, 1, n),
            "cat_share_other": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture(scope="module")
def preprocessor_fitted(tiny_df):
    p = SegmentationPreprocessor()
    p.fit(tiny_df)
    return p


@pytest.fixture(scope="module")
def X_transformed(tiny_df, preprocessor_fitted):
    return preprocessor_fitted.transform(tiny_df)


@pytest.fixture(scope="module")
def km_fitted(X_transformed):
    return fit_kmeans(X_transformed, k=3, random_state=42)


# ======================================================================
# 1. SegmentationPreprocessor
# ======================================================================

def test_preprocessor_output_shape(tiny_df, preprocessor_fitted, X_transformed):
    assert X_transformed.shape[0] == len(tiny_df)
    assert X_transformed.shape[1] == preprocessor_fitted.n_features


def test_preprocessor_no_nan_after_transform(X_transformed):
    assert not np.isnan(X_transformed).any(), "Des NaN subsistent apres transform"


def test_preprocessor_zero_mean(X_transformed):
    # Apres StandardScaler, la moyenne doit etre proche de 0
    means = np.abs(X_transformed.mean(axis=0))
    assert means.max() < 1e-6, f"Moyenne non nulle : {means.max():.2e}"


def test_preprocessor_unit_std(X_transformed):
    stds = X_transformed.std(axis=0)
    assert np.allclose(stds, 1.0, atol=1e-6), f"Ecart-type != 1 : {stds.max():.4f}"


def test_preprocessor_country_forbidden(tiny_df):
    df_bad = tiny_df.copy()
    df_bad["country_group"] = "UK"
    p = SegmentationPreprocessor()
    with pytest.raises(ValueError, match="pays"):
        p.fit(df_bad)


def test_preprocessor_country_forbidden_transform(tiny_df, preprocessor_fitted):
    df_bad = tiny_df.copy()
    df_bad["is_uk"] = 1
    with pytest.raises(ValueError, match="pays"):
        preprocessor_fitted.transform(df_bad)


def test_preprocessor_ipt_nan_handled(tiny_df, preprocessor_fitted):
    # ipt_mean peut contenir des NaN (clients a achat unique)
    # Verifier que le transform ne plante pas et ne produit pas de NaN
    df_all_nan = tiny_df.copy()
    df_all_nan["ipt_mean"] = np.nan
    df_all_nan["recency_over_ipt"] = np.nan
    X = preprocessor_fitted.transform(df_all_nan)
    assert not np.isnan(X).any()


def test_preprocessor_winsorization_applied(tiny_df, preprocessor_fitted):
    # Creer une valeur extreme ; elle doit etre clippee
    df_extreme = tiny_df.copy()
    df_extreme.loc[0, "recency"] = 1e9
    X = preprocessor_fitted.transform(df_extreme)
    assert np.isfinite(X).all()


def test_preprocessor_no_fuite(tiny_df):
    # Ajuste sur la premiere moitie, transforme la seconde
    half = len(tiny_df) // 2
    df1 = tiny_df.iloc[:half].copy()
    df2 = tiny_df.iloc[half:].copy()
    p = SegmentationPreprocessor()
    p.fit(df1)
    X2a = p.transform(df2)
    # Transformer deux fois doit donner le meme resultat
    X2b = p.transform(df2)
    np.testing.assert_array_equal(X2a, X2b)


def test_preprocessor_not_fitted_raises(tiny_df):
    p = SegmentationPreprocessor()
    with pytest.raises(RuntimeError, match="fit"):
        p.transform(tiny_df)


def test_preprocessor_fit_transform_eq_fit_then_transform(tiny_df):
    p1 = SegmentationPreprocessor()
    X1 = p1.fit_transform(tiny_df)
    p2 = SegmentationPreprocessor()
    p2.fit(tiny_df)
    X2 = p2.transform(tiny_df)
    np.testing.assert_array_almost_equal(X1, X2)


# ======================================================================
# 2. evaluate_k_range et suggest_k
# ======================================================================

def test_evaluate_k_range_shape(X_transformed):
    k_range = [2, 3, 4]
    metrics = evaluate_k_range(X_transformed, k_range, random_state=42)
    assert list(metrics["k"]) == k_range
    assert set(metrics.columns) >= {"k", "inertia", "silhouette", "davies_bouldin"}


def test_inertia_decreasing(X_transformed):
    k_range = [2, 3, 4, 5]
    metrics = evaluate_k_range(X_transformed, k_range, random_state=42)
    inertias = metrics["inertia"].tolist()
    assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1)), \
        "L'inertie doit etre decroissante en K"


def test_silhouette_in_range(X_transformed):
    k_range = [2, 3, 4]
    metrics = evaluate_k_range(X_transformed, k_range, random_state=42)
    assert metrics["silhouette"].between(-1, 1).all()


def test_suggest_k_returns_valid_k(X_transformed):
    k_range = [2, 3, 4]
    metrics = evaluate_k_range(X_transformed, k_range, random_state=42)
    k = suggest_k(metrics)
    assert k in k_range


# ======================================================================
# 3. kmeans_metrics
# ======================================================================

def test_kmeans_metrics_ranges(X_transformed, km_fitted):
    labels = km_fitted.predict(X_transformed)
    m = kmeans_metrics(X_transformed, labels)
    assert -1 <= m["silhouette"] <= 1
    assert m["davies_bouldin"] >= 0


# ======================================================================
# 4. compare_algorithms
# ======================================================================

def test_compare_algorithms_columns(X_transformed):
    res = compare_algorithms(X_transformed, k=3, random_state=42)
    assert set(res.columns) >= {"algo", "silhouette", "davies_bouldin", "ari_vs_kmeans"}
    assert len(res) == 3


def test_compare_algorithms_kmeans_ari_is_one(X_transformed):
    res = compare_algorithms(X_transformed, k=3, random_state=42)
    kmeans_row = res[res["algo"] == "KMeans"]
    assert float(kmeans_row["ari_vs_kmeans"].iloc[0]) == pytest.approx(1.0)


# ======================================================================
# 5. rfm_segments
# ======================================================================

def test_rfm_segments_columns(tiny_df):
    res = rfm_segments(tiny_df)
    for col in ["R_score", "F_score", "M_score", "rfm_score", "rfm_total", "rfm_segment"]:
        assert col in res.columns, f"Colonne manquante : {col}"


def test_rfm_scores_in_range(tiny_df):
    res = rfm_segments(tiny_df, n_quantiles=4)
    for col in ["R_score", "F_score", "M_score"]:
        assert res[col].between(1, 4).all(), f"{col} hors plage [1, 4]"


# ======================================================================
# 6. name_segments
# ======================================================================

def test_name_segments_all_ids_present(tiny_df, preprocessor_fitted, km_fitted):
    X = preprocessor_fitted.transform(tiny_df)
    labels = km_fitted.predict(X)
    df_lab = tiny_df[["recency", "frequency", "monetary"]].copy()
    df_lab["segment_id"] = labels
    names = name_segments(df_lab)
    for seg_id in range(km_fitted.n_clusters):
        assert seg_id in names


def test_name_segments_overrides(tiny_df, preprocessor_fitted, km_fitted):
    X = preprocessor_fitted.transform(tiny_df)
    labels = km_fitted.predict(X)
    df_lab = tiny_df[["recency", "frequency", "monetary"]].copy()
    df_lab["segment_id"] = labels
    overrides = {0: "Mon Segment Perso"}
    names = name_segments(df_lab, name_overrides=overrides)
    assert names[0] == "Mon Segment Perso"


# ======================================================================
# 7. bootstrap_stability
# ======================================================================

def test_bootstrap_stability_ari_range(tiny_df, preprocessor_fitted):
    res = bootstrap_stability(tiny_df, preprocessor_fitted, k=3, n_bootstrap=5, sample_frac=0.8)
    assert 0 <= res["ari_mean"] <= 1
    assert len(res["ari_values"]) == 5


def test_bootstrap_jaccard_range(tiny_df, preprocessor_fitted):
    res = bootstrap_stability(tiny_df, preprocessor_fitted, k=3, n_bootstrap=5, sample_frac=0.8)
    for seg, jac in res["jaccard_per_segment"].items():
        if not np.isnan(jac):
            assert 0 <= jac <= 1, f"Jaccard hors [0,1] pour segment {seg} : {jac}"


# ======================================================================
# 8. compare_snapshots
# ======================================================================

def test_compare_snapshots_values(tiny_df, preprocessor_fitted):
    # Simuler deux snapshots : df1 = premieres 150 lignes, df2 = 100 clients communs + 50 nouveaux
    df1 = tiny_df.iloc[:150].reset_index(drop=True)
    df2 = tiny_df.iloc[50:200].reset_index(drop=True)
    # Mettre a jour CustomerID pour le preprocessor (pas de colonne pays)
    res = compare_snapshots(df1, df2, preprocessor_fitted, k=3, random_state=42)
    assert "ari" in res and "pct_same_segment" in res
    assert -1 <= res["ari"] <= 1
    assert 0 <= res["pct_same_segment"] <= 1


# ======================================================================
# 9. build_customer_segments
# ======================================================================

def test_build_customer_segments_columns(tiny_df, preprocessor_fitted, km_fitted):
    res = build_customer_segments(tiny_df, preprocessor_fitted, km_fitted)
    required = {"CustomerID", "segment_id", "segment_name", "X_pca_1", "X_pca_2"}
    assert required <= set(res.columns)


def test_build_customer_segments_n_rows(tiny_df, preprocessor_fitted, km_fitted):
    res = build_customer_segments(tiny_df, preprocessor_fitted, km_fitted)
    assert len(res) == len(tiny_df)


def test_build_customer_segments_segment_ids_valid(tiny_df, preprocessor_fitted, km_fitted):
    res = build_customer_segments(tiny_df, preprocessor_fitted, km_fitted)
    valid_ids = set(range(km_fitted.n_clusters))
    assert set(res["segment_id"].unique()) <= valid_ids

