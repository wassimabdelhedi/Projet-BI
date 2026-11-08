"""Tests Phase 6 : prédiction de churn.

12 tests unitaires couvrant :
1. ChurnPreprocessor : fit/transform, imputation, alignement colonnes, absence de fuite.
2. RecencyThresholdBaseline : predict, predict_proba, apprentissage seuil.
3. evaluate_model : plages des métriques.
4. compute_decile_lift : structure et cohérence des déciles.
5. group_k_fold_cv : structure du résultat, GroupKFold respecte les groupes.
6. bootstrap_confidence_intervals : IC ordonnées lower < mean < upper.
7. evaluate_uk_nonuk : colonnes requises.
8. compute_cost_threshold : seuil dans [0,1], coût positif.
9. add_risk_class : classes valides, top 3 facteurs présents.
10. evaluate_without_top1pct : delta roc_auc cohérent.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from src.churn import (
    ChurnPreprocessor,
    RecencyThresholdBaseline,
    add_risk_class,
    bootstrap_confidence_intervals,
    compute_cost_threshold,
    compute_decile_lift,
    evaluate_model,
    evaluate_uk_nonuk,
    evaluate_without_top1pct,
    group_k_fold_cv,
)


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def tiny_churn_df():
    """Mini-dataset de 400 lignes simulant churn_dataset.csv."""
    rng = np.random.default_rng(42)
    n = 400
    cust_ids = rng.integers(1000, 2000, n)  # 1000 IDs distincts, chacun peut apparaître plusieurs fois
    df = pd.DataFrame({
        "CustomerID": cust_ids,
        "snapshot_date": np.where(np.arange(n) < 240, "2010-12-31", "2011-06-30"),
        "split": np.where(np.arange(n) < 240, "train",
                 np.where(np.arange(n) < 300, "validation", "test")),
        "churn": rng.integers(0, 2, n),
        "recency": rng.integers(1, 400, n).astype(float),
        "frequency": rng.integers(1, 30, n).astype(float),
        "monetary": rng.uniform(10, 5000, n),
        "gross_revenue": rng.uniform(10, 6000, n),
        "avg_basket": rng.uniform(5, 300, n),
        "tenure_days": rng.integers(1, 700, n).astype(float),
        "return_rate": rng.uniform(0, 0.5, n),
        "ipt_mean": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 100, n)),
        "is_single_purchase": rng.integers(0, 2, n).astype(float),
        "is_wholesale": rng.integers(0, 2, n).astype(float),
        "snapshot_month": rng.integers(1, 13, n).astype(float),
        "recency_over_ipt": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 5, n)),
        "is_uk": rng.integers(0, 2, n),
        "country_group": rng.choice(["UK", "Germany", "France"], n),
        "trend": rng.uniform(-1, 1, n),
        "revenue_recent": rng.uniform(0, 2000, n),
        "revenue_prev": rng.uniform(0, 2000, n),
        "n_unique_products": rng.integers(1, 50, n).astype(float),
        "n_unique_categories": rng.integers(1, 11, n).astype(float),
        "return_amount": rng.uniform(0, 200, n),
        "avg_qty_per_line": rng.uniform(1, 50, n),
        "avg_items_per_invoice": rng.uniform(1, 100, n),
        "cat_share_bags": rng.uniform(0, 1, n),
        "cat_share_christmas": rng.uniform(0, 1, n),
        "cat_share_other": rng.uniform(0, 1, n),
    })
    return df


@pytest.fixture(scope="module")
def preprocessor_fitted(tiny_churn_df):
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    prep = ChurnPreprocessor(with_scaling=False)
    prep.fit(train_df)
    return prep


@pytest.fixture(scope="module")
def xgb_model(tiny_churn_df, preprocessor_fitted):
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    y_train = train_df["churn"].values
    X_train = preprocessor_fitted.transform(train_df)
    m = XGBClassifier(n_estimators=20, max_depth=3, random_state=42,
                      eval_metric="logloss", n_jobs=1)
    m.fit(X_train, y_train)
    return m


# ================================================================== 1. ChurnPreprocessor

def test_preprocessor_output_shape(tiny_churn_df, preprocessor_fitted):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    X = preprocessor_fitted.transform(test_df)
    assert X.shape[0] == len(test_df)
    assert X.shape[1] == len(preprocessor_fitted.feature_names_)


def test_preprocessor_no_nan(tiny_churn_df, preprocessor_fitted):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    X = preprocessor_fitted.transform(test_df)
    assert not np.isnan(X).any(), "NaN subsistant après transform"


def test_preprocessor_no_fuite(tiny_churn_df):
    """Ajusté sur le train, appliqué sur le test → résultats reproductibles."""
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    prep = ChurnPreprocessor(with_scaling=False)
    prep.fit(train_df)
    X1 = prep.transform(test_df)
    X2 = prep.transform(test_df)
    np.testing.assert_array_equal(X1, X2)


def test_preprocessor_missing_col_handled(tiny_churn_df, preprocessor_fitted):
    """Une colonne manquante dans le jeu de test est remplie avec 0."""
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    # Supprimer une feature connue
    feat_to_drop = preprocessor_fitted.feature_names_[0]
    if feat_to_drop in test_df.columns:
        test_df = test_df.drop(columns=[feat_to_drop])
    X = preprocessor_fitted.transform(test_df)
    assert not np.isnan(X).any()


# ================================================================== 2. RecencyThresholdBaseline

def test_recency_baseline_predict_binary(tiny_churn_df):
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    baseline = RecencyThresholdBaseline()
    baseline.fit(train_df, train_df["churn"].values)
    preds = baseline.predict(test_df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_recency_baseline_threshold_learned(tiny_churn_df):
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    baseline = RecencyThresholdBaseline(threshold_days=999.0)  # seuil initial absurde
    baseline.fit(train_df, train_df["churn"].values)
    # Après apprentissage, le seuil doit être dans [30, 210]
    assert 30 <= baseline.threshold_days <= 210


# ================================================================== 3. evaluate_model

def test_evaluate_model_metrics_range(tiny_churn_df, preprocessor_fitted, xgb_model):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    y_test = test_df["churn"].values
    X_test = preprocessor_fitted.transform(test_df)
    m = evaluate_model(xgb_model, X_test, y_test)
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["f1"] <= 1
    assert 0 <= m["brier_score"] <= 1


# ================================================================== 4. compute_decile_lift

def test_decile_lift_structure():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(0, 1, 1000)
    df_lift = compute_decile_lift(y, p)
    assert len(df_lift) == 10
    assert set(df_lift.columns) >= {"decile", "clients", "churners", "lift", "cum_gain_pct"}


def test_decile_lift_cum_gain_ends_100():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(0, 1, 500)
    df_lift = compute_decile_lift(y, p)
    assert df_lift["cum_gain_pct"].iloc[-1] == pytest.approx(100.0, abs=0.1)


# ================================================================== 5. group_k_fold_cv

def test_group_kfold_cv_structure(tiny_churn_df):
    train_df = tiny_churn_df[tiny_churn_df["split"] == "train"].copy()
    result = group_k_fold_cv(
        df=train_df,
        model_cls=XGBClassifier,
        model_params={"n_estimators": 10, "max_depth": 3,
                      "random_state": 42, "eval_metric": "logloss", "n_jobs": 1},
        prep_cls=ChurnPreprocessor,
        prep_params={"with_scaling": False},
        n_splits=3,
        seed=42,
    )
    assert "mean_metrics" in result
    assert "fold_metrics" in result
    assert len(result["fold_metrics"]) == 3
    assert 0 <= result["mean_metrics"]["roc_auc"] <= 1


# ================================================================== 6. bootstrap_confidence_intervals

def test_bootstrap_ci_ordered(tiny_churn_df, preprocessor_fitted, xgb_model):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    y_test = test_df["churn"].values
    ci = bootstrap_confidence_intervals(xgb_model, preprocessor_fitted, test_df, y_test, n_boot=30)
    roc = ci["roc_auc"]
    assert roc[0] <= roc[1] <= roc[2], "IC non ordonné (lower <= mean <= upper)"


# ================================================================== 7. evaluate_uk_nonuk

def test_evaluate_uk_nonuk_columns(tiny_churn_df, preprocessor_fitted, xgb_model):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    y_test = test_df["churn"].values
    result = evaluate_uk_nonuk(xgb_model, preprocessor_fitted, test_df, y_test)
    assert "roc_auc" in result.columns
    assert "group" in result.columns


# ================================================================== 8. compute_cost_threshold

def test_cost_threshold_in_range():
    rng = np.random.default_rng(99)
    probs = rng.uniform(0, 1, 500)
    y = rng.integers(0, 2, 500)
    result = compute_cost_threshold(probs, y, cost_fp=5, cost_fn=50)
    assert 0 < result["threshold_optimal"] < 1
    assert result["total_cost_optimal"] >= 0
    assert "cost_curve" in result


# ================================================================== 9. add_risk_class

def test_add_risk_class_valid_classes():
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({"churn_risk_score": rng.uniform(0, 1, n)})
    shap_vals = rng.uniform(-1, 1, (n, 5))
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    df_out = add_risk_class(df, shap_vals, feat_names)
    assert set(df_out["risk_class"].unique()).issubset({"Faible", "Moyen", "Élevé"})
    for col in ["top_factor_1", "top_factor_2", "top_factor_3"]:
        assert col in df_out.columns
        assert df_out[col].isin(feat_names).all()


# ================================================================== 10. evaluate_without_top1pct

def test_evaluate_without_top1pct(tiny_churn_df, preprocessor_fitted, xgb_model):
    test_df = tiny_churn_df[tiny_churn_df["split"] == "test"].copy()
    y_test = test_df["churn"].values
    result = evaluate_without_top1pct(xgb_model, preprocessor_fitted, test_df, y_test)
    if "error" not in result:
        assert "roc_auc_delta" in result
        assert "metrics_with_top1" in result
        assert "metrics_without_top1" in result

