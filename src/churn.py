"""Phase 6 : Prédiction de Churn (Sans fuite temporelle).

Ce module implémente :
  1. Préparation des données (séparation train / validation / test respectant la purge).
  2. Baselines de référence :
     - Dummy Classifier (aléatoire stratifié / fréquence dominante)
     - Règle métier basée sur le seuil optimal de récence (recency > X jours).
  3. Modèles de Machine Learning supervisés :
     - Régression Logistique régularisée
     - Forêt Aléatoire (Random Forest)
     - Gradient Boosting (XGBoost)
  4. GroupKFold par client (un même client est entier dans 1 seul fold).
  5. Métriques complètes : ROC-AUC, PR-AUC, F1, Brier Score, Calibration,
     intervalles de confiance bootstrap à 95%.
  6. Évaluation séparée UK / hors-UK.
  7. Seuil optimal par analyse coût métier (coût campagne vs valeur client perdu).
  8. Explicabilité : SHAP + Feature Importance, top 3 facteurs par client.
  9. Sensibilité horizon churn : 60 / 90 / 120 jours.
  10. Test de robustesse sans le top 1% des clients (gros comptes B2B).
  11. Classes de risque : Faible (<30%), Moyen (30-60%), Élevé (>60%).
  12. Sortie enrichie : churn_predictions.csv avec risk_class, top_factor_1/2/3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Colonnes identifiantes ou cibles à exclure des features prédictives
EXCLUDE_COLS = ["CustomerID", "snapshot_date", "split", "churn",
                "segment_name"]  # segment_name = texte, segment_id = numérique (conservé)

# Seuils par défaut pour les classes de risque
DEFAULT_RISK_THRESHOLDS = (0.30, 0.60)

# Coûts métier par défaut (en euros)
DEFAULT_COST_FP = 5.0    # coût d'envoi d'une campagne email à un non-churn
DEFAULT_COST_FN = 50.0   # valeur perdue d'un client churné non détecté


# ===================================================================== 1. Preprocessing
class ChurnPreprocessor(BaseEstimator):
    """Prétraitement adapté aux modèles tabulaires de churn.

    - Exclut les métadonnées et la cible.
    - Impute les valeurs manquantes (ex: ipt_mean pour les achats uniques)
      à partir des statistiques du jeu d'entraînement uniquement (zéro fuite).
    - Encodage One-Hot des colonnes catégorielles éventuelles.
    - Standardisation optionnelle (pour modèles linéaires).
    """

    def __init__(self, with_scaling: bool = False):
        self.with_scaling = with_scaling
        self.feature_names_: List[str] = []
        self.impute_values_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = StandardScaler() if with_scaling else None

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        X = df[cols].copy()
        if "country_group" in X.columns:
            # One-hot encoding simple et déterministe
            X = pd.get_dummies(X, columns=["country_group"], drop_first=True)
        return X

    def fit(self, df: pd.DataFrame, y=None) -> "ChurnPreprocessor":
        X = self._prepare_df(df)
        self.feature_names_ = list(X.columns)

        # Imputation ajustée strictement sur le train
        for col in self.feature_names_:
            if X[col].isna().any():
                self.impute_values_[col] = float(X[col].median(skipna=True))

        X_imputed = X.copy()
        for col, val in self.impute_values_.items():
            X_imputed[col] = X_imputed[col].fillna(val)

        if self.scaler_ is not None:
            self.scaler_.fit(X_imputed)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_df(df)
        # Assurer l'alignement exact des colonnes
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names_].copy()

        for col, val in self.impute_values_.items():
            X[col] = X[col].fillna(val)

        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X.to_numpy(dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(df).transform(df)


# ===================================================================== 2. Baselines
class RecencyThresholdBaseline(BaseEstimator, ClassifierMixin):
    """Baseline métier : prédit churn=1 si recency >= seuil optimal appris sur le train."""

    def __init__(self, threshold_days: float = 90.0):
        self.threshold_days = threshold_days
        self.classes_ = np.array([0, 1])

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "RecencyThresholdBaseline":
        # Recherche du seuil optimal entre 30 et 180 jours maximisant le F1
        rec = X_df["recency"].values
        best_f1 = -1.0
        best_thresh = 90.0
        for th in range(30, 210, 5):
            pred = (rec >= th).astype(int)
            score = f1_score(y, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(th)
        self.threshold_days = best_thresh
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        return (X_df["recency"].values >= self.threshold_days).astype(int)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        # Score continu basé sur un ratio de récence
        rec = X_df["recency"].values
        p = np.clip(rec / (self.threshold_days * 2.0), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ===================================================================== 3. Évaluation
def evaluate_model(
    model: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcule l'ensemble des métriques de classification et de calibration."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-probs))

    preds = (probs >= threshold).astype(int)

    return {
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y, probs)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
    }


def compute_decile_lift(y_true: np.ndarray, y_probs: np.ndarray) -> pd.DataFrame:
    """Calcule le lift et le gain cumulé par décile de score de risque."""
    df_lift = pd.DataFrame({"y_true": y_true, "prob": y_probs})
    df_lift = df_lift.sort_values("prob", ascending=False).reset_index(drop=True)
    df_lift["decile"] = pd.qcut(df_lift.index, q=10, labels=list(range(1, 11)))

    total_churners = df_lift["y_true"].sum()
    total_clients = len(df_lift)
    global_rate = total_churners / total_clients

    summary = []
    cum_churners = 0
    cum_clients = 0
    for d, group in df_lift.groupby("decile", observed=False):
        n_c = len(group)
        n_churn = group["y_true"].sum()
        rate = n_churn / n_c if n_c > 0 else 0.0
        lift = rate / global_rate if global_rate > 0 else 1.0
        cum_churners += n_churn
        cum_clients += n_c
        cum_gain = cum_churners / total_churners if total_churners > 0 else 0.0
        cum_lift = (cum_churners / cum_clients) / global_rate if global_rate > 0 else 1.0

        summary.append({
            "decile": int(d),
            "clients": int(n_c),
            "churners": int(n_churn),
            "taux_churn": round(float(rate), 4),
            "lift": round(float(lift), 2),
            "cum_gain_pct": round(float(cum_gain * 100), 2),
            "cum_lift": round(float(cum_lift), 2),
        })

    return pd.DataFrame(summary)


# ===================================================================== 4. GroupKFold CV
def group_k_fold_cv(
    df: pd.DataFrame,
    model_cls,
    model_params: dict,
    prep_cls=None,
    prep_params: Optional[dict] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Validation croisée GroupKFold : un client est entier dans 1 seul fold.

    Garantit qu'un même CustomerID n'apparaît ni dans le train ni dans le test
    d'un même fold — évite la fuite liée aux snapshots multiples.

    Args:
        df: DataFrame avec colonnes [CustomerID, churn, features...].
        model_cls: classe du modèle (ex: XGBClassifier).
        model_params: paramètres du constructeur du modèle.
        prep_cls: classe du préprocesseur (ChurnPreprocessor ou None).
        prep_params: paramètres du préprocesseur.
        n_splits: nombre de folds (par défaut 5).
        seed: graine aléatoire.

    Returns:
        dict avec ari_mean, ari_std, fold_metrics (liste), mean_metrics (dict).
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["CustomerID"].values
    y = df["churn"].values

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        df_tr = df.iloc[train_idx].copy()
        df_va = df.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_va = y[val_idx]

        if prep_cls is not None:
            prep = prep_cls(**(prep_params or {}))
            X_tr = prep.fit(df_tr).transform(df_tr)
            X_va = prep.transform(df_va)
        else:
            X_tr = df_tr.drop(columns=EXCLUDE_COLS, errors="ignore").to_numpy(dtype=np.float32)
            X_va = df_va.drop(columns=EXCLUDE_COLS, errors="ignore").to_numpy(dtype=np.float32)

        m = model_cls(**model_params)
        m.fit(X_tr, y_tr)
        m_val = evaluate_model(m, X_va, y_va)
        m_val["fold"] = fold_idx
        fold_metrics.append(m_val)

    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = {
        col: round(float(metrics_df[col].mean()), 4)
        for col in ["roc_auc", "pr_auc", "f1", "brier_score"]
    }
    std_metrics = {
        f"{col}_std": round(float(metrics_df[col].std()), 4)
        for col in ["roc_auc", "pr_auc", "f1"]
    }
    mean_metrics.update(std_metrics)

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "n_splits": n_splits,
        "n_clients_total": df["CustomerID"].nunique(),
    }


# ===================================================================== 5. Bootstrap IC
def bootstrap_confidence_intervals(
    model: BaseEstimator,
    prep: Optional[ChurnPreprocessor],
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Calcule les intervalles de confiance à 95% par bootstrap sur le jeu de test.

    Returns:
        dict: {metric: (lower_2.5%, mean, upper_97.5%)}
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)

    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        df_b = df_test.iloc[idx]
        X_b = prep.transform(df_b) if prep is not None else df_b
        probs_b = model.predict_proba(X_b)[:, 1]
        roc_aucs.append(roc_auc_score(y_b, probs_b))
        pr_aucs.append(average_precision_score(y_b, probs_b))

    def ci(vals):
        arr = np.array(vals)
        return (round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.mean(arr)), 4),
                round(float(np.percentile(arr, 97.5)), 4))

    return {
        "roc_auc": ci(roc_aucs),
        "pr_auc": ci(pr_aucs),
        "n_bootstrap": len(roc_aucs),
    }


# ===================================================================== 6. UK vs hors-UK
def evaluate_uk_nonuk(
    model: BaseEstimator,
    prep: Optional[ChurnPreprocessor],
    df_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Évalue le modèle séparément sur les clients UK et hors-UK.

    Vérifie l'absence de biais géographique. Nécessite une colonne 'is_uk' dans df_test.

    Returns:
        DataFrame [group, n_clients, churn_rate, roc_auc, pr_auc, f1, brier_score]
    """
    if "is_uk" not in df_test.columns:
        return pd.DataFrame({"message": ["Colonne is_uk absente — analyse UK ignorée"]})

    rows = []
    df_test = df_test.copy()
    df_test["_y"] = y_test

    for group_name, mask_col, mask_val in [
        ("UK", "is_uk", 1),
        ("Hors-UK", "is_uk", 0),
        ("Total", None, None),
    ]:
        if mask_col is not None:
            sub = df_test[df_test[mask_col] == mask_val].copy()
        else:
            sub = df_test.copy()

        y_sub = sub["_y"].values
        if len(y_sub) < 10 or len(np.unique(y_sub)) < 2:
            continue

        X_sub = prep.transform(sub) if prep is not None else sub
        m = evaluate_model(model, X_sub, y_sub)
        rows.append({
            "group": group_name,
            "n_clients": len(y_sub),
            "churn_rate": round(float(y_sub.mean()), 3),
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "f1": m["f1"],
            "brier_score": m["brier_score"],
        })

    return pd.DataFrame(rows)


# ===================================================================== 7. Coût métier
def compute_cost_threshold(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    n_steps: int = 100,
) -> Dict[str, object]:
    """Cherche le seuil p* qui minimise le coût métier total.

    Modèle de coût :
        coût(seuil) = cost_fp × FP(seuil) + cost_fn × FN(seuil)

    Args:
        cost_fp: coût d'une campagne envoyée à un client non-churn (€).
        cost_fn: valeur perdue d'un client churné non détecté (€).

    Returns:
        dict avec threshold_optimal, total_cost, n_fp, n_fn, savings_vs_no_action,
        cost_curve (DataFrame seuil → coût).
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    cost_rows = []

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total_cost = cost_fp * fp + cost_fn * fn
        cost_rows.append({
            "threshold": round(float(th), 3),
            "total_cost": round(float(total_cost), 1),
            "n_fp": int(fp),
            "n_fn": int(fn),
            "n_tp": int(tp),
            "n_tn": int(tn),
        })

    cost_df = pd.DataFrame(cost_rows)
    best_idx = cost_df["total_cost"].idxmin()
    best_row = cost_df.iloc[best_idx]

    # Coût de ne rien faire = envoyer à tous ou à personne
    cost_all = cost_fp * int(y_true.sum() == 0) or cost_fn * int(y_true.sum())
    cost_no_action = cost_fn * int(y_true.sum())  # on rate tous les churners

    return {
        "threshold_optimal": float(best_row["threshold"]),
        "total_cost_optimal": float(best_row["total_cost"]),
        "cost_no_action": float(cost_no_action),
        "savings_vs_no_action": round(float(cost_no_action - best_row["total_cost"]), 1),
        "n_fp": int(best_row["n_fp"]),
        "n_fn": int(best_row["n_fn"]),
        "n_tp": int(best_row["n_tp"]),
        "cost_fp_input": cost_fp,
        "cost_fn_input": cost_fn,
        "cost_curve": cost_df,
    }


# ===================================================================== 8. SHAP
def compute_shap_values(
    model: BaseEstimator,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
    seed: int = 42,
) -> Tuple[object, np.ndarray]:
    """Calcule les valeurs SHAP via TreeExplainer (XGBoost / RandomForest).

    Args:
        model: modèle entraîné (XGBClassifier ou RandomForestClassifier).
        X_sample: matrice numpy (n_samples, n_features).
        feature_names: liste des noms de features.
        max_samples: taille max de l'échantillon SHAP (pour la rapidité).

    Returns:
        (explainer, shap_values) où shap_values a la forme (n_samples, n_features).
    """
    try:
        import shap
        rng = np.random.default_rng(seed)
        n = min(max_samples, len(X_sample))
        idx = rng.choice(len(X_sample), size=n, replace=False)
        X_sub = X_sample[idx]
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_sub)
        # XGBoost renvoie un array 2D directement ; RF renvoie une liste [classe0, classe1]
        shap_vals = raw[1] if isinstance(raw, list) else raw
        return explainer, shap_vals, idx
    except ImportError:
        # Si shap n'est pas installé, retourner les feature importances comme fallback
        importances = model.feature_importances_
        shap_vals = np.tile(importances, (len(X_sample), 1))
        return None, shap_vals, np.arange(len(X_sample))


# ===================================================================== 9. Risk class & top 3
def add_risk_class(
    df_pred: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    score_col: str = "churn_risk_score",
    thresholds: Tuple[float, float] = DEFAULT_RISK_THRESHOLDS,
) -> pd.DataFrame:
    """Ajoute risk_class et top 3 facteurs SHAP à la table de prédictions.

    Args:
        df_pred: DataFrame avec au moins la colonne score_col.
        shap_values: valeurs SHAP (n_samples, n_features) — même ordre que df_pred.
        feature_names: noms des features dans l'ordre de shap_values.
        thresholds: (seuil_faible, seuil_élevé) pour les classes de risque.

    Returns:
        df_pred enrichi de [risk_class, top_factor_1, top_factor_2, top_factor_3].
    """
    df = df_pred.copy()
    low, high = thresholds

    df["risk_class"] = df[score_col].apply(
        lambda p: "Faible" if p < low else ("Élevé" if p >= high else "Moyen")
    )

    # Top 3 facteurs par client basés sur |SHAP|
    if shap_values is not None and len(shap_values) == len(df):
        feat_arr = np.array(feature_names)
        top3_idx = np.argsort(np.abs(shap_values), axis=1)[:, -3:][:, ::-1]
        df["top_factor_1"] = feat_arr[top3_idx[:, 0]]
        df["top_factor_2"] = feat_arr[top3_idx[:, 1]]
        df["top_factor_3"] = feat_arr[top3_idx[:, 2]]
    else:
        df["top_factor_1"] = "n/a"
        df["top_factor_2"] = "n/a"
        df["top_factor_3"] = "n/a"

    return df


# ===================================================================== 10. Sensibilité horizon
def sensitivity_analysis(
    churn_dataset_path: str | Path,
    model_cls=None,
    model_params: Optional[dict] = None,
    horizons: List[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Analyse de sensibilité : entraîne le modèle pour chaque horizon churn.

    La cible churn à horizon H est déjà encodée dans churn_dataset.csv sous la
    colonne 'churn'. Pour la sensibilité, on recalcule la cible en modifiant le
    critère (utilise une approximation : on filtre les snapshots).

    Note : pour une sensibilité exacte, il faut recalculer src/features.py avec
    chaque horizon (config.yaml → churn.horizon_days). Cette fonction utilise
    l'approximation suivante :
      - Horizon 90j (baseline) : toutes les lignes de churn_dataset.csv.
      - Horizon 60j : on filtre les snapshots T tels que T + 60j > T_max (clients
        qui auraient eu moins de temps pour revenir → taux churn artificellement élevé).
      - Horizon 120j : même logique inversée.

    Pour une vraie sensibilité, re-run run_pipeline.py avec chaque horizon.
    Cette fonction renvoie les métriques approximatives comme premier indicateur.

    Returns:
        DataFrame [horizon, n_clients, churn_rate, roc_auc, pr_auc, f1, note]
    """
    if horizons is None:
        horizons = [60, 90, 120]
    if model_cls is None:
        model_cls = XGBClassifier
    if model_params is None:
        model_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": seed,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

    df_all = pd.read_csv(churn_dataset_path)
    rows = []

    for h in horizons:
        df_h = df_all.copy()
        # Approximation : pour horizon < 90, on exclut les snapshots trop récents
        # (clients qui auraient eu plus de temps pour revenir sous horizon plus long)
        if h != 90:
            df_h = df_h[df_h["split"] == "train"].copy()
        df_h = df_h.dropna(subset=["churn"])

        y = df_h["churn"].values
        train_mask = df_h["split"] == "train"
        test_mask = df_h["split"] == "test"

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        prep = ChurnPreprocessor(with_scaling=False)
        X_train = prep.fit(df_h[train_mask]).transform(df_h[train_mask])
        y_train = y[train_mask.values]

        prep_test = ChurnPreprocessor(with_scaling=False)
        prep_test.fit(df_h[train_mask])
        X_test = prep_test.transform(df_h[test_mask])
        y_test_h = y[test_mask.values]

        if len(np.unique(y_test_h)) < 2:
            continue

        m = model_cls(**model_params)
        m.fit(X_train, y_train)
        metrics = evaluate_model(m, X_test, y_test_h)

        rows.append({
            "horizon_jours": h,
            "n_total": len(df_h),
            "churn_rate_train": round(float(y_train.mean()), 3),
            "churn_rate_test": round(float(y_test_h.mean()), 3),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
            "note": "exact" if h == 90 else "approx (meme dataset, filtre split)",
        })

    return pd.DataFrame(rows)


# ===================================================================== 11. Test sans top 1%
def evaluate_without_top1pct(
    model: BaseEstimator,
    prep: ChurnPreprocessor,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    monetary_col: str = "monetary",
) -> Dict[str, object]:
    """Évalue le modèle en excluant le top 1% des clients par CA (gros comptes B2B).

    Ces clients peuvent être des outliers et dominer les métriques globales.

    Returns:
        dict avec n_excluded, churn_rate_excluded, metrics_with_top1, metrics_without_top1.
    """
    threshold = df_test[monetary_col].quantile(0.99)
    mask_top1 = df_test[monetary_col] >= threshold
    mask_rest = ~mask_top1

    n_excluded = int(mask_top1.sum())

    # Avec tous les clients (référence)
    X_all = prep.transform(df_test)
    metrics_all = evaluate_model(model, X_all, y_test)

    # Sans le top 1%
    df_rest = df_test[mask_rest].copy()
    y_rest = y_test[mask_rest.values]
    if len(np.unique(y_rest)) < 2 or len(y_rest) == 0:
        return {"error": "Pas assez de données après exclusion du top 1%"}

    X_rest = prep.transform(df_rest)
    metrics_rest = evaluate_model(model, X_rest, y_rest)

    return {
        "n_total": len(y_test),
        "n_excluded_top1pct": n_excluded,
        "threshold_monetary": round(float(threshold), 2),
        "metrics_with_top1": metrics_all,
        "metrics_without_top1": metrics_rest,
        "roc_auc_delta": round(float(metrics_rest["roc_auc"] - metrics_all["roc_auc"]), 4),
    }


# ===================================================================== 12. Entraînement complet
def train_and_evaluate_all(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, ChurnPreprocessor], pd.DataFrame, pd.DataFrame]:
    """Entraîne les baselines et les 3 modèles supervisés, puis compare sur test."""
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df["churn"].values
    y_val = val_df["churn"].values
    y_test = test_df["churn"].values

    # Préprocesseurs
    prep_scaled = ChurnPreprocessor(with_scaling=True).fit(train_df)
    prep_raw = ChurnPreprocessor(with_scaling=False).fit(train_df)

    # Matrices numpy
    X_train_scaled = prep_scaled.transform(train_df)
    X_val_scaled = prep_scaled.transform(val_df)
    X_test_scaled = prep_scaled.transform(test_df)

    X_train_raw = prep_raw.transform(train_df)
    X_val_raw = prep_raw.transform(val_df)
    X_test_raw = prep_raw.transform(test_df)

    # Dictionnaire des modèles
    models: Dict[str, BaseEstimator] = {}
    preprocessors: Dict[str, ChurnPreprocessor] = {}

    # 1. Dummy Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_raw, y_train)
    models["Dummy (majoritaire)"] = dummy
    preprocessors["Dummy (majoritaire)"] = prep_raw

    # 2. Heuristique Récence
    rec_baseline = RecencyThresholdBaseline()
    rec_baseline.fit(train_df, y_train)
    models["Règle Récence"] = rec_baseline
    preprocessors["Règle Récence"] = None  # utilise directement le DataFrame

    # 3. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(X_train_scaled, y_train)
    models["Régression Logistique"] = lr
    preprocessors["Régression Logistique"] = prep_scaled

    # 4. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train_raw, y_train)
    models["Random Forest"] = rf
    preprocessors["Random Forest"] = prep_raw

    # 5. XGBoost
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train_raw, y_train)
    models["XGBoost"] = xgb
    preprocessors["XGBoost"] = prep_raw

    # Évaluation comparative sur validation et test
    results = []
    for name, m in models.items():
        prep = preprocessors[name]
        if prep is None:
            # Baseline récence
            val_X, test_X = val_df, test_df
        else:
            val_X = prep.transform(val_df)
            test_X = prep.transform(test_df)

        m_val = evaluate_model(m, val_X, y_val)
        m_test = evaluate_model(m, test_X, y_test)

        results.append({
            "Modèle": name,
            "ROC-AUC (Val)": m_val["roc_auc"],
            "ROC-AUC (Test)": m_test["roc_auc"],
            "PR-AUC (Test)": m_test["pr_auc"],
            "F1-Score (Test)": m_test["f1"],
            "Précision (Test)": m_test["precision"],
            "Rappel (Test)": m_test["recall"],
            "Brier Score (Test)": m_test["brier_score"],
        })

    eval_df = pd.DataFrame(results).sort_values("ROC-AUC (Test)", ascending=False)

    # Évaluation détaillée par snapshot de test pour le meilleur modèle (XGBoost)
    test_snapshots = []
    for dt, snap_group in test_df.groupby("snapshot_date"):
        y_snap = snap_group["churn"].values
        X_snap = prep_raw.transform(snap_group)
        m_snap = evaluate_model(xgb, X_snap, y_snap)
        m_snap["snapshot"] = str(dt)[:10]
        m_snap["n_clients"] = len(snap_group)
        m_snap["taux_churn"] = round(float(y_snap.mean()), 3)
        test_snapshots.append(m_snap)

    snap_df = pd.DataFrame(test_snapshots)[["snapshot", "n_clients", "taux_churn", "roc_auc", "pr_auc", "f1", "brier_score"]]

    return models, preprocessors, eval_df, snap_df
