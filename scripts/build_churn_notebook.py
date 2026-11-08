"""Générateur du notebook 05_churn_prediction.ipynb sans dépendance externe (pur JSON)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def make_cell(cell_type: str, source: str):
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = [
    make_cell("markdown", """# 05 - Prédiction de Churn (Phase 6)

**Objectif :** Anticiper le risque de churn des clients sur un horizon futur de 90 jours sans fuite temporelle.

> **Principe clé anti-fuite :**
> - Features calculées strictement sur le passé `< T`
> - Cible churn calculée sur la fenêtre future `[T, T + 90j)`
> - Purge stricte : la fenêtre de cible du train se termine avant le premier snapshot de validation
> - Liaison Phase 5 : les segments K-Means sont joints comme variable explicative supplémentaire

**Plan du notebook :**
1. Chargement et audit des données (splits, taux de churn)
2. Baseline de référence (DummyClassifier, Règle Récence)
3. Modèles de Machine Learning (Régression Logistique, Random Forest, XGBoost)
4. Validation croisée GroupKFold (groupée par client pour éviter toute fuite intra-client)
5. Benchmark comparatif sur le jeu de test temporel (ROC-AUC, PR-AUC, F1, Brier Score)
6. Intervalles de confiance bootstrap à 95%
7. Analyse d'équité géographique (UK vs Hors-UK)
8. Optimisation du seuil de décision par coût métier
9. Interprétabilité SHAP et Feature Importance
10. Analyse de robustesse (sans top 1% B2B) et sensibilité aux horizons
11. Export final des prédictions (churn_predictions.csv)
"""),
    make_cell("code", """import sys
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.churn import (
    ChurnPreprocessor,
    RecencyThresholdBaseline,
    add_risk_class,
    bootstrap_confidence_intervals,
    compute_cost_threshold,
    compute_decile_lift,
    compute_shap_values,
    evaluate_model,
    evaluate_uk_nonuk,
    evaluate_without_top1pct,
    group_k_fold_cv,
    sensitivity_analysis,
    train_and_evaluate_all,
)

cfg = load_config()
FIG = resolve(cfg, 'figures_dir')
plt.rcParams.update({'figure.figsize': (8, 5), 'axes.grid': True, 'grid.alpha': 0.3})
print("Environnement Phase 6 prêt.")
"""),
    make_cell("markdown", "## 1. Chargement du dataset churn et des segments Phase 5"),
    make_cell("code", """df = pd.read_csv(ROOT / 'data/processed/churn_dataset.csv')
print(f"Total lignes : {len(df):,} | Colonnes : {df.shape[1]}")

# Jointure avec les segments Phase 5 (enrichissement)
seg_path = ROOT / 'data/processed/customer_segments.csv'
if seg_path.exists():
    seg_df = pd.read_csv(seg_path)[['CustomerID', 'segment_id', 'segment_name']]
    df = df.merge(seg_df, on='CustomerID', how='left')
    df['segment_id'] = df['segment_id'].fillna(-1).astype(int)
    print(f"+ Segments Phase 5 joints avec succes ({len(seg_df):,} clients).")

for s, g in df.groupby('split'):
    print(f"  Split {s:10s} : {len(g):5,d} lignes | Taux churn : {g['churn'].mean():.1%}")

df.head()
"""),
    make_cell("markdown", "## 2. Entraînement des modèles et Benchmark comparatif"),
    make_cell("code", """models, preprocessors, eval_df, snap_df = train_and_evaluate_all(df, seed=cfg.get('seed', 42))
print("Benchmark sur le jeu de test temporel :")
eval_df
"""),
    make_cell("markdown", "## 3. Performance par Snapshot de Test"),
    make_cell("code", """print("Performances XGBoost par snapshot de test :")
snap_df
"""),
    make_cell("markdown", "## 4. Validation Croisée GroupKFold par Client"),
    make_cell("code", """from xgboost import XGBClassifier
train_df = df[df['split'] == 'train'].copy()
scale_pos = (len(train_df['churn']) - train_df['churn'].sum()) / train_df['churn'].sum()

gkf = group_k_fold_cv(
    df=train_df,
    model_cls=XGBClassifier,
    model_params={'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05,
                  'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1,
                  'scale_pos_weight': float(scale_pos)},
    prep_cls=ChurnPreprocessor,
    prep_params={'with_scaling': False},
    n_splits=5,
    seed=42,
)
print("Resultats GroupKFold (5 folds) :")
for k, v in gkf['mean_metrics'].items():
    print(f"  {k:15s} : {v}")
"""),
    make_cell("markdown", "## 5. Intervalles de Confiance Bootstrap à 95%"),
    make_cell("code", """test_df = df[df['split'] == 'test'].copy()
y_test = test_df['churn'].values
best_model = models['XGBoost']
best_prep = preprocessors['XGBoost']

ci = bootstrap_confidence_intervals(best_model, best_prep, test_df, y_test, n_boot=200, seed=42)
print(f"ROC-AUC Test : {ci['roc_auc'][1]:.4f}  IC 95% [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][2]:.4f}]")
print(f"PR-AUC  Test : {ci['pr_auc'][1]:.4f}  IC 95% [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][2]:.4f}]")
"""),
    make_cell("markdown", "## 6. Analyse d'Équité Géographique : UK vs Hors-UK"),
    make_cell("code", """uk_df = evaluate_uk_nonuk(best_model, best_prep, test_df, y_test)
uk_df
"""),
    make_cell("markdown", "## 7. Optimisation du Seuil par Coût Métier"),
    make_cell("code", """X_test = best_prep.transform(test_df)
y_probs = best_model.predict_proba(X_test)[:, 1]

cost_res = compute_cost_threshold(y_probs, y_test, cost_fp=5.0, cost_fn=50.0)
print(f"Seuil optimal de decision : {cost_res['threshold_optimal']:.2f}")
print(f"Cout optimal total        : {cost_res['total_cost_optimal']:,.0f} €")
print(f"Economies realisees       : {cost_res['savings_vs_no_action']:,.0f} €")
print(f"Matrice : FP={cost_res['n_fp']}, FN={cost_res['n_fn']}, TP={cost_res['n_tp']}")
"""),
    make_cell("markdown", "## 8. Lift et Gain Cumulé par Décile"),
    make_cell("code", """lift_df = compute_decile_lift(y_test, y_probs)
lift_df
"""),
    make_cell("markdown", "## 9. Interprétabilité : SHAP et Top 15 Variables"),
    make_cell("code", """_, shap_vals, _ = compute_shap_values(best_model, X_test, best_prep.feature_names_, max_samples=500, seed=42)
shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=best_prep.feature_names_).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
shap_imp.head(15)[::-1].plot(kind='barh', ax=ax, color='#e53e3e')
ax.set_title("Top 15 Variables - Impact SHAP moyen (|delta proba|)")
ax.set_xlabel("Impact absolu moyen")
plt.tight_layout()
plt.show()
"""),
    make_cell("markdown", "## 10. Robustesse (sans top 1% B2B) et Sensibilité"),
    make_cell("code", """# Test sans le top 1% des plus gros clients
rob = evaluate_without_top1pct(best_model, best_prep, test_df, y_test)
print(f"ROC-AUC avec top 1% : {rob['metrics_with_top1']['roc_auc']:.4f}")
print(f"ROC-AUC sans top 1% : {rob['metrics_without_top1']['roc_auc']:.4f}")
print(f"Delta               : {rob['roc_auc_delta']:+.4f}")
"""),
    make_cell("markdown", "## 11. Export des Prédictions Enrichies"),
    make_cell("code", """pred_path = ROOT / 'data/processed/churn_predictions.csv'
if pred_path.exists():
    preds = pd.read_csv(pred_path)
    print(f"churn_predictions.csv charge : {len(preds):,} lignes")
    print("\nRepartition des classes de risque :")
    print(preds['risk_class'].value_counts())
    print("\nExemple de lignes avec top 3 facteurs d'explication :")
    display(preds[['CustomerID', 'churn_risk_score', 'risk_class', 'top_factor_1', 'top_factor_2', 'top_factor_3']].head(10))
""")
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = ROOT / "notebooks/05_churn_prediction.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook généré avec succès -> {out_path}")
