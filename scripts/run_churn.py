"""Script d'exécution complet pour la Phase 6 : Modélisation et Prédiction de Churn.

Ce script produit TOUS les livrables du cahier des charges Phase 6 :
  - churn_predictions.csv (tous les clients : train+val+test, avec risk_class et top 3 facteurs)
  - churn_model.joblib
  - churn_metrics.json (enrichi : GroupKFold, IC bootstrap, UK/hors-UK, coût métier, sensibilité)
  - 10 graphiques dans reports/figures/
  - Prêt pour le notebook 05_churn_prediction.ipynb
"""
import json
import sys
import warnings
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
from xgboost import XGBClassifier

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
from src.config import load_config, resolve


def main():
    cfg = load_config()
    FIG = resolve(cfg, "figures_dir")
    MODELS = ROOT / "models"
    MODELS.mkdir(parents=True, exist_ok=True)
    SEED = cfg.get("seed", 42)
    COST_FP = 5.0    # coût d'une campagne envoyée à un non-churn (€)
    COST_FN = 50.0   # valeur perdue d'un churn non détecté (€)

    print("=" * 65)
    print("PHASE 6 : PRÉDICTION DE CHURN (Modélisation supervisée)")
    print("=" * 65)

    # ---------------------------------------------------------------- 0. Chargement
    data_path = ROOT / "data/processed/churn_dataset.csv"
    seg_path = ROOT / "data/processed/customer_segments.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} introuvable. Lancez d'abord run_pipeline.py.")

    print("\n0. Chargement des données...")
    df = pd.read_csv(data_path)
    print(f"   churn_dataset.csv : {len(df):,} lignes | {df.shape[1]} colonnes")
    for split_name, g in df.groupby("split"):
        print(f"   - {split_name:10s} : {len(g):5,d} lignes | Taux churn : {g['churn'].mean():.1%}")

    # Jointure avec les segments Phase 5 (enrichissement Phase 5→6)
    if seg_path.exists():
        seg_df = pd.read_csv(seg_path)[["CustomerID", "segment_id", "segment_name"]]
        df = df.merge(seg_df, on="CustomerID", how="left")
        df["segment_id"] = df["segment_id"].fillna(-1).astype(int)
        print(f"   + Segments Phase 5 joints : {df['segment_name'].notna().sum():,} clients enrichis")
    else:
        print("   ! customer_segments.csv absent — liaison Phase 5 ignorée")

    # ---------------------------------------------------------------- 1. Entraînement
    print("\n1. Entraînement des modèles et des baselines...")
    models, preprocessors, eval_df, snap_df = train_and_evaluate_all(df, seed=SEED)

    print("\n=== BENCHMARK DES MODÈLES SUR LE JEU DE TEST TEMPOREL ===")
    print(eval_df.to_string(index=False))

    print("\n=== PERFORMANCE DÉTAILLÉE PAR SNAPSHOT DE TEST (XGBoost) ===")
    print(snap_df.to_string(index=False))

    # Modèle principal : XGBoost
    best_name = "XGBoost"
    best_model = models[best_name]
    best_prep = preprocessors[best_name]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    y_test = test_df["churn"].values
    X_test_prep = best_prep.transform(test_df)
    y_probs_xgb = best_model.predict_proba(X_test_prep)[:, 1]

    # ---------------------------------------------------------------- 2. GroupKFold
    print("\n2. Validation croisée GroupKFold (n_splits=5, groupé par CustomerID)...")
    scale_pos = (len(train_df["churn"]) - train_df["churn"].sum()) / train_df["churn"].sum()
    gkf_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": SEED,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "scale_pos_weight": float(scale_pos),
    }
    gkf_results = group_k_fold_cv(
        df=df[df["split"] == "train"].copy(),
        model_cls=XGBClassifier,
        model_params=gkf_params,
        prep_cls=ChurnPreprocessor,
        prep_params={"with_scaling": False},
        n_splits=5,
        seed=SEED,
    )
    print(f"   ROC-AUC GroupKFold : {gkf_results['mean_metrics']['roc_auc']:.4f} ± {gkf_results['mean_metrics']['roc_auc_std']:.4f}")
    print(f"   PR-AUC  GroupKFold : {gkf_results['mean_metrics']['pr_auc']:.4f} ± {gkf_results['mean_metrics']['pr_auc_std']:.4f}")
    print(f"   F1      GroupKFold : {gkf_results['mean_metrics']['f1']:.4f} ± {gkf_results['mean_metrics']['f1_std']:.4f}")

    # ---------------------------------------------------------------- 3. Bootstrap IC 95%
    print("\n3. Intervalles de confiance bootstrap (n=200) sur le jeu de test...")
    ci_results = bootstrap_confidence_intervals(
        model=best_model,
        prep=best_prep,
        df_test=test_df,
        y_test=y_test,
        n_boot=200,
        seed=SEED,
    )
    roc_ci = ci_results["roc_auc"]
    pr_ci = ci_results["pr_auc"]
    print(f"   ROC-AUC : {roc_ci[1]:.4f}  IC 95% [{roc_ci[0]:.4f}, {roc_ci[2]:.4f}]")
    print(f"   PR-AUC  : {pr_ci[1]:.4f}  IC 95% [{pr_ci[0]:.4f}, {pr_ci[2]:.4f}]")

    # ---------------------------------------------------------------- 4. UK vs hors-UK
    print("\n4. Analyse UK vs Hors-UK...")
    uk_df = evaluate_uk_nonuk(best_model, best_prep, test_df, y_test)
    print(uk_df.to_string(index=False))

    # ---------------------------------------------------------------- 5. Seuil coût métier
    print(f"\n5. Seuil optimal par analyse coût métier (FP={COST_FP}€ / FN={COST_FN}€)...")
    cost_result = compute_cost_threshold(y_probs_xgb, y_test, cost_fp=COST_FP, cost_fn=COST_FN)
    opt_threshold = cost_result["threshold_optimal"]
    print(f"   Seuil optimal : {opt_threshold:.2f}")
    print(f"   Coût total optimal : {cost_result['total_cost_optimal']:,.0f} €")
    print(f"   Économies vs aucune action : {cost_result['savings_vs_no_action']:,.0f} €")
    print(f"   FP : {cost_result['n_fp']} | FN : {cost_result['n_fn']} | TP : {cost_result['n_tp']}")

    # ---------------------------------------------------------------- 6. Lift par décile
    print("\n6. Lift et Gain cumulé par décile (XGBoost)...")
    lift_df = compute_decile_lift(y_test, y_probs_xgb)
    print(lift_df.to_string(index=False))

    # ---------------------------------------------------------------- 7. SHAP
    print("\n7. Calcul des valeurs SHAP (explicabilité)...")
    feat_names = best_prep.feature_names_
    _, shap_vals, shap_idx = compute_shap_values(
        best_model, X_test_prep, feat_names, max_samples=500, seed=SEED
    )
    shap_mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_importance = pd.Series(shap_mean_abs, index=feat_names).sort_values(ascending=False)
    print("   Top 10 SHAP features :")
    for feat, val in shap_importance.head(10).items():
        print(f"     {feat:35s} : {val:.4f}")

    # ---------------------------------------------------------------- 8. Test sans top 1%
    print("\n8. Robustesse : évaluation sans le top 1% des clients (gros B2B)...")
    if "monetary" in test_df.columns:
        top1_result = evaluate_without_top1pct(best_model, best_prep, test_df, y_test)
        print(f"   Clients exclus : {top1_result['n_excluded_top1pct']} (seuil CA = {top1_result['threshold_monetary']:,.0f})")
        print(f"   ROC-AUC avec top1% : {top1_result['metrics_with_top1']['roc_auc']:.4f}")
        print(f"   ROC-AUC sans top1% : {top1_result['metrics_without_top1']['roc_auc']:.4f}")
        print(f"   Delta : {top1_result['roc_auc_delta']:+.4f}")
    else:
        top1_result = {"note": "colonne monetary absente"}

    # ---------------------------------------------------------------- 9. Sensibilité horizon
    print("\n9. Analyse de sensibilité : horizons 60 / 90 / 120 jours...")
    horizons = cfg.get("churn", {}).get("sensitivity_horizons", [60, 90, 120])
    sens_df = sensitivity_analysis(data_path, horizons=horizons, seed=SEED)
    print(sens_df.to_string(index=False))

    # ---------------------------------------------------------------- 10. Graphiques
    print("\n10. Génération des graphiques dans reports/figures/...")

    # A. Courbes ROC comparatives
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in models.items():
        prep = preprocessors[name]
        if prep is None:
            probs = m.predict_proba(test_df)[:, 1]
        else:
            probs = m.predict_proba(prep.transform(test_df))[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = eval_df.loc[eval_df["Modèle"] == name, "ROC-AUC (Test)"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Hasard pur (0.50)")
    ax.axhline(0.75, color="red", ls=":", alpha=0.5, label="Cible minimale (>0.75)")
    ax.set_title("Courbes ROC — Test Temporel sans fuite")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "churn_roc_curves.png", dpi=150)
    plt.close()

    # B. Courbes Precision-Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["Règle Récence", "Régression Logistique", "Random Forest", "XGBoost"]:
        m = models[name]
        prep = preprocessors[name]
        probs = m.predict_proba(test_df if prep is None else prep.transform(test_df))[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, probs)
        score = eval_df.loc[eval_df["Modèle"] == name, "PR-AUC (Test)"].iloc[0]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {score:.3f})")
    ax.set_title("Courbes Precision-Recall — Test Temporel")
    ax.set_xlabel("Rappel (Recall)")
    ax.set_ylabel("Précision")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "churn_pr_curves.png", dpi=150)
    plt.close()

    # C. Courbe de calibration
    fig, ax = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_probs_xgb, n_bins=10)
    ax.plot(prob_pred, prob_true, marker="o", label="XGBoost", color="#2b6cb0")
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    brier = eval_df.loc[eval_df["Modèle"] == "XGBoost", "Brier Score (Test)"].iloc[0]
    ax.set_title(f"Courbe de calibration XGBoost (Brier Score = {brier:.4f})")
    ax.set_xlabel("Probabilité moyenne prédite")
    ax.set_ylabel("Fraction de réels churners")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "churn_calibration.png", dpi=150)
    plt.close()

    # D. Gain cumulé et Lift par décile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lift_df["decile"], lift_df["cum_gain_pct"], marker="o", color="#2b6cb0", label="XGBoost")
    axes[0].plot([1, 10], [10, 100], "k--", label="Ciblage aléatoire")
    axes[0].set_title("Gain cumulé par décile (%)")
    axes[0].set_xlabel("Décile de risque (1 = risque le plus élevé)")
    axes[0].set_ylabel("% du total des churners capturés")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend()

    axes[1].bar(lift_df["decile"], lift_df["lift"], color="#4299e1", edgecolor="#2b6cb0")
    axes[1].axhline(1.0, color="red", ls="--", label="Lift neutre (1.0)")
    axes[1].set_title("Lift par décile")
    axes[1].set_xlabel("Décile de risque")
    axes[1].set_ylabel("Multiplicateur de risque vs moyenne")
    axes[1].set_xticks(range(1, 11))
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "churn_decile_lift.png", dpi=150)
    plt.close()

    # E. Feature Importance XGBoost (Top 15)
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp[::-1].plot(kind="barh", ax=ax, color="#3182ce")
    ax.set_title("Top 15 Variables prédictives du Churn (XGBoost Feature Importance)")
    ax.set_xlabel("Importance relative (Gain)")
    plt.tight_layout()
    plt.savefig(FIG / "churn_feature_importance.png", dpi=150)
    plt.close()

    # F. SHAP summary (barplot des moyennes |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_importance.head(15)[::-1].plot(kind="barh", ax=ax, color="#e53e3e")
    ax.set_title("Top 15 Variables — Valeurs SHAP moyennes |∆ probabilité|")
    ax.set_xlabel("Impact moyen absolu sur la probabilité de churn")
    plt.tight_layout()
    plt.savefig(FIG / "churn_shap_summary.png", dpi=150)
    plt.close()

    # G. Matrice de confusion au seuil optimal
    y_preds_opt = (y_probs_xgb >= opt_threshold).astype(int)
    cm = confusion_matrix(y_test, y_preds_opt)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Churn (0)", "Churn (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de Confusion XGBoost (seuil={opt_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(FIG / "churn_confusion_matrix.png", dpi=150)
    plt.close()

    # H. Courbe coût métier
    cost_curve = cost_result["cost_curve"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cost_curve["threshold"], cost_curve["total_cost"], color="#2b6cb0", lw=2)
    ax.axvline(opt_threshold, color="red", ls="--",
               label=f"Seuil optimal = {opt_threshold:.2f} (coût = {cost_result['total_cost_optimal']:,.0f}€)")
    ax.axhline(cost_result["cost_no_action"], color="orange", ls=":",
               label=f"Sans modèle = {cost_result['cost_no_action']:,.0f}€")
    ax.set_title(f"Coût métier par seuil (FP={COST_FP}€ / FN={COST_FN}€)")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût total (€)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "churn_cost_threshold.png", dpi=150)
    plt.close()

    # I. UK vs hors-UK barplot
    if len(uk_df) >= 2 and "roc_auc" in uk_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        uk_plot = uk_df[uk_df["group"].isin(["UK", "Hors-UK"])]
        bars = ax.bar(uk_plot["group"], uk_plot["roc_auc"], color=["#4299e1", "#48bb78"])
        ax.set_title("ROC-AUC XGBoost : UK vs Hors-UK")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0, 1)
        ax.axhline(0.75, color="red", ls="--", label="Objectif >0.75")
        for bar, val in zip(bars, uk_plot["roc_auc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG / "churn_uk_vs_nonuk.png", dpi=150)
        plt.close()

    # J. Sensibilité aux horizons
    if len(sens_df) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sens_df["horizon_jours"], sens_df["roc_auc"], marker="o",
                color="#2b6cb0", label="ROC-AUC")
        ax.plot(sens_df["horizon_jours"], sens_df["pr_auc"], marker="s",
                color="#38a169", label="PR-AUC")
        ax.axhline(0.75, color="red", ls="--", alpha=0.5, label="Objectif >0.75")
        ax.set_title("Sensibilité aux horizons de churn (60 / 90 / 120 jours)")
        ax.set_xlabel("Horizon churn (jours)")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG / "churn_sensitivity.png", dpi=150)
        plt.close()

    print(f"   10 graphiques sauvegardés dans {FIG}")

    # ---------------------------------------------------------------- 11. Export prédictions complètes
    print("\n11. Construction de churn_predictions.csv (tous les clients)...")

    all_preds_parts = []
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split_df = split_df.copy()
        X_split = best_prep.transform(split_df)
        probs_split = best_model.predict_proba(X_split)[:, 1]

        # SHAP sur cet ensemble (pour top 3 facteurs)
        _, shap_split, shap_split_idx = compute_shap_values(
            best_model, X_split, feat_names, max_samples=len(split_df), seed=SEED
        )

        part = split_df[["CustomerID", "snapshot_date", "split", "churn"]].copy()
        if "is_uk" in split_df.columns:
            part["is_uk"] = split_df["is_uk"].values
        if "segment_name" in split_df.columns:
            part["segment_name"] = split_df["segment_name"].values
        if "recency" in split_df.columns:
            part["recency"] = split_df["recency"].values
        if "monetary" in split_df.columns:
            part["monetary"] = split_df["monetary"].values
        if "frequency" in split_df.columns:
            part["frequency"] = split_df["frequency"].values

        part["churn_risk_score"] = np.round(probs_split, 4)
        part["churn_predicted"] = (part["churn_risk_score"] >= opt_threshold).astype(int)
        part["risk_decile"] = pd.qcut(
            part["churn_risk_score"], q=10,
            labels=list(range(10, 0, -1)),
            duplicates="drop"
        )

        # risk_class + top 3 facteurs SHAP
        part = add_risk_class(part, shap_split, feat_names, score_col="churn_risk_score")
        all_preds_parts.append(part)

    churn_preds_all = pd.concat(all_preds_parts, ignore_index=True)
    pred_out = ROOT / "data/processed/churn_predictions.csv"
    churn_preds_all.to_csv(pred_out, index=False)
    print(f"    -> {pred_out} ({len(churn_preds_all):,} lignes)")
    print(f"    Classes de risque : {churn_preds_all['risk_class'].value_counts().to_dict()}")

    # ---------------------------------------------------------------- 12. Sauvegarde artefacts
    print("\n12. Sauvegarde des artefacts...")

    artifact_payload = {
        "model": best_model,
        "preprocessor": best_prep,
        "feature_names": feat_names,
        "metrics_summary": eval_df.to_dict(orient="records"),
        "cost_threshold": opt_threshold,
        "risk_thresholds": [0.30, 0.60],
    }
    model_out = MODELS / "churn_model.joblib"
    joblib.dump(artifact_payload, model_out)
    print(f"    -> Modèle sauvegardé : {model_out}")

    metrics_out = MODELS / "churn_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_evaluation": eval_df.to_dict(orient="records"),
                "snapshot_evaluation": snap_df.to_dict(orient="records"),
                "group_kfold_cv": {
                    "mean_metrics": gkf_results["mean_metrics"],
                    "n_splits": gkf_results["n_splits"],
                    "n_clients": gkf_results["n_clients_total"],
                },
                "bootstrap_confidence_intervals_95pct": {
                    "roc_auc": {"lower": roc_ci[0], "mean": roc_ci[1], "upper": roc_ci[2]},
                    "pr_auc": {"lower": pr_ci[0], "mean": pr_ci[1], "upper": pr_ci[2]},
                    "n_bootstrap": ci_results["n_bootstrap"],
                },
                "uk_nonuk_evaluation": uk_df.to_dict(orient="records"),
                "cost_threshold_analysis": {
                    "threshold_optimal": opt_threshold,
                    "total_cost_optimal_eur": cost_result["total_cost_optimal"],
                    "cost_no_action_eur": cost_result["cost_no_action"],
                    "savings_eur": cost_result["savings_vs_no_action"],
                    "cost_fp_eur": COST_FP,
                    "cost_fn_eur": COST_FN,
                    "n_fp": cost_result["n_fp"],
                    "n_fn": cost_result["n_fn"],
                    "n_tp": cost_result["n_tp"],
                },
                "sensitivity_horizons": sens_df.to_dict(orient="records"),
                "top1pct_robustness": top1_result if isinstance(top1_result, dict) else str(top1_result),
                "decile_lift": lift_df.to_dict(orient="records"),
                "shap_top10_features": shap_importance.head(10).round(4).to_dict(),
                "risk_class_distribution": churn_preds_all["risk_class"].value_counts().to_dict(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"    -> Métriques sauvegardées : {metrics_out}")

    # ---------------------------------------------------------------- Résumé final
    print("\n" + "=" * 65)
    print("=== RÉSUMÉ PHASE 6 ===")
    print("=" * 65)
    print(f"Meilleur modèle   : {best_name}")
    print(f"ROC-AUC Test      : {eval_df.loc[eval_df['Modèle']==best_name, 'ROC-AUC (Test)'].iloc[0]:.4f}  "
          f"IC95% [{roc_ci[0]:.4f}, {roc_ci[2]:.4f}]")
    print(f"PR-AUC  Test      : {eval_df.loc[eval_df['Modèle']==best_name, 'PR-AUC (Test)'].iloc[0]:.4f}  "
          f"IC95% [{pr_ci[0]:.4f}, {pr_ci[2]:.4f}]")
    print(f"F1-Score Test     : {eval_df.loc[eval_df['Modèle']==best_name, 'F1-Score (Test)'].iloc[0]:.4f}")
    print(f"GroupKFold ROC-AUC: {gkf_results['mean_metrics']['roc_auc']:.4f} ± {gkf_results['mean_metrics']['roc_auc_std']:.4f}")
    print(f"Seuil coût métier : {opt_threshold:.2f} (économies = {cost_result['savings_vs_no_action']:,.0f}€)")
    print(f"Clients exportés  : {len(churn_preds_all):,} (train+val+test)")
    print(f"\nLivrables générés :")
    print(f"  data/processed/churn_predictions.csv  ({len(churn_preds_all):,} lignes)")
    print(f"  models/churn_model.joblib")
    print(f"  models/churn_metrics.json")
    print(f"  reports/figures/churn_*.png (10 figures)")
    print("\nPhase 6 terminée avec succès. ✓")


if __name__ == "__main__":
    main()
