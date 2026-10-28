"""Execute les phases 2 a 4 de bout en bout : audit -> nettoyage -> features/cible.

Usage :
    python run_pipeline.py                       # utilise config.yaml (data/raw/online_retail_II.xlsx)
    python run_pipeline.py --raw data/raw/sample_retail.xlsx   # test avec les donnees synthetiques

Sorties :
    data/interim/raw_merged.parquet        cache du fichier brut fusionne
    data/processed/transactions_clean.*    transactions nettoyees
    data/processed/customer_features.*     table client pour la segmentation
    data/processed/churn_dataset.*         table (client x snapshot) pour le churn
    outputs/exclusion_log.csv              tableau 'lignes exclues et pourquoi'
    outputs/audit_summary.csv              chiffres cles de qualite
"""
import argparse
import time

from src.cleaning import audit_summary, clean, load_raw
from src.config import ROOT, load_config, resolve, save_table
from src.features import build_churn_dataset, build_customer_features, check_split_purge, snapshot_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="chemin du fichier brut (remplace config.paths.raw_file)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    raw = ROOT / (a.raw or cfg["paths"]["raw_file"])
    cache = None if a.no_cache else resolve(cfg, "interim_dir") / "raw_merged.parquet"
    out_dir = resolve(cfg, "outputs_dir")

    t0 = time.time()
    print("[Phase 2] Chargement et audit ...")
    if cache is not None and a.raw is not None and cache.exists():
        cache.unlink()  # un fichier different a ete passe : on ne reutilise pas un ancien cache
    df_raw = load_raw(raw, cache)
    audit_summary(df_raw, cfg).to_csv(out_dir / "audit_summary.csv", index=False)
    print(f"   {len(df_raw):,} lignes brutes ({time.time() - t0:.0f}s)")

    print("[Phase 3] Nettoyage ...")
    df, log = clean(df_raw, cfg)
    log.to_csv(out_dir / "exclusion_log.csv", index=False)
    print(log[["etape", "regle", "lignes_exclues", "% du brut"]].to_string(index=False))
    print("   ->", save_table(df, "transactions_clean", cfg))

    print("[Phase 4] Features et cible ...")
    horizon = cfg["churn"]["horizon_days"]
    sched = snapshot_schedule(cfg, df, horizon)
    print(sched.to_string(index=False))
    problems = check_split_purge(sched)
    if problems:
        raise SystemExit("\n".join(problems))
    seg = build_customer_features(df, cfg)
    print("   ->", save_table(seg, "customer_features", cfg), f"({len(seg):,} clients)")
    churn = build_churn_dataset(df, cfg, horizon)
    print("   ->", save_table(churn, "churn_dataset", cfg), f"({len(churn):,} lignes client x snapshot)")
    print(churn.groupby("split")["churn"].agg(["size", "mean"]).rename(columns={"size": "lignes", "mean": "taux_churn"}).round(3))
    print(f"Termine en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
