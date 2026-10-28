"""Genere toutes les figures de l'EDA et de l'analyse dans reports/figures/."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # Mode sans fenetre graphique
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve, load_table
from src.features import snapshot_schedule, check_split_purge

def main():
    cfg = load_config()
    FIG = resolve(cfg, "figures_dir")
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3})

    print("Chargement des transactions...")
    df = load_table("transactions_clean", cfg)
    sales = df[~df["IsReturn"]]

    print("1. Generation ca_mensuel.png...")
    m = df.groupby("YearMonth").agg(ca_net=("Revenue", "sum"))
    fig, ax = plt.subplots()
    m["ca_net"].plot(kind="bar", ax=ax, color="#2b6cb0")
    ax.set_title("CA net mensuel (GBP)")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(FIG / "ca_mensuel.png", dpi=150)
    plt.close()

    print("2. Generation saisonnalite.png...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    sales.groupby("Month")["Revenue"].sum().plot(kind="bar", ax=axes[0], title="CA par mois de l'annee")
    sales.groupby("DayOfWeek")["Invoice"].nunique().plot(kind="bar", ax=axes[1], title="Commandes par jour (0 = lundi)")
    sales.groupby("Hour")["Invoice"].nunique().plot(kind="bar", ax=axes[2], title="Commandes par heure")
    plt.tight_layout()
    plt.savefig(FIG / "saisonnalite.png", dpi=150)
    plt.close()

    print("3. Generation top_produits.png...")
    top_q = sales.groupby("Description")["Quantity"].sum().nlargest(10)
    top_r = sales.groupby("Description")["Revenue"].sum().nlargest(10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    top_q[::-1].plot(kind="barh", ax=axes[0], title="Top 10 par quantite")
    top_r[::-1].plot(kind="barh", ax=axes[1], title="Top 10 par CA")
    plt.tight_layout()
    plt.savefig(FIG / "top_produits.png", dpi=150)
    plt.close()

    print("4. Generation paniers.png...")
    basket = sales.groupby("Invoice")["Revenue"].sum()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    basket.clip(upper=basket.quantile(0.99)).plot(kind="hist", bins=60, ax=ax[0], title="Panier (tronque au 99e percentile)")
    np.log10(basket[basket > 0]).plot(kind="hist", bins=60, ax=ax[1], title="log10(panier) : distribution tres asymetrique")
    plt.tight_layout()
    plt.savefig(FIG / "paniers.png", dpi=150)
    plt.close()

    print("5. Generation pareto.png...")
    cust = df[df["HasCustomer"] & ~df["IsReturn"]].groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
    cum = cust.cumsum() / cust.sum()
    share_x = np.arange(1, len(cust) + 1) / len(cust)
    fig, ax = plt.subplots()
    ax.plot(share_x * 100, cum.values * 100)
    ax.axhline(80, color="grey", ls="--")
    ax.set_xlabel("% des clients (du plus gros au plus petit)")
    ax.set_ylabel("% du CA cumule")
    ax.set_title("Courbe de Pareto")
    plt.savefig(FIG / "pareto.png", dpi=150)
    plt.close()

    print("6. Generation delais_inter_achats.png...")
    days = sales[sales["HasCustomer"]][["CustomerID", "InvoiceDay"]].drop_duplicates().sort_values(["CustomerID", "InvoiceDay"])
    gaps = days.groupby("CustomerID")["InvoiceDay"].diff().dt.days.dropna()
    fig, ax = plt.subplots()
    gaps.clip(upper=365).plot(kind="hist", bins=73, ax=ax)
    for p, col in [(0.8, "red"), (0.9, "orange")]:
        ax.axvline(gaps.quantile(p), color=col, ls="--", label=f"p{int(p*100)} = {gaps.quantile(p):.0f} j")
    ax.axvline(cfg["churn"]["horizon_days"], color="green", label=f"horizon choisi = {cfg['churn']['horizon_days']} j")
    ax.legend()
    ax.set_title("Delais entre achats consecutifs (tronque a 365 j)")
    ax.set_xlabel("jours")
    plt.savefig(FIG / "delais_inter_achats.png", dpi=150)
    plt.close()

    print("7. Generation cohortes.png...")
    first = sales[sales["HasCustomer"]].groupby("CustomerID")["InvoiceDay"].min().dt.to_period("M").rename("cohort")
    act = sales[sales["HasCustomer"]].merge(first, on="CustomerID")
    act["period"] = act["InvoiceDay"].dt.to_period("M")
    act["age"] = (act["period"] - act["cohort"]).apply(lambda x: x.n)
    coh = act.groupby(["cohort", "age"])["CustomerID"].nunique().unstack(fill_value=0)
    ret = coh.div(coh[0], axis=0).round(3)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(ret.iloc[:, :13].values, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(ret)))
    ax.set_yticklabels(ret.index.astype(str))
    ax.set_xlabel("mois depuis la premiere commande")
    ax.set_title("Retention par cohorte (part des clients actifs)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIG / "cohortes.png", dpi=150)
    plt.close()

    print("8. Generation profil_b2b.png...")
    c = sales[sales["HasCustomer"]].groupby("CustomerID").agg(qty=("Quantity", "sum"), lines=("StockCode", "size"))
    c["avg_qty_per_line"] = c["qty"] / c["lines"]
    thr = cfg["features"]["wholesale_avg_qty_per_line"]
    fig, ax = plt.subplots()
    np.log10(c["avg_qty_per_line"]).plot(kind="hist", bins=60, ax=ax)
    ax.axvline(np.log10(thr), color="red", ls="--", label=f"seuil = {thr}")
    ax.legend()
    ax.set_title("log10 de la quantite moyenne par ligne, par client")
    plt.savefig(FIG / "profil_b2b.png", dpi=150)
    plt.close()

    print("9. Generation calendrier_snapshots.png...")
    horizon = cfg["churn"]["horizon_days"]
    sched = snapshot_schedule(cfg, df, horizon)
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"train": "#2b6cb0", "validation": "#d69e2e", "test": "#c53030"}
    for i, r in sched.iterrows():
        ax.barh(i, (r["label_end"] - r["T"]).days, left=r["T"], color=colors[r["split"]], alpha=0.8)
        ax.plot(r["T"], i, "k|", ms=12)
    ax.set_yticks(range(len(sched)))
    ax.set_yticklabels([f"T={t:%Y-%m-%d} ({s})" for t, s in zip(sched["T"], sched["split"])])
    ax.set_title("Fenetres de cible [T, T+90 j) par snapshot (bleu = train, jaune = validation, rouge = test)")
    plt.tight_layout()
    plt.savefig(FIG / "calendrier_snapshots.png", dpi=150)
    plt.close()

    print("Toutes les figures ont ete generees dans reports/figures/ !")

if __name__ == "__main__":
    main()

