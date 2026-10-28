"""Phase 4 : variables client SANS FUITE de donnees.

Regle d'or :
    build_features(df, T)  -> n'utilise QUE les lignes avec InvoiceDate <  T   (le passe)
    build_target(df, T, h) -> regarde UNIQUEMENT   T <= InvoiceDate < T + h    (le futur)

Un client est "churn" (1) s'il n'achete rien pendant les h jours suivant T.
Seules les VENTES comptent comme achat (un retour n'est pas un achat).
"""
import numpy as np
import pandas as pd

EUROPE = {
    "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Italy",
    "Finland", "Austria", "Norway", "Denmark", "Sweden", "Channel Islands", "Cyprus", "Greece", "Iceland",
    "Poland", "Lithuania", "Malta", "Czech Republic", "European Community", "Hungary", "Romania", "Latvia",
}


def country_group(country: str) -> str:
    if country == "United Kingdom":
        return "UK"
    if country in EUROPE:
        return "Europe"
    return "Rest of World"


# ------------------------------------------------------------------ features
def build_features(df: pd.DataFrame, T, cfg: dict) -> pd.DataFrame:
    """Une ligne par client ayant au moins 1 achat AVANT T. Toutes les variables sont calculees avant T."""
    T = pd.Timestamp(T)
    Tday = T.normalize()
    win = int(cfg["features"]["trend_window_days"])
    wholesale_thr = cfg["features"]["wholesale_avg_qty_per_line"]

    h = df[(df["InvoiceDate"] < T) & df["CustomerID"].notna()]
    sales = h[~h["IsReturn"]]
    rets = h[h["IsReturn"]]
    if sales.empty:
        return pd.DataFrame()

    # ---- niveau facture (ventes)
    inv = sales.groupby(["CustomerID", "Invoice"], as_index=False).agg(
        InvoiceDay=("InvoiceDay", "min"),
        Revenue=("Revenue", "sum"),
        Qty=("Quantity", "sum"),
        Lines=("StockCode", "size"),
    )
    g = inv.groupby("CustomerID")
    f = pd.DataFrame({
        "last_purchase": g["InvoiceDay"].max(),
        "first_purchase": g["InvoiceDay"].min(),
        "frequency": g["Invoice"].nunique(),
        "gross_revenue": g["Revenue"].sum(),
        "avg_basket": g["Revenue"].mean(),
        "total_qty": g["Qty"].sum(),
        "total_lines": g["Lines"].sum(),
    })

    # ---- RFM
    f["recency"] = (Tday - f["last_purchase"]).dt.days
    f["tenure_days"] = (Tday - f["first_purchase"]).dt.days

    # ---- retours (Monetary NET = ventes - retours ; Frequency ne compte que les ventes)
    ret_rev = rets.groupby("CustomerID")["Revenue"].sum().reindex(f.index).fillna(0.0)  # <= 0
    f["return_amount"] = -ret_rev
    f["monetary"] = f["gross_revenue"] + ret_rev
    f["return_rate"] = np.where(f["gross_revenue"] > 0, (-ret_rev / f["gross_revenue"]).clip(0, 1), 0.0)

    # ---- comportement
    f["avg_qty_per_line"] = f["total_qty"] / f["total_lines"]
    f["avg_items_per_invoice"] = f["total_qty"] / f["frequency"]
    f["n_unique_products"] = sales.groupby("CustomerID")["StockCode"].nunique()
    f["n_unique_categories"] = sales.groupby("CustomerID")["Category"].nunique()

    # ---- intervalle entre achats (jours distincts d'achat)
    days = inv[["CustomerID", "InvoiceDay"]].drop_duplicates().sort_values(["CustomerID", "InvoiceDay"])
    days["gap"] = days.groupby("CustomerID")["InvoiceDay"].diff().dt.days
    f["ipt_mean"] = days.groupby("CustomerID")["gap"].mean()          # NaN si un seul jour d'achat
    f["is_single_purchase"] = f["ipt_mean"].isna().astype(int)
    f["recency_over_ipt"] = f["recency"] / f["ipt_mean"]               # >1 : en retard sur son rythme habituel

    # ---- tendance recente (CA des `win` derniers jours vs les `win` precedents)
    last = inv[inv["InvoiceDay"] >= Tday - pd.Timedelta(days=win)].groupby("CustomerID")["Revenue"].sum()
    prev = inv[(inv["InvoiceDay"] < Tday - pd.Timedelta(days=win)) &
               (inv["InvoiceDay"] >= Tday - pd.Timedelta(days=2 * win))].groupby("CustomerID")["Revenue"].sum()
    f["revenue_recent"] = last.reindex(f.index).fillna(0.0)
    f["revenue_prev"] = prev.reindex(f.index).fillna(0.0)
    denom = f["revenue_recent"].abs() + f["revenue_prev"].abs()
    f["trend"] = np.where(denom > 0, (f["revenue_recent"] - f["revenue_prev"]) / denom, 0.0)  # dans [-1, 1]

    # ---- profil B2B (seuil FIXE de config : pas de statistique globale => pas de fuite)
    f["is_wholesale"] = (f["avg_qty_per_line"] >= wholesale_thr).astype(int)

    # ---- pays (dernier pays connu avant T)
    last_country = h.sort_values("InvoiceDate").groupby("CustomerID")["Country"].last().reindex(f.index)
    f["country_group"] = last_country.map(country_group)
    f["is_uk"] = (f["country_group"] == "UK").astype(int)

    # ---- parts de CA par categorie derivee
    cat = sales.groupby(["CustomerID", "Category"])["Revenue"].sum().unstack(fill_value=0.0).clip(lower=0)
    cat = cat.div(cat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    cat.columns = [f"cat_share_{c.lower()}" for c in cat.columns]
    f = f.join(cat.reindex(f.index).fillna(0.0))

    # ---- saison
    f["snapshot_month"] = T.month
    f["snapshot_date"] = T

    f = f.drop(columns=["last_purchase", "first_purchase", "total_qty", "total_lines"]).reset_index()
    return f


def build_target(df: pd.DataFrame, T, horizon: int, customers) -> pd.DataFrame:
    """churn = 1 si le client n'achete rien dans [T, T + horizon). Leve une erreur si la fenetre depasse les donnees."""
    T = pd.Timestamp(T)
    end = T + pd.Timedelta(days=int(horizon))
    data_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    if end > data_end:
        raise ValueError(
            f"Fenetre de cible incomplete : T={T.date()} + {horizon} j = {end.date()} > fin des donnees ({data_end.date()}). "
            "Choisis un T plus ancien."
        )
    fut = df[(df["InvoiceDate"] >= T) & (df["InvoiceDate"] < end) & (~df["IsReturn"]) & df["CustomerID"].notna()]
    buyers = set(fut["CustomerID"].dropna().unique())
    out = pd.DataFrame({"CustomerID": pd.Series(list(customers), dtype="Int64")})
    out["churn"] = (~out["CustomerID"].isin(buyers)).astype(int)
    return out


# ------------------------------------------------------------------ snapshots
def snapshot_schedule(cfg: dict, df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Tableau des snapshots : T, fin de fenetre de cible, split, validite (fenetre complete ?)."""
    horizon = int(horizon or cfg["churn"]["horizon_days"])
    data_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rows = []
    for s in cfg["snapshots"]:
        T = pd.Timestamp(s["date"])
        end = T + pd.Timedelta(days=horizon)
        rows.append({"T": T, "label_end": end, "split": s["split"], "fenetre_complete": end <= data_end})
    return pd.DataFrame(rows)


def check_split_purge(schedule: pd.DataFrame) -> list[str]:
    """Verifie que la fenetre de cible d'un split finit AVANT le premier T du split suivant (aucun recouvrement)."""
    order = ["train", "validation", "test"]
    problems = []
    present = [s for s in order if s in set(schedule["split"])]
    for a, b in zip(present[:-1], present[1:]):
        a_end = schedule.loc[schedule["split"] == a, "label_end"].max()
        b_start = schedule.loc[schedule["split"] == b, "T"].min()
        if a_end > b_start:
            problems.append(
                f"Fuite temporelle : la cible de '{a}' va jusqu'au {a_end.date()} mais '{b}' commence le {b_start.date()}."
            )
    return problems


def build_churn_dataset(df: pd.DataFrame, cfg: dict, horizon: int | None = None) -> pd.DataFrame:
    """Table empilee (client x snapshot) : features avant T + cible dans [T, T+h) + colonne 'split'."""
    horizon = int(horizon or cfg["churn"]["horizon_days"])
    sched = snapshot_schedule(cfg, df, horizon)
    problems = check_split_purge(sched)
    if problems:
        raise ValueError("\n".join(problems))
    if not sched["fenetre_complete"].all():
        bad = sched.loc[~sched["fenetre_complete"], "T"].dt.date.tolist()
        raise ValueError(f"Snapshots dont la fenetre depasse les donnees : {bad}")
    parts = []
    for _, r in sched.iterrows():
        feats = build_features(df, r["T"], cfg)
        tgt = build_target(df, r["T"], horizon, feats["CustomerID"])
        part = feats.merge(tgt, on="CustomerID", how="left")
        part["split"] = r["split"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_customer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Table pour la SEGMENTATION (descriptive, sans cible) : features sur tout l'historique, T = fin des donnees + 1 jour."""
    T = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    return build_features(df, T, cfg)
