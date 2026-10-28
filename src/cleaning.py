"""Phases 2-3 : chargement, audit et nettoyage d'Online Retail II.

Politique documentee (voir reports/01_cadrage.md) :
  * doublons exacts supprimes (inclut le chevauchement entre les 2 feuilles) ;
  * factures 'A' (ajustement de dettes) et codes non-produits exclus ;
  * Price <= 0 et Quantity = 0 exclus ;
  * Quantity < 0 SANS prefixe 'C' = ajustement de stock -> exclu ;
  * retours/annulations (facture 'C...') CONSERVES, marques IsReturn=True, Revenue negatif ;
  * lignes sans CustomerID CONSERVEES (ventes globales) mais marquees HasCustomer=False ;
  * outliers de quantite/prix MARQUES, jamais supprimes.
Chaque etape est tracee dans un tableau 'lignes exclues et pourquoi'.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .categories import assign_category, normalize_description

COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "Customer ID": "CustomerID",
    "Customer_ID": "CustomerID",
}
REQUIRED = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


# ---------------------------------------------------------------- chargement
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}. Colonnes trouvees : {list(df.columns)}")
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    # Description peut contenir des valeurs non-string (int, float) dans le xlsx reel
    df["Description"] = df["Description"].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Customer ID est lu en float64 a cause des NaN (13085.0) -> entier nullable
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    return df


def load_raw(path: str | Path, cache_path: str | Path | None = None) -> pd.DataFrame:
    """Lit le fichier brut (xlsx a 2 feuilles ou csv) et fusionne les feuilles.

    La lecture du xlsx (~1 M de lignes) prend 1 a 3 minutes : un cache parquet est utilise si fourni.
    La colonne 'SourceSheet' garde l'origine de chaque ligne (utile pour auditer le chevauchement).
    """
    path = Path(path)
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Telecharge Online Retail II (UCI ou Kaggle) et place le fichier dans data/raw/."
        )
    dtypes = {"Invoice": str, "InvoiceNo": str, "StockCode": str}
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=dtypes)
        frames = []
        for name, d in sheets.items():
            d = d.copy()
            d["SourceSheet"] = name
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        try:
            df = pd.read_csv(path, dtype=dtypes)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=dtypes, encoding="latin-1")
        df["SourceSheet"] = "csv"
    df = _standardize(df)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache_path, index=False)
        except ImportError:
            pass
    return df


# ---------------------------------------------------------------- audit
def audit_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Chiffres cles de qualite AVANT nettoyage (a copier dans la section 'Comprehension des donnees')."""
    c = cfg["cleaning"]
    rev = df["Quantity"] * df["Price"]
    is_c = df["Invoice"].str.startswith(c["return_invoice_prefix"])
    is_a = df["Invoice"].str.startswith(c["adjustment_invoice_prefix"])
    non_prod = _non_product_mask(df["StockCode"], c)
    body = df.drop(columns=["SourceSheet"], errors="ignore")
    rows = {
        "Lignes (brut)": len(df),
        "Factures distinctes": df["Invoice"].nunique(),
        "Clients distincts (ID non nul)": df["CustomerID"].nunique(),
        "Produits distincts (StockCode)": df["StockCode"].nunique(),
        "Pays distincts": df["Country"].nunique(),
        "Date min": df["InvoiceDate"].min(),
        "Date max": df["InvoiceDate"].max(),
        "Doublons exacts": int(body.duplicated().sum()),
        "Lignes sans CustomerID": int(df["CustomerID"].isna().sum()),
        "% lignes sans CustomerID": round(100 * df["CustomerID"].isna().mean(), 2),
        "% CA (brut) sans CustomerID": round(100 * rev[df["CustomerID"].isna()].sum() / rev.sum(), 2),
        "Lignes sans Description": int(df["Description"].isna().sum()),
        "Lignes de factures 'C' (retours/annulations)": int(is_c.sum()),
        "Lignes de factures 'A' (ajustements)": int(is_a.sum()),
        "Quantity < 0": int((df["Quantity"] < 0).sum()),
        "Quantity = 0": int((df["Quantity"] == 0).sum()),
        "Price <= 0": int((df["Price"] <= 0).sum()),
        "Lignes codes non-produits": int(non_prod.sum()),
        "StockCode avec plusieurs Description": int(df.groupby("StockCode")["Description"].nunique().gt(1).sum()),
    }
    return pd.DataFrame({"indicateur": rows.keys(), "valeur": rows.values()})


def _non_product_mask(stock: pd.Series, c: dict) -> pd.Series:
    codes = {x.upper() for x in c["non_product_codes"]}
    mask = stock.isin(codes)
    for pref in c["non_product_prefixes"]:
        mask |= stock.str.upper().str.startswith(pref.upper())
    return mask


# ---------------------------------------------------------------- nettoyage
def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique la politique de nettoyage. Retourne (transactions_clean, journal_exclusions)."""
    c = cfg["cleaning"]
    df = df.drop(columns=["SourceSheet"], errors="ignore").copy()
    n0 = len(df)
    rev0 = float((df["Quantity"] * df["Price"]).sum())
    log = []

    def drop(mask: pd.Series, step: str, reason: str):
        nonlocal df
        mask = mask.fillna(False).astype(bool)
        removed = df[mask]
        log.append({
            "etape": step,
            "regle": reason,
            "lignes_exclues": int(mask.sum()),
            "% du brut": round(100 * mask.sum() / n0, 3),
            "CA_exclu (Quantity x Price)": round(float((removed["Quantity"] * removed["Price"]).sum()), 2),
        })
        df = df[~mask]

    drop(df.duplicated(), "1", "Doublons exacts (dont chevauchement entre les 2 feuilles)")
    drop(df["Invoice"].str.startswith(c["adjustment_invoice_prefix"]), "2", "Factures 'A' : ajustement comptable de dettes")
    drop(_non_product_mask(df["StockCode"], c), "3", "Codes non-produits (frais de port, commissions, tests, ajustements...)")
    drop(df["Price"] <= 0, "4", "Price <= 0 (dons, erreurs, ajustements)")
    drop(df["Quantity"] == 0, "5", "Quantity = 0")
    is_return = df["Invoice"].str.startswith(c["return_invoice_prefix"])
    drop(~is_return & (df["Quantity"] < 0), "6", "Quantity < 0 sans prefixe 'C' (ajustement de stock, casse, perte)")
    is_return = df["Invoice"].str.startswith(c["return_invoice_prefix"])
    drop(is_return & (df["Quantity"] > 0), "7", "Facture 'C' avec Quantity > 0 (incoherent)")

    df = df.reset_index(drop=True)
    df["IsReturn"] = df["Invoice"].str.startswith(c["return_invoice_prefix"])
    df["Revenue"] = df["Quantity"] * df["Price"]  # negatif pour les retours
    df["HasCustomer"] = df["CustomerID"].notna()

    # Description canonique : la plus frequente par StockCode, normalisee
    df["Description"] = normalize_description(df["Description"])
    mode_desc = (
        df.dropna(subset=["Description"]).groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0])
    )
    df["Description"] = df["StockCode"].map(mode_desc).fillna("UNKNOWN").astype(str)
    df["Category"] = assign_category(df["Description"])

    # Variables temporelles
    d = df["InvoiceDate"]
    df["InvoiceDay"] = d.dt.normalize()
    df["Year"] = d.dt.year
    df["Month"] = d.dt.month
    df["YearMonth"] = d.dt.to_period("M").astype(str)
    df["DayOfWeek"] = d.dt.dayofweek
    df["Hour"] = d.dt.hour

    # Outliers : MARQUES, pas supprimes (une grosse commande B2B est legitime)
    q = c["outlier_quantile"]
    sales = ~df["IsReturn"]
    df["IsQtyOutlier"] = df["Quantity"].abs() > df.loc[sales, "Quantity"].quantile(q)
    df["IsPriceOutlier"] = df["Price"] > df.loc[sales, "Price"].quantile(q)

    log_df = pd.DataFrame(log)
    total = {
        "etape": "TOTAL", "regle": "Lignes exclues au total",
        "lignes_exclues": int(n0 - len(df)), "% du brut": round(100 * (n0 - len(df)) / n0, 3),
        "CA_exclu (Quantity x Price)": round(rev0 - float(df["Revenue"].sum()), 2),
    }
    kept = {
        "etape": "RESTE", "regle": "Lignes conservees (transactions_clean)",
        "lignes_exclues": len(df), "% du brut": round(100 * len(df) / n0, 3),
        "CA_exclu (Quantity x Price)": round(float(df["Revenue"].sum()), 2),
    }
    log_df = pd.concat([log_df, pd.DataFrame([total, kept])], ignore_index=True)
    return df, log_df


def quality_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compare les factures AVEC et SANS CustomerID (le biais a signaler dans les limites)."""
    sales = df[~df["IsReturn"]]
    inv = sales.groupby("Invoice").agg(
        has_id=("HasCustomer", "first"),
        basket=("Revenue", "sum"),
        n_lines=("StockCode", "size"),
        n_items=("Quantity", "sum"),
        country=("Country", "first"),
    )
    out = inv.groupby("has_id").agg(
        n_factures=("basket", "size"),
        panier_moyen=("basket", "mean"),
        panier_median=("basket", "median"),
        lignes_par_facture=("n_lines", "mean"),
        articles_par_facture=("n_items", "mean"),
        part_UK=("country", lambda s: (s == "United Kingdom").mean()),
    ).round(2)
    out.index = out.index.map({True: "Avec CustomerID", False: "Sans CustomerID"})
    return out
