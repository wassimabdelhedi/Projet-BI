"""Chargement de la configuration et utilitaires de chemins."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Chemin absolu d'une entree de cfg['paths'] (cree le dossier si c'est un dossier)."""
    p = ROOT / cfg["paths"][key]
    if key.endswith("_dir"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df, name: str, cfg: dict, folder: str = "processed_dir") -> Path:
    """Sauvegarde en parquet (ou csv selon config). Repli automatique sur csv si pyarrow absent."""
    base = resolve(cfg, folder)
    fmt = cfg["paths"].get("processed_format", "parquet")
    if fmt == "parquet":
        try:
            out = base / f"{name}.parquet"
            df.to_parquet(out, index=False)
            return out
        except ImportError:
            pass
    out = base / f"{name}.csv"
    df.to_csv(out, index=False)
    return out


def load_table(name: str, cfg: dict, folder: str = "processed_dir"):
    import pandas as pd

    base = resolve(cfg, folder)
    for ext, reader in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
        f = base / f"{name}.{ext}"
        if f.exists():
            df = reader(f)
            for col in ("InvoiceDate", "InvoiceDay", "snapshot_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            if "CustomerID" in df.columns:
                df["CustomerID"] = pd.to_numeric(df["CustomerID"]).astype("Int64")
            return df
    raise FileNotFoundError(f"{name} introuvable dans {base}. Lance d'abord les phases precedentes.")
