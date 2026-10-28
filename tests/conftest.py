import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean  # noqa: E402
from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_raw(rows):
    """rows: (Invoice, StockCode, Description, Quantity, 'YYYY-MM-DD HH:MM', Price, CustomerID, Country)"""
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = pd.to_numeric(df["CustomerID"]).astype("Int64")
    return df


@pytest.fixture()
def tiny(cfg):
    """Mini-dataset a valeurs connues (clean() applique la vraie politique)."""
    raw = make_raw([
        # client 1 : 3 achats + 1 retour avant T=2011-01-01
        ("1001", "10001", "RED MUG", 2, "2010-10-01 10:00", 5.0, 1, "United Kingdom"),
        ("1001", "10002", "LUNCH BAG", 1, "2010-10-01 10:00", 10.0, 1, "United Kingdom"),
        ("1002", "10001", "RED MUG", 4, "2010-11-01 11:00", 5.0, 1, "United Kingdom"),
        ("1003", "10003", "CHRISTMAS CARD", 10, "2010-12-20 09:00", 1.0, 1, "United Kingdom"),
        ("C1004", "10001", "RED MUG", -1, "2010-12-22 09:00", 5.0, 1, "United Kingdom"),
        # client 2 : 1 seul achat, grossiste (qte moyenne par ligne elevee)
        ("1005", "10001", "RED MUG", 48, "2010-09-15 14:00", 4.0, 2, "Germany"),
        # client 1 rachete APRES T (dans la fenetre de 90 j) ; client 2 jamais
        ("1006", "10001", "RED MUG", 1, "2011-02-10 10:00", 5.0, 1, "United Kingdom"),
        # client 3 : ne commence qu'apres T (ne doit PAS etre dans les features de T)
        ("1007", "10002", "LUNCH BAG", 1, "2011-02-11 10:00", 10.0, 3, "United Kingdom"),
        # client 4 : achat avant T, seul un RETOUR dans la fenetre -> churn = 1
        ("1008", "10002", "LUNCH BAG", 2, "2010-12-01 10:00", 10.0, 4, "France"),
        ("C1009", "10002", "LUNCH BAG", -1, "2011-01-15 10:00", 10.0, 4, "France"),
        # ligne sans client + non-produit + prix 0 + facture A (a exclure/garder selon la politique)
        ("1010", "10001", "RED MUG", 3, "2011-03-01 10:00", 5.0, None, "United Kingdom"),
        ("1011", "POST", "POSTAGE", 1, "2011-03-01 10:00", 18.0, 1, "United Kingdom"),
        ("1012", "10004", "FREE GIFT", 1, "2011-03-02 10:00", 0.0, 1, "United Kingdom"),
        ("A1013", "B", "Adjust bad debt", 1, "2011-03-03 10:00", -100.0, None, "United Kingdom"),
        # fin des donnees (sert de borne pour la validite des fenetres)
        ("1014", "10001", "RED MUG", 1, "2011-06-01 10:00", 5.0, 1, "United Kingdom"),
    ])
    df, log = clean(raw, cfg)
    return df
