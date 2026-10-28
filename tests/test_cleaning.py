import pandas as pd

from src.categories import category_of


def test_exclusion_rules(tiny):
    assert not (tiny["StockCode"] == "POST").any()                 # non-produit exclu
    assert not (tiny["Price"] <= 0).any()                          # prix nul exclu
    assert not tiny["Invoice"].str.startswith("A").any()           # facture A exclue


def test_returns_kept_with_negative_revenue(tiny):
    ret = tiny[tiny["IsReturn"]]
    assert len(ret) == 2 and (ret["Revenue"] < 0).all() and (ret["Quantity"] < 0).all()


def test_missing_customer_kept_but_flagged(tiny):
    guest = tiny[~tiny["HasCustomer"]]
    assert len(guest) == 1 and guest["CustomerID"].isna().all()    # conserve pour les ventes globales


def test_dtypes_and_columns(tiny):
    assert str(tiny["CustomerID"].dtype) == "Int64"
    for c in ["Revenue", "IsReturn", "HasCustomer", "Category", "InvoiceDay", "YearMonth", "IsQtyOutlier"]:
        assert c in tiny.columns


def test_exclusion_log_adds_up(cfg):
    from tests.conftest import make_raw
    from src.cleaning import clean
    raw = make_raw([
        ("1", "10001", "MUG", 1, "2010-01-01 10:00", 1.0, 1, "United Kingdom"),
        ("1", "10001", "MUG", 1, "2010-01-01 10:00", 1.0, 1, "United Kingdom"),   # doublon
        ("2", "POST", "POSTAGE", 1, "2010-01-02 10:00", 9.0, 1, "United Kingdom"),
        ("3", "10002", "BAG", 1, "2010-01-03 10:00", 0.0, 1, "United Kingdom"),
    ])
    df, log = clean(raw, cfg)
    total = log.loc[log["etape"] == "TOTAL", "lignes_exclues"].iloc[0]
    kept = log.loc[log["etape"] == "RESTE", "lignes_exclues"].iloc[0]
    assert total + kept == len(raw) and kept == len(df) == 1


def test_category_rules_order():
    assert category_of("CHRISTMAS CARD") == "CHRISTMAS"            # Noel prime sur papeterie
    assert category_of("LUNCH BAG RED") == "BAGS"                  # sac prime sur cuisine
    assert category_of("RETRO MUG") == "KITCHEN_DINING"
    assert category_of("XYZ") == "OTHER"
