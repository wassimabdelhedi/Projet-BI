"""Tests anti-fuite : aucune donnee posterieure a T ne doit influencer les features."""
import pandas as pd
import pytest

from src.features import build_churn_dataset, build_features, build_target, check_split_purge, snapshot_schedule

T = pd.Timestamp("2011-01-01")


def test_features_ignore_everything_after_T(tiny, cfg):
    base = build_features(tiny, T, cfg).set_index("CustomerID").sort_index()

    # On ajoute un ENORME achat futur pour le client 1 et un nouveau client futur
    future = tiny.iloc[[0]].copy()
    future["InvoiceDate"] = pd.Timestamp("2011-02-01 10:00")
    future["InvoiceDay"] = pd.Timestamp("2011-02-01")
    future["Quantity"], future["Revenue"] = 100000, 100000 * float(future["Price"].iloc[0])
    future["CustomerID"] = pd.array([1], dtype="Int64")
    perturbed = pd.concat([tiny, future], ignore_index=True)
    after = build_features(perturbed, T, cfg).set_index("CustomerID").sort_index()

    pd.testing.assert_frame_equal(base, after)


def test_customers_first_seen_after_T_are_absent(tiny, cfg):
    feats = build_features(tiny, T, cfg)
    assert 3 not in set(feats["CustomerID"])            # client 3 commence apres T
    assert feats["CustomerID"].isin([1, 2, 4]).all()


def test_rfm_values(tiny, cfg):
    f = build_features(tiny, T, cfg).set_index("CustomerID")
    c1 = f.loc[1]
    assert c1["frequency"] == 3                          # 3 factures de vente (le retour ne compte pas)
    assert c1["recency"] == (T - pd.Timestamp("2010-12-20")).days
    assert c1["gross_revenue"] == pytest.approx(2 * 5 + 10 + 4 * 5 + 10 * 1)  # 10+10+20+10 = 50
    assert c1["return_amount"] == pytest.approx(5.0)
    assert c1["monetary"] == pytest.approx(45.0)         # CA net = ventes - retours
    assert c1["return_rate"] == pytest.approx(5.0 / 50.0)
    assert f.loc[2, "is_single_purchase"] == 1 and pd.isna(f.loc[2, "ipt_mean"])
    assert f.loc[2, "is_wholesale"] == 1 and f.loc[1, "is_wholesale"] == 0
    assert f.loc[2, "country_group"] == "Europe" and f.loc[1, "country_group"] == "UK"
    assert (f["snapshot_date"] == T).all()


def test_target_definition(tiny, cfg):
    feats = build_features(tiny, T, cfg)
    tgt = build_target(tiny, T, 90, feats["CustomerID"]).set_index("CustomerID")["churn"]
    assert tgt[1] == 0       # client 1 rachete le 2011-02-10 (dans [T, T+90))
    assert tgt[2] == 1       # client 2 n'a jamais rachete
    assert tgt[4] == 1       # un RETOUR n'est pas un achat


def test_target_horizon_sensitivity(tiny, cfg):
    feats = build_features(tiny, T, cfg)
    short = build_target(tiny, T, 30, feats["CustomerID"]).set_index("CustomerID")["churn"]
    assert short[1] == 1     # sur 30 jours, le rachat du 10 fevrier est hors fenetre


def test_target_refuses_incomplete_window(tiny):
    with pytest.raises(ValueError, match="Fenetre de cible incomplete"):
        build_target(tiny, pd.Timestamp("2011-05-01"), 90, [1])   # donnees finissent le 2011-06-01


def test_purge_detects_overlap():
    sched = pd.DataFrame({
        "T": pd.to_datetime(["2011-01-31", "2011-03-01"]),
        "label_end": pd.to_datetime(["2011-05-01", "2011-05-30"]),
        "split": ["train", "validation"],
    })
    assert len(check_split_purge(sched)) == 1


def test_project_calendar_is_leak_free(cfg, tiny):
    big = tiny.copy()
    last = big.iloc[[-1]].copy()
    last["InvoiceDate"] = pd.Timestamp("2011-12-09 12:50")
    big = pd.concat([big, last], ignore_index=True)
    sched = snapshot_schedule(cfg, big, 90)
    assert check_split_purge(sched) == []
    assert sched["fenetre_complete"].all()
    assert sched["T"].max() == pd.Timestamp("2011-09-09")
