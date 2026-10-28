"""Genere un jeu de donnees SYNTHETIQUE au format d'Online Retail II (2 feuilles, memes colonnes).

BUT UNIQUE : tester que le pipeline tourne de bout en bout avant d'avoir telecharge le vrai fichier.
Les resultats obtenus sur ces donnees n'ont AUCUNE valeur d'analyse (ne jamais les citer dans le rapport).

Il reproduit les defauts du vrai dataset : ~20 % de lignes sans client, factures 'C', codes non-produits,
Price = 0, factures 'A', doublons, chevauchement des 2 feuilles, descriptions multiples, grossistes.

Usage : python scripts/generate_sample_data.py --out data/raw/sample_retail.xlsx --customers 1500
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORDS = ["HEART", "VINTAGE", "RETRO", "CHRISTMAS", "BAG", "MUG", "CANDLE", "T-LIGHT HOLDER", "CARD", "WRAP",
         "NOTEBOOK", "CUSHION", "FRAME", "CLOCK", "TOY", "GARDEN", "BOX", "BASKET", "JAR", "NECKLACE", "LUNCH BAG",
         "TEAPOT", "BOWL", "SIGN", "LANTERN", "PEN", "RIBBON", "DOLL", "VASE", "HOOK"]
ADJ = ["PINK", "BLUE", "RED", "GREEN", "SET OF 3", "SMALL", "LARGE", "WOODEN", "METAL", "SPOTTY", "FLORAL", "RETROSPOT"]


def make_catalog(rng, n=300):
    codes = [f"{rng.integers(10000, 99999)}{rng.choice(['', 'A', 'B'])}" for _ in range(n)]
    codes = list(dict.fromkeys(codes))
    desc = [f"{rng.choice(ADJ)} {rng.choice(WORDS)} {rng.choice(ADJ)}" for _ in codes]
    price = np.round(rng.lognormal(1.0, 0.6, len(codes)), 2).clip(0.2, 60)
    pop = rng.dirichlet(np.ones(len(codes)) * 0.5)
    return pd.DataFrame({"StockCode": codes, "Description": desc, "Price": price, "pop": pop})


def generate(n_customers=1500, seed=0):
    rng = np.random.default_rng(seed)
    cat = make_catalog(rng)
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09 12:50")
    span = (end - start).days
    rows, inv_no = [], 500000

    def add_invoice(cust, when, wholesale, country, prefix=""):
        nonlocal inv_no
        inv_no += 1
        n_lines = rng.integers(1, 12) if not wholesale else rng.integers(5, 30)
        items = cat.sample(n_lines, weights="pop", random_state=int(rng.integers(1e9)))
        inv = f"{prefix}{inv_no}"
        for _, it in items.iterrows():
            q = int(rng.integers(1, 8)) if not wholesale else int(rng.choice([12, 24, 48, 72, 144]))
            if prefix == "C":
                q = -q
            desc = it.Description if rng.random() > 0.05 else it.Description.title()   # descriptions incoherentes
            rows.append((inv, it.StockCode, desc, q, when, it.Price, cust, country))
        if rng.random() < 0.15 and prefix == "":
            rows.append((inv, "POST", "POSTAGE", 1, when, 18.0, cust, country))
        return inv

    for c in range(n_customers):
        cust = 12000 + c
        wholesale = rng.random() < 0.10
        country = "United Kingdom" if rng.random() < 0.9 else rng.choice(["Germany", "France", "EIRE", "Spain", "Australia"])
        first = start + pd.Timedelta(days=int(rng.integers(0, span - 30)))
        rate = rng.lognormal(-3.2, 0.7)                      # achats par jour
        lifetime = rng.exponential(300) + 30                 # duree d'activite (jours) -> vrai churn
        t = first
        last_t = first
        while t < min(end, first + pd.Timedelta(days=lifetime)):
            # saisonnalite : plus d'achats en nov-dec
            if rng.random() < (1.8 if t.month in (10, 11, 12) else 1.0) / 1.8:
                when = t.replace(hour=int(rng.integers(8, 18)), minute=int(rng.integers(0, 60)))
                add_invoice(cust, when, wholesale, country)
                if rng.random() < 0.03:
                    add_invoice(cust, when + pd.Timedelta(days=int(rng.integers(1, 10))), wholesale, country, prefix="C")
                last_t = t
            t = t + pd.Timedelta(days=float(rng.exponential(1 / rate)) + 1)
    # clients invites (sans ID) : ~20 % des lignes
    n_guest = int(len(rows) * 0.25 / 6)
    for _ in range(n_guest):
        when = start + pd.Timedelta(days=int(rng.integers(0, span))) + pd.Timedelta(hours=int(rng.integers(8, 18)))
        add_invoice(np.nan, when, False, "United Kingdom")

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    df = df[df["InvoiceDate"] <= end]

    # defauts divers
    df = pd.concat([df,
        pd.DataFrame({"Invoice": ["A" + str(900000 + i) for i in range(5)], "StockCode": "B", "Description": "Adjust bad debt",
                      "Quantity": 1, "InvoiceDate": pd.Timestamp("2010-08-12 14:00"), "Price": -11062.06,
                      "Customer ID": np.nan, "Country": "United Kingdom"}),
        pd.DataFrame({"Invoice": ["TEST1"], "StockCode": ["TEST001"], "Description": ["This is a test product."],
                      "Quantity": [5], "InvoiceDate": [pd.Timestamp("2010-03-01 10:00")], "Price": [4.5],
                      "Customer ID": [np.nan], "Country": ["United Kingdom"]}),
    ], ignore_index=True)
    zero = df.sample(60, random_state=1).copy(); zero["Price"] = 0.0; zero["Description"] = np.nan; zero["Customer ID"] = np.nan
    stock_adj = df.sample(40, random_state=2).copy(); stock_adj["Quantity"] = -stock_adj["Quantity"].abs()
    stock_adj["Invoice"] = stock_adj["Invoice"].str.replace("C", "", regex=False); stock_adj["Customer ID"] = np.nan
    stock_adj["Description"] = "damaged"
    df = pd.concat([df, zero, stock_adj], ignore_index=True)
    df = pd.concat([df, df.sample(int(len(df) * 0.005), random_state=3)], ignore_index=True)   # doublons exacts
    df["Customer ID"] = df["Customer ID"].astype("float64")
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/sample_retail.xlsx")
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = generate(a.customers, a.seed)
    cut1, cut2 = pd.Timestamp("2010-12-10"), pd.Timestamp("2010-12-01")
    s1 = df[df["InvoiceDate"] < cut1]                       # feuille 2009-2010 (jusqu'au 9 dec 2010)
    s2 = df[df["InvoiceDate"] >= cut2]                      # feuille 2010-2011 (des le 1er dec 2010) -> chevauchement
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".xlsx":
        with pd.ExcelWriter(out) as w:
            s1.to_excel(w, sheet_name="Year 2009-2010", index=False)
            s2.to_excel(w, sheet_name="Year 2010-2011", index=False)
    else:
        df.to_csv(out, index=False)
    print(f"{len(df):,} lignes ecrites dans {out} (donnees SYNTHETIQUES, pour tester le pipeline uniquement)")


if __name__ == "__main__":
    main()
