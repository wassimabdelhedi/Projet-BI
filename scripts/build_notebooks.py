"""Construit les 3 notebooks (nbformat). Lancer : python scripts/build_notebooks.py"""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config, resolve, save_table, load_table
cfg = load_config()
FIG = resolve(cfg, "figures_dir")
OUT = resolve(cfg, "outputs_dir")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": .3})
'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, NB / name)


# ============================================================ 01 AUDIT
nb([
("md", """# 01 - Audit des données (Phase 2)
**Objectif :** comprendre le dataset *Online Retail II* et **quantifier** ses défauts avant tout nettoyage.
Tous les chiffres de ce notebook alimentent la section « Compréhension des données » et « Limites » du rapport.

> Prérequis : placer `online_retail_II.xlsx` dans `data/raw/` (ou changer `paths.raw_file` dans `config.yaml`).
> La première lecture du xlsx prend 1 à 3 minutes ; un cache parquet est ensuite utilisé."""),
("code", SETUP + '''
from src.cleaning import load_raw, audit_summary, clean, quality_bias_table
RAW = ROOT / cfg["paths"]["raw_file"]
CACHE = resolve(cfg, "interim_dir") / "raw_merged.parquet"
print("Fichier brut :", RAW)'''),
("md", "## 1. Chargement et fusion des deux feuilles"),
("code", '''df = load_raw(RAW, CACHE)
print(f"{len(df):,} lignes x {df.shape[1]} colonnes")
print(df.groupby("SourceSheet")["InvoiceDate"].agg(["count", "min", "max"]))
df.head()'''),
("code", "df.dtypes"),
("md", """## 2. Chevauchement entre les deux feuilles
Les deux feuilles partagent les premiers jours de décembre 2010. Les lignes présentes dans **les deux** sont des doublons exacts à supprimer après fusion."""),
("code", '''body = df.drop(columns=["SourceSheet"])
overlap = df[body.duplicated(keep=False)]
print("Lignes impliquées dans un doublon exact :", len(overlap))
print("Doublons à supprimer (garder 1 exemplaire) :", int(body.duplicated().sum()))
if len(overlap):
    print("Période concernée :", overlap["InvoiceDate"].min(), "->", overlap["InvoiceDate"].max())
    print(overlap.groupby("SourceSheet").size())'''),
("md", "## 3. Tableau de synthèse qualité (avant nettoyage)"),
("code", '''summary = audit_summary(df, cfg)
summary.to_csv(OUT / "audit_summary.csv", index=False)
summary'''),
("md", "## 4. Valeurs manquantes"),
("code", '''miss = df.isna().agg(["sum", "mean"]).T
miss["mean"] = (100 * miss["mean"]).round(2)
miss.columns = ["manquantes", "% manquantes"]
miss'''),
("md", """## 5. Clients absents (`CustomerID`) : poids et biais
Les lignes sans identifiant ne peuvent entrer ni dans le RFM, ni dans le churn, ni dans la recommandation par client.
Il faut mesurer **combien elles pèsent** et **en quoi elles diffèrent**."""),
("code", '''rev = df["Quantity"] * df["Price"]
no_id = df["CustomerID"].isna()
print(f"% lignes sans ID : {100*no_id.mean():.1f} %")
print(f"% du CA brut sans ID : {100*rev[no_id].sum()/rev.sum():.1f} %")'''),
("md", "## 6. Annulations, quantités et prix anormaux"),
("code", '''prefix = df["Invoice"].str.extract(r"^([A-Za-z]+)")[0].fillna("(numérique)")
print("Préfixes de facture :"); print(prefix.value_counts())
print("\\nQuantity <= 0 :", int((df["Quantity"] <= 0).sum()), "| Price <= 0 :", int((df["Price"] <= 0).sum()))
neg_no_c = (df["Quantity"] < 0) & ~df["Invoice"].str.startswith("C")
print("Quantity < 0 SANS préfixe C (ajustements de stock) :", int(neg_no_c.sum()))
print("\\nQuantité : percentiles"); print(df["Quantity"].quantile([0, .001, .01, .5, .99, .999, 1]))
print("\\nPrix : percentiles"); print(df["Price"].quantile([0, .001, .01, .5, .99, .999, 1]))'''),
("md", "## 7. Codes produits non-produits et descriptions incohérentes"),
("code", '''non_alpha = df.loc[~df["StockCode"].str.match(r"^\\d{5}[A-Z]*$"), "StockCode"].value_counts()
print("StockCode qui ne ressemblent pas à un code produit (5 chiffres + lettres) :")
print(non_alpha.head(25))
multi = df.groupby("StockCode")["Description"].nunique()
print("\\nStockCode avec plusieurs descriptions :", int((multi > 1).sum()), "sur", len(multi))
print(df[df["StockCode"].isin(multi[multi > 1].index[:3])].groupby(["StockCode", "Description"]).size())'''),
("md", "## 8. Pays"),
("code", '''c = df["Country"].value_counts()
print(c.head(10)); print(f"\\nRoyaume-Uni : {100*c.get('United Kingdom', 0)/c.sum():.1f} % des lignes")'''),
("md", """## 9. Nettoyage selon la politique documentée (`src/cleaning.py`)
Chaque règle est appliquée dans l'ordre et tracée dans le tableau **« lignes exclues et pourquoi »**, à copier dans le rapport."""),
("code", '''clean_df, log = clean(df, cfg)
log.to_csv(OUT / "exclusion_log.csv", index=False)
log'''),
("code", '''print("Biais : factures avec vs sans CustomerID")
quality_bias_table(clean_df)'''),
("md", """## 10. À retenir pour le rapport
Compléter après exécution :
- part de lignes et de CA sans `CustomerID` : ...
- nombre de doublons (dont chevauchement) : ...
- retours (facture `C`) : ...% des lignes
- lignes exclues au total : ...%
- les factures sans ID diffèrent-elles (panier, pays) ? -> limite à signaler"""),
], "01_data_audit.ipynb")

# ============================================================ 02 EDA
nb([
("md", """# 02 - Prétraitement et EDA (Phase 3)
**Objectif :** explorer les ventes, justifier la définition du churn (distribution des délais inter-achats), mesurer la concentration du CA (Pareto), la rétention par cohorte et le profil B2B.

> Prérequis : `python run_pipeline.py` (ou le notebook 01) pour produire `data/processed/transactions_clean`."""),
("code", SETUP + '''
df = load_table("transactions_clean", cfg)
sales = df[~df["IsReturn"]]
print(f"{len(df):,} lignes | période {df['InvoiceDate'].min():%Y-%m-%d} -> {df['InvoiceDate'].max():%Y-%m-%d}")'''),
("md", "## 1. Évolution mensuelle des ventes (CA net = ventes - retours)"),
("code", '''m = df.groupby("YearMonth").agg(ca_net=("Revenue", "sum"), commandes=("Invoice", lambda s: s[~df.loc[s.index, "IsReturn"]].nunique()),
                                 clients=("CustomerID", "nunique"))
fig, ax = plt.subplots(); m["ca_net"].plot(kind="bar", ax=ax, color="#2b6cb0"); ax.set_title("CA net mensuel (GBP)")
plt.xticks(rotation=60); plt.tight_layout(); plt.savefig(FIG / "ca_mensuel.png", dpi=150); plt.show()
m.round(0)'''),
("md", """**Attention :** décembre 2011 est incomplet (données jusqu'au 9 décembre). Ne pas conclure à une chute."""),
("md", "## 2. Saisonnalité : mois, jour de la semaine, heure"),
("code", '''fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
sales.groupby("Month")["Revenue"].sum().plot(kind="bar", ax=axes[0], title="CA par mois de l'année")
sales.groupby("DayOfWeek")["Invoice"].nunique().plot(kind="bar", ax=axes[1], title="Commandes par jour (0 = lundi)")
sales.groupby("Hour")["Invoice"].nunique().plot(kind="bar", ax=axes[2], title="Commandes par heure")
plt.tight_layout(); plt.savefig(FIG / "saisonnalite.png", dpi=150); plt.show()'''),
("md", "## 3. Pays"),
("code", '''c = sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
print((100 * c / c.sum()).head(10).round(2))
uk = c.get("United Kingdom", 0) / c.sum()
print(f"\\nRoyaume-Uni = {100*uk:.1f} % du CA -> pays exclu du clustering, utilisé pour profiler les segments")'''),
("md", "## 4. Produits populaires"),
("code", '''top_q = sales.groupby("Description")["Quantity"].sum().nlargest(10)
top_r = sales.groupby("Description")["Revenue"].sum().nlargest(10)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
top_q[::-1].plot(kind="barh", ax=axes[0], title="Top 10 par quantité"); top_r[::-1].plot(kind="barh", ax=axes[1], title="Top 10 par CA")
plt.tight_layout(); plt.savefig(FIG / "top_produits.png", dpi=150); plt.show()
print("Répartition du CA par catégorie dérivée (approximative) :")
print((100 * sales.groupby("Category")["Revenue"].sum() / sales["Revenue"].sum()).sort_values(ascending=False).round(1))'''),
("md", "## 5. Distribution du panier (CA par facture)"),
("code", '''basket = sales.groupby("Invoice")["Revenue"].sum()
print(basket.describe().round(1))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
basket.clip(upper=basket.quantile(.99)).plot(kind="hist", bins=60, ax=ax[0], title="Panier (tronqué au 99e percentile)")
np.log10(basket[basket > 0]).plot(kind="hist", bins=60, ax=ax[1], title="log10(panier) : distribution très asymétrique")
plt.tight_layout(); plt.savefig(FIG / "paniers.png", dpi=150); plt.show()'''),
("md", """## 6. Courbe de Pareto : concentration du CA par client
Mesure combien de clients font l'essentiel du CA (et le poids des grossistes)."""),
("code", '''cust = df[df["HasCustomer"] & ~df["IsReturn"]].groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
cum = cust.cumsum() / cust.sum(); share_x = np.arange(1, len(cust) + 1) / len(cust)
fig, ax = plt.subplots(); ax.plot(share_x * 100, cum.values * 100); ax.axhline(80, color="grey", ls="--")
ax.set_xlabel("% des clients (du plus gros au plus petit)"); ax.set_ylabel("% du CA cumulé"); ax.set_title("Courbe de Pareto")
plt.savefig(FIG / "pareto.png", dpi=150); plt.show()
top20 = cum.iloc[int(0.2 * len(cust)) - 1]; top1 = cust.head(max(1, int(0.01 * len(cust)))).sum() / cust.sum()
print(f"Les 20 % de clients les plus gros font {100*top20:.1f} % du CA ; le top 1 % fait {100*top1:.1f} %")'''),
("md", """## 7. Délais inter-achats : justification de l'horizon de churn
**Question :** au bout de combien de jours sans achat un client est-il « probablement perdu » ?
On regarde le délai entre deux achats consécutifs (jours distincts) de chaque client. Si 80 % des clients qui rachètent le font en moins de *X* jours, un silence de *X* jours est un signal raisonnable."""),
("code", '''days = (sales[sales["HasCustomer"]][["CustomerID", "InvoiceDay"]].drop_duplicates().sort_values(["CustomerID", "InvoiceDay"]))
gaps = days.groupby("CustomerID")["InvoiceDay"].diff().dt.days.dropna()
pct = gaps.quantile([.5, .7, .8, .9, .95]).round(0)
print("Percentiles des délais entre deux achats (jours) :"); print(pct)
fig, ax = plt.subplots(); gaps.clip(upper=365).plot(kind="hist", bins=73, ax=ax)
for p, col in [(.8, "red"), (.9, "orange")]:
    ax.axvline(gaps.quantile(p), color=col, ls="--", label=f"p{int(p*100)} = {gaps.quantile(p):.0f} j")
ax.axvline(cfg["churn"]["horizon_days"], color="green", label=f"horizon choisi = {cfg['churn']['horizon_days']} j")
ax.legend(); ax.set_title("Délais entre achats consécutifs (tronqué à 365 j)"); ax.set_xlabel("jours")
plt.savefig(FIG / "delais_inter_achats.png", dpi=150); plt.show()
p80, h = gaps.quantile(.8), cfg["churn"]["horizon_days"]
print(f"\\np80 des délais = {p80:.0f} j | p90 = {gaps.quantile(.9):.0f} j | horizon configuré = {h} j")
print("-> Si l'horizon est très éloigné du p80, argumente le choix (ou ajuste churn.horizon_days dans config.yaml) puis reporte-le dans le cadrage.")
print(f"Sensibilité testée au notebook 03 : {cfg['churn']['sensitivity_horizons']}")'''),
("md", "## 8. Cohortes de rétention (mois de première commande)"),
("code", '''first = sales[sales["HasCustomer"]].groupby("CustomerID")["InvoiceDay"].min().dt.to_period("M").rename("cohort")
act = sales[sales["HasCustomer"]].merge(first, on="CustomerID")
act["period"] = act["InvoiceDay"].dt.to_period("M")
act["age"] = (act["period"] - act["cohort"]).apply(lambda x: x.n)
coh = act.groupby(["cohort", "age"])["CustomerID"].nunique().unstack(fill_value=0)
ret = coh.div(coh[0], axis=0).round(3)
fig, ax = plt.subplots(figsize=(12, 7)); im = ax.imshow(ret.iloc[:, :13].values, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
ax.set_yticks(range(len(ret))); ax.set_yticklabels(ret.index.astype(str)); ax.set_xlabel("mois depuis la première commande")
ax.set_title("Rétention par cohorte (part des clients actifs)"); plt.colorbar(im); plt.savefig(FIG / "cohortes.png", dpi=150); plt.show()
print("Rétention moyenne à M+1, M+3, M+6, M+12 :"); print(ret[[1, 3, 6, 12]].mean().round(3) if ret.shape[1] > 12 else ret.mean().round(3).head(13))'''),
("md", "## 9. Retours"),
("code", '''ret_df = df[df["IsReturn"]]
print(f"Lignes de retour : {len(ret_df):,} ({100*len(ret_df)/len(df):.2f} %) | montant retourné : {ret_df['Revenue'].sum():,.0f} GBP "
      f"({100*abs(ret_df['Revenue'].sum())/sales['Revenue'].sum():.2f} % du CA brut)")
print("Part des retours sans CustomerID :", f"{100*(~ret_df['HasCustomer']).mean():.1f} %")'''),
("md", """## 10. Profil B2B / B2C : justifier le seuil `is_wholesale`
Le seuil est **fixe** dans `config.yaml` (jamais calculé sur toutes les données, pour éviter toute fuite). Ce graphique sert à le justifier."""),
("code", '''c = sales[sales["HasCustomer"]].groupby("CustomerID").agg(qty=("Quantity", "sum"), lines=("StockCode", "size"), rev=("Revenue", "sum"), n_inv=("Invoice", "nunique"))
c["avg_qty_per_line"] = c["qty"] / c["lines"]
thr = cfg["features"]["wholesale_avg_qty_per_line"]
fig, ax = plt.subplots(); np.log10(c["avg_qty_per_line"]).plot(kind="hist", bins=60, ax=ax)
ax.axvline(np.log10(thr), color="red", ls="--", label=f"seuil = {thr}"); ax.legend(); ax.set_title("log10 de la quantité moyenne par ligne, par client")
plt.savefig(FIG / "profil_b2b.png", dpi=150); plt.show()
w = c["avg_qty_per_line"] >= thr
print(f"Clients 'grossistes' : {100*w.mean():.1f} % des clients, {100*c.loc[w,'rev'].sum()/c['rev'].sum():.1f} % du CA")'''),
("md", """## 11. Conclusions de l'EDA (à compléter)
- Saisonnalité : ...
- Concentration (Pareto) : ...
- Horizon de churn justifié par p80 = ... jours : ...
- Rétention : ...
- Part du B2B : ..."""),
], "02_eda.ipynb")

# ============================================================ 03 FEATURES
nb([
("md", """# 03 - Feature engineering sans fuite (Phase 4)
**Objectif :** construire (a) la table client pour la segmentation, (b) la table `client × snapshot` pour le churn, **sans aucune fuite temporelle**.

Principe : à chaque date d'observation **T**, les variables n'utilisent que le passé (`InvoiceDate < T`) ; la cible regarde le futur (`T ≤ InvoiceDate < T + horizon`)."""),
("code", SETUP + '''
from src.features import (build_features, build_target, build_churn_dataset, build_customer_features,
                          snapshot_schedule, check_split_purge)
from src.categories import export_validation_sample
df = load_table("transactions_clean", cfg)
HORIZON = cfg["churn"]["horizon_days"]
print(f"Données : {df['InvoiceDate'].min():%Y-%m-%d} -> {df['InvoiceDate'].max():%Y-%m-%d} | horizon = {HORIZON} j")'''),
("md", """## 1. Calendrier des snapshots et règle de purge
Une fenêtre de cible `[T, T+90j)` regarde le futur. Si la cible d'un snapshot d'entraînement se termine **après** le début de la validation, le modèle a « vu » une partie de l'avenir de la validation (fuite). On impose donc que la cible d'un split finisse **avant** le premier T du split suivant."""),
("code", '''sched = snapshot_schedule(cfg, df, HORIZON)
print(sched.to_string(index=False))
problems = check_split_purge(sched)
print("\\nProblèmes de fuite :", problems or "aucun")
assert sched["fenetre_complete"].all(), "un snapshot dépasse la fin des données"

fig, ax = plt.subplots(figsize=(11, 4)); colors = {"train": "#2b6cb0", "validation": "#d69e2e", "test": "#c53030"}
for i, r in sched.iterrows():
    ax.barh(i, (r["label_end"] - r["T"]).days, left=r["T"], color=colors[r["split"]], alpha=.8)
    ax.plot(r["T"], i, "k|", ms=12)
ax.set_yticks(range(len(sched))); ax.set_yticklabels([f"T={t:%Y-%m-%d} ({s})" for t, s in zip(sched["T"], sched["split"])])
ax.set_title("Fenêtres de cible [T, T+90 j) par snapshot (bleu = train, jaune = validation, rouge = test)")
plt.tight_layout(); plt.savefig(FIG / "calendrier_snapshots.png", dpi=150); plt.show()'''),
("md", """## 2. Exemple : features d'un snapshot et preuve d'absence de fuite"""),
("code", '''T = pd.Timestamp("2011-03-31")
f = build_features(df, T, cfg)
print(f"{len(f):,} clients avec au moins un achat avant {T.date()}")
f.head()'''),
("code", '''# Preuve : on supprime TOUT ce qui est >= T ; les features doivent être strictement identiques.
f_past_only = build_features(df[df["InvoiceDate"] < T], T, cfg)
pd.testing.assert_frame_equal(f.sort_values("CustomerID").reset_index(drop=True),
                              f_past_only.sort_values("CustomerID").reset_index(drop=True))
print("OK : les features ne dépendent d'aucune donnée postérieure à T")'''),
("md", "## 3. Dictionnaire des variables"),
("code", '''desc = {
 "recency": "jours entre le dernier achat et T",
 "frequency": "nombre de factures de vente avant T (les retours ne comptent pas)",
 "monetary": "CA net = ventes - retours",
 "gross_revenue": "CA des ventes seules",
 "return_rate": "retours / ventes (borné à [0, 1])",
 "tenure_days": "ancienneté : jours depuis le premier achat",
 "avg_basket": "panier moyen (CA par facture)",
 "avg_items_per_invoice": "articles par facture",
 "avg_qty_per_line": "quantité moyenne par ligne (sert à détecter les grossistes)",
 "n_unique_products / n_unique_categories": "diversité des produits / catégories achetés",
 "ipt_mean": "intervalle moyen entre deux jours d'achat (NaN si un seul achat)",
 "is_single_purchase": "1 si un seul jour d'achat",
 "recency_over_ipt": "récence / intervalle habituel (>1 : en retard sur son rythme)",
 "revenue_recent / revenue_prev": "CA des 90 derniers jours / des 90 jours précédents",
 "trend": "(recent - prev) / (|recent| + |prev|) dans [-1, 1]",
 "is_wholesale": "1 si quantité moyenne par ligne >= seuil fixe (heuristique B2B)",
 "country_group / is_uk": "UK / Europe / Rest of World ; indicateur UK",
 "cat_share_*": "part du CA par catégorie dérivée de la Description (approximative)",
 "snapshot_month": "mois de T (saisonnalité)",
}
pd.DataFrame({"variable": desc.keys(), "définition": desc.values()})'''),
("md", "## 4. Valeurs manquantes à traiter à la phase de modélisation"),
("code", '''na = f.isna().mean().mul(100).round(1); na[na > 0]'''),
("md", """`ipt_mean` et `recency_over_ipt` sont manquants pour les clients à achat unique : c'est **normal** (information réelle, signalée par `is_single_purchase`). L'imputation se fera **dans le pipeline de modélisation, ajustée sur le train uniquement**."""),
("md", "## 5. Catégories dérivées de la Description : validation sur échantillon étiqueté à la main"),
("code", '''print((100 * df.drop_duplicates("StockCode")["Category"].value_counts(normalize=True)).round(1))
sample = export_validation_sample(df, n=150, seed=cfg["seed"])
path = OUT / "validation_categories_sample.csv"
if not path.exists():
    sample.to_csv(path, index=False)
    print("\\nÉchantillon écrit dans", path, "\\n-> remplis la colonne 'Category_manual' à la main (valeurs : voir CATEGORY_KEYWORDS ou OTHER), puis relance la cellule suivante.")
sample.head(10)'''),
("code", '''lab = pd.read_csv(path)
lab = lab[lab["Category_manual"].notna() & (lab["Category_manual"].astype(str).str.strip() != "")]
if len(lab) >= 30:
    acc = (lab["Category_auto"].str.upper() == lab["Category_manual"].astype(str).str.upper().str.strip()).mean()
    print(f"Précision de la catégorisation automatique sur {len(lab)} produits étiquetés : {100*acc:.1f} %")
    print(pd.crosstab(lab["Category_manual"].str.upper(), lab["Category_auto"]))
else:
    print(f"{len(lab)} produits étiquetés : étiquette au moins 100 produits pour mesurer la précision.")'''),
("md", "## 6. Table pour la segmentation (`customer_features`)\nFeatures sur tout l'historique, T = fin des données + 1 jour. Cette table est **descriptive** (pas de cible)."),
("code", '''seg = build_customer_features(df, cfg)
print(f"{len(seg):,} clients | T = {seg['snapshot_date'].iloc[0]:%Y-%m-%d}")
print(save_table(seg, "customer_features", cfg))
seg[["recency", "frequency", "monetary", "avg_basket", "tenure_days", "return_rate"]].describe().round(1)'''),
("md", "## 7. Table pour le churn (`churn_dataset` : client × snapshot)"),
("code", '''churn = build_churn_dataset(df, cfg, HORIZON)
print(save_table(churn, "churn_dataset", cfg))
tab = churn.groupby(["split", "snapshot_date"])["churn"].agg(lignes="size", taux_churn="mean").round(3)
tab'''),
("code", '''fig, ax = plt.subplots(); churn.groupby("snapshot_date")["churn"].mean().plot(marker="o", ax=ax)
ax.set_ylabel("taux de churn"); ax.set_title(f"Taux de churn par snapshot (horizon {HORIZON} j)")
plt.savefig(FIG / "taux_churn_snapshots.png", dpi=150); plt.show()
print("Un taux de churn stable d'un snapshot à l'autre est bon signe ; une forte variation est à expliquer (saisonnalité).")'''),
("md", "## 8. Analyse de sensibilité de l'horizon (60 / 90 / 120 jours)"),
("code", '''rows = []
for h in cfg["churn"]["sensitivity_horizons"]:
    for _, r in snapshot_schedule(cfg, df, h).iterrows():
        if not r["fenetre_complete"]:
            continue
        feats = build_features(df, r["T"], cfg)
        t = build_target(df, r["T"], h, feats["CustomerID"])
        rows.append({"horizon": h, "T": r["T"].date(), "split": r["split"], "taux_churn": t["churn"].mean()})
sens = pd.DataFrame(rows)
sens.pivot_table(index=["T", "split"], columns="horizon", values="taux_churn").round(3)'''),
("md", """*Remarque :* pour l'horizon 120 j, certains snapshots tardifs sont exclus (fenêtre incomplète) et le contrôle de purge de `build_churn_dataset` doit être relancé avec ce même horizon avant la phase 6.

## 9. Corrélations entre variables (anticiper la multicolinéarité)"""),
("code", '''num = ["recency", "frequency", "monetary", "avg_basket", "tenure_days", "return_rate", "n_unique_products", "ipt_mean", "trend"]
corr = np.log1p(seg[num].clip(lower=0)).corr(method="spearman").round(2)
fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=60, ha="right"); ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
for i in range(len(num)):
    for j in range(len(num)): ax.text(j, i, corr.iloc[i, j], ha="center", va="center", fontsize=7)
plt.colorbar(im); plt.title("Corrélation de Spearman (log1p)"); plt.tight_layout(); plt.savefig(FIG / "correlations.png", dpi=150); plt.show()'''),
("md", """## 10. Livrables de la phase 4
- `data/processed/customer_features` (segmentation)
- `data/processed/churn_dataset` (churn, colonne `split` = train / validation / test)
- `tests/test_features_no_leakage.py` : `pytest tests -q` doit afficher « passed »

**Rappel :** si les segments K-Means sont utilisés plus tard comme variable du modèle de churn, ils doivent être recalculés à chaque snapshot (sinon fuite)."""),
], "03_feature_engineering.ipynb")
print("3 notebooks écrits dans", NB)
