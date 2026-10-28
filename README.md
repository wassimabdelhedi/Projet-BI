# Projet 10 : Customer Segmentation, Churn Prediction & Product Recommendation
**Contenu de cette archive : phases 1 à 4** (cadrage, audit, nettoyage + EDA, features sans fuite).
Les phases 5 à 11 (segmentation, churn, recommandation, Power BI, validation) viennent ensuite.

## Ce qui est livré

| Phase | Livrable | Où |
|---|---|---|
| 1. Cadrage | Document de cadrage, dictionnaire de données, configuration | `reports/01_cadrage.md`, `reports/dictionnaire_donnees.md`, `config.yaml` |
| 2. Audit | Notebook d'audit, tableau de synthèse, tableau « lignes exclues et pourquoi » | `notebooks/01_data_audit.ipynb`, `outputs/` |
| 3. Nettoyage + EDA | Module de nettoyage, notebook EDA (ventes, saisonnalité, Pareto, délais inter-achats, cohortes, B2B) | `src/cleaning.py`, `notebooks/02_eda.ipynb` |
| 4. Features | Variables RFM et comportementales, cible churn, snapshots, tests anti-fuite | `src/features.py`, `notebooks/03_feature_engineering.ipynb`, `tests/` |

## Important : ce qui a été testé et ce qui ne l'a pas été

- Le code et les notebooks ont été exécutés **de bout en bout sur un jeu de données synthétique** (`scripts/generate_sample_data.py`) qui imite le format et les défauts d'Online Retail II. Les 14 tests passent.
- Ils n'ont **pas** été exécutés sur le vrai fichier `online_retail_II.xlsx` (non disponible ici). Aucun chiffre de cette archive n'est un résultat sur les vraies données.
- Les notebooks sont livrés **sans sorties** : c'est à toi de les exécuter et de compléter les sections « À retenir » avec **tes** chiffres.
- Les seuils (horizon de 90 jours, seuil grossiste de 15 articles par ligne) sont des choix de départ à confirmer avec le notebook 02.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

1. Télécharge **Online Retail II** (UCI : archive.ics.uci.edu/dataset/502/online+retail+ii, ou Kaggle) et place le fichier dans `data/raw/`.
   - Garde le nom `online_retail_II.xlsx`, ou change `paths.raw_file` dans `config.yaml` (un `.csv` avec les mêmes colonnes fonctionne aussi).
2. Lance tout le pipeline des phases 2 à 4 :
   ```bash
   python run_pipeline.py
   ```
   La première lecture du xlsx prend 1 à 3 minutes ; un cache parquet est ensuite utilisé.
3. Ouvre les notebooks **dans l'ordre** : `jupyter lab`, puis `01_data_audit`, `02_eda`, `03_feature_engineering`.
4. Vérifie l'absence de fuite : `pytest tests -q` doit afficher `passed`.

### Test rapide sans le vrai fichier (données synthétiques)

```bash
python scripts/generate_sample_data.py --out data/raw/sample_retail.xlsx
python run_pipeline.py --raw data/raw/sample_retail.xlsx
```
Ces résultats servent **uniquement** à vérifier que l'installation fonctionne.

## Structure

```
projet10-customer-analytics/
├── config.yaml                 # dates des snapshots, horizon de churn, seuils, chemins, seed
├── run_pipeline.py             # phases 2 a 4 en une commande
├── requirements.txt
├── data/ (raw, interim, processed)
├── notebooks/                  # 01_data_audit, 02_eda, 03_feature_engineering
├── src/
│   ├── config.py               # chargement config, sauvegarde parquet/csv
│   ├── cleaning.py             # chargement, audit, politique de nettoyage, journal des exclusions
│   ├── categories.py           # categories derivees de Description (mots-cles)
│   └── features.py             # build_features(df, T), build_target(df, T, h), snapshots, purge
├── tests/                      # tests anti-fuite et regles de nettoyage
├── scripts/                    # generateur de donnees synthetiques, constructeur de notebooks
├── reports/                    # cadrage, dictionnaire, figures
└── outputs/                    # tableaux CSV (audit, exclusions)
```

## Principes de conception à connaître

- **Anti-fuite :** `build_features(df, T)` n'utilise que `InvoiceDate < T`. `build_target` regarde `[T, T+h)`. Le test `test_features_ignore_everything_after_T` ajoute volontairement un achat futur énorme et vérifie que les features ne changent pas.
- **Purge :** la cible d'un split finit avant le premier T du split suivant (`check_split_purge`). Les snapshots de janvier et février 2011 sont donc volontairement absents du train.
- **Retours :** conservés, déduits du CA net, exclus de la fréquence. Un retour n'est pas un achat pour la cible.
- **Imputation :** `ipt_mean` et `recency_over_ipt` sont manquants pour les clients à achat unique. L'imputation se fera dans le pipeline de modélisation, ajustée sur le train seul.
- **Segments et churn :** si les segments K-Means servent plus tard de variable au modèle de churn, ils doivent être recalculés à chaque snapshot.
- **Catégories :** approximatives. Étiquette ~150 produits à la main (`outputs/validation_categories_sample.csv`) pour mesurer leur précision et la citer dans les limites.

## Dépannage

| Problème | Solution |
|---|---|
| `FileNotFoundError ... online_retail_II.xlsx` | Place le fichier dans `data/raw/` ou règle `paths.raw_file` |
| Lecture du xlsx très lente | Normal la première fois (cache ensuite). Tu peux aussi convertir en `.csv` |
| `ImportError: pyarrow` | `pip install pyarrow`, ou mets `processed_format: csv` dans `config.yaml` |
| Colonnes manquantes | Les noms `InvoiceNo`, `UnitPrice`, `Customer ID` sont acceptés ; sinon renomme vers `Invoice`, `Price`, `CustomerID` |
| Notebook : `ModuleNotFoundError: src` | Lance Jupyter depuis la racine du projet ou depuis `notebooks/` |
| Erreur « Fenêtre de cible incomplète » | Un snapshot dépasse la fin des données : corrige la date dans `config.yaml` |

## Prochaines étapes
Phase 5 : segmentation (K-Means, stabilité) · Phase 6 : churn (baselines, LightGBM, SHAP) · Phase 7 : recommandation · Phase 8-9 : synthèse et Power BI · Phase 10 : temps réel simulé (bonus).
