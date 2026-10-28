# Projet 10 : Customer Segmentation, Churn Prediction & Product Recommendation
## Document de cadrage (Phase 1)

> Les décisions ci-dessous sont **figées avant toute modélisation** pour éviter d'ajuster les choix aux résultats.
> Les cases « à confirmer » sont des chiffres à remplir après exécution des notebooks 01 et 02.

---

## 1. Contexte et question centrale

Un détaillant en ligne britannique de cadeaux (clientèle mixte particuliers / grossistes) dispose de deux ans de transactions.
Il veut **fidéliser** ses clients, **anticiper** ceux qui vont cesser d'acheter et **personnaliser** ses offres.

**Question de recherche :**
> Comment identifier des groupes de clients homogènes, anticiper la perte de clients et recommander les bons produits à chacun, de façon fiable et exploitable par une équipe marketing ?

## 2. Objectifs et résultats attendus

| # | Objectif métier | Résultat technique attendu | Phase |
|---|---|---|---|
| O1 | Connaître les types de clients | Segments interprétables et stables | 5 |
| O2 | Réduire la perte de clients | Score de risque de churn par client, validé dans le temps | 6 |
| O3 | Personnaliser l'offre | Top 5 produits par client ou segment | 7 |
| O4 | Piloter par la donnée | Tableau de bord décisionnel (CA à risque, clients prioritaires) | 8-9 |

**Profils cibles :** Champions, clients fidèles, occasionnels, à risque, perdus, nouveaux (noms définitifs fixés d'après les profils réels, jamais avant).

**Valeur attendue :** répondre à « *combien de chiffre d'affaires est à risque et sur qui agir en priorité ?* ».

## 3. Décisions de cadrage

| Décision | Choix | Justification |
|---|---|---|
| Dataset | **Online Retail II** (UCI), deux feuilles fusionnées, du 01/12/2009 au 09/12/2011 | Identifiant client, dates, retours, deux ans d'historique (voir section 6) |
| Unité de modélisation | **Client** avec `CustomerID` non nul | Le RFM, le churn et la recommandation exigent un historique par client |
| Lignes sans `CustomerID` | Conservées pour l'analyse globale des ventes, **exclues** de la modélisation client | Ne jamais inventer un identifiant (faux clients, RFM faussé) |
| Définition d'un achat | Une facture de **vente** (non `C`). Un retour n'est pas un achat | Évite de compter une annulation comme une activité |
| **Définition du churn** | Aucun achat pendant les **90 jours** suivant la date d'observation T | Horizon à confirmer par le percentile 80 des délais inter-achats (notebook 02) |
| Sensibilité | Horizons 60 / 90 / 120 jours | Montrer la robustesse de la conclusion |
| Retours (facture `C`) | Conservés ; déduits du CA net (`monetary`), exclus de `frequency` ; `return_rate` en variable | Politique cohérente et traçable |
| Codes non-produits | Exclus (`POST`, `D`, `M`, `BANK CHARGES`, `DOT`, `CRUK`, `AMAZONFEE`, `TEST*`…) | Frais et ajustements ne sont pas des achats de produits |
| B2B / B2C | Variable `is_wholesale` (seuil fixe de quantité moyenne par ligne), log-transform, test de sensibilité sans le top 1 % | Quelques grossistes dominent le CA |
| Pays | Exclu du clustering ; regroupé UK / Europe / Reste du monde pour le profilage | Un pays très dominant apporte peu d'information |
| Catégories produit | Dérivées de `Description` par mots-clés, **validées sur un échantillon étiqueté à la main** | Pas de catégorie native |
| Validation | **Temporelle uniquement**, jamais aléatoire, avec purge entre splits | Éviter la fuite de données |

## 4. Définitions opérationnelles

- **Date d'observation T (snapshot)** : date fictive « d'aujourd'hui ». Les variables n'utilisent que `InvoiceDate < T`.
- **Cible** : `churn = 1` si aucun achat dans `[T, T + 90 j)`, sinon `0`. Seuls les clients ayant au moins un achat avant T sont observés.
- **Recency** : jours entre le dernier achat et T. **Frequency** : nombre de factures de vente avant T. **Monetary** : CA net (ventes - retours) avant T.
- **Client actif à T** : au moins une facture de vente avant T.
- **Horloge** : toujours la date de la transaction, jamais `datetime.now()`.

## 5. Indicateurs de succès (objectifs, pas garanties)

| Domaine | Indicateur | Objectif |
|---|---|---|
| Segmentation | Silhouette | > 0,30 |
| Segmentation | Stabilité (ARI moyen entre rééchantillonnages) | > 0,80 |
| Churn | ROC-AUC et PR-AUC sur le **test temporel** | Supérieurs aux baselines (Dummy, règle « récence > X j ») ; ROC-AUC visée > 0,75 |
| Churn | Calibration, lift par décile | Rapportés |
| Recommandation | Precision@5 / HitRate@5 | Supérieurs à la popularité globale |
| BI | Cohérence Python ↔ Power BI | 100 % sur CA total, nombre de clients, nombre de commandes |
| Qualité | Test anti-fuite automatisé | `pytest` vert |

**Règle d'intégrité :** un objectif non atteint est rapporté tel quel et expliqué. On ne modifie pas la définition du churn ni le calendrier pour améliorer un score.

## 6. Choix du dataset

| Dataset évalué | Verdict | Raison |
|---|---|---|
| `merged_sales_data.csv` (ventes 2019, magasin d'électronique) | Écarté | Aucun identifiant client : RFM, churn et recommandation par client impossibles |
| `new_retail_data.csv` (302 010 lignes) | Écarté | **Synthétique** : 65 à 87 % des `Customer_ID` ont plusieurs noms, e-mails, âges ou pays ; ROC-AUC d'un modèle RFM ≈ 0,50 (aucun signal) ; prix indépendants du produit ; une seule année |
| **Online Retail II** | **Retenu** | `CustomerID`, dates, factures d'annulation, deux ans, très documenté |
| Olist, Dunnhumby, REES46, Instacart | Alternatives | Écartés (churn peu pertinent, lourdeur, ou absence de temporalité) |

Le test de validité (cohérence des identifiants, signal RFM) peut être refait sur tout nouveau dataset avant de l'adopter.

## 7. Calendrier des snapshots et règle de purge

Les données s'arrêtent le **09/12/2011** : aucun T postérieur au **09/09/2011** n'a de fenêtre de 90 jours complète.

| Split | Dates T | Fin de la cible |
|---|---|---|
| Train | 30/06/2010, 31/07, 31/08, 30/09, 31/10, 30/11, 31/12/2010 | jusqu'au 31/03/2011 |
| Validation | 31/03/2011 | 29/06/2011 |
| Test | 30/06/2011, 09/09/2011 | 28/09/2011 et 08/12/2011 |

```
2010-06 ........ 2010-12-31 | 2011-03-31 | 2011-06-30 ....... 2011-09-09
   T train (x7) ───cible──►|  T valid ──►| T test 1 ─►  T test 2 ──► (fin des données 2011-12-09)
```

**Purge :** la cible d'un split se termine **avant** le premier T du split suivant (31/03/2011 ≤ 31/03/2011 et 29/06/2011 < 30/06/2011). Les mois de janvier et février 2011 ne sont volontairement pas utilisés comme T d'entraînement : leurs cibles chevaucheraient la validation. Cette règle est **vérifiée automatiquement** (`check_split_purge`).

Les deux snapshots de test ont des fenêtres qui se recouvrent partiellement : leurs résultats sont rapportés séparément, non comme deux échantillons indépendants.

## 8. Politique de nettoyage

| Étape | Règle | Effet |
|---|---|---|
| 1 | Doublons exacts supprimés (dont chevauchement des feuilles en décembre 2010) | Ligne exclue |
| 2 | Factures `A` (ajustement de dettes) | Exclue |
| 3 | Codes non-produits | Exclue |
| 4 | `Price <= 0` | Exclue |
| 5 | `Quantity = 0` | Exclue |
| 6 | `Quantity < 0` sans préfixe `C` (ajustement de stock) | Exclue |
| 7 | Facture `C` avec `Quantity > 0` (incohérent) | Exclue |
| - | Factures `C` cohérentes | **Conservées**, `IsReturn = True`, `Revenue` négatif |
| - | Lignes sans `CustomerID` | **Conservées**, `HasCustomer = False` |
| - | Outliers de quantité / prix | **Marqués** (`IsQtyOutlier`, `IsPriceOutlier`), non supprimés |

Le tableau « lignes exclues et pourquoi » est produit automatiquement dans `outputs/exclusion_log.csv`.

## 9. Risques et limites anticipés

| Limite | Mitigation retenue | Où la traiter |
|---|---|---|
| Environ 20 à 23 % de lignes sans client (à confirmer) | Exclusion de la modélisation client, usage complet pour les ventes, biais quantifié | Notebook 01 |
| Pas de catégories natives | Mots-clés + validation manuelle de la précision | Notebook 03 |
| Mix B2B / B2C | `is_wholesale`, log-transform, test sans top 1 % | Phases 5-6 |
| Deux ans seulement | Snapshots multiples, validation temporelle, sensibilité 60/90/120 j, variable `snapshot_month` | Phase 4, 6 |
| Pays dominant (Royaume-Uni) | Pays exclu du clustering, évaluation UK / hors UK | Phase 6 |
| Churn défini par une règle métier | Sensibilité à l'horizon, baselines | Phase 6 |
| Même client dans plusieurs snapshots (observations corrélées) | `GroupKFold` par client en complément du split temporel | Phase 6 |
| Décembre 2011 incomplet | Aucun T après le 09/09/2011 | Phase 4 |

## 10. Éthique et données personnelles

Les données sont des identifiants numériques anonymisés, sans nom ni coordonnées. Le risque d'usage concerne la **sur-sollicitation** des clients identifiés « à risque » : les campagnes doivent être limitées en fréquence et le score ne doit pas servir à des décisions discriminatoires.

## 11. Livrables et planning des phases 1 à 4

| Phase | Livrable | Fichier |
|---|---|---|
| 1 | Document de cadrage, dictionnaire de données | `reports/01_cadrage.md`, `reports/dictionnaire_donnees.md`, `config.yaml` |
| 2 | Audit qualité, tableau des exclusions | `notebooks/01_data_audit.ipynb`, `outputs/audit_summary.csv`, `outputs/exclusion_log.csv` |
| 3 | Données propres, EDA | `src/cleaning.py`, `data/processed/transactions_clean.parquet`, `notebooks/02_eda.ipynb` |
| 4 | Features et cible sans fuite, tests | `src/features.py`, `customer_features`, `churn_dataset`, `notebooks/03_feature_engineering.ipynb`, `tests/` |

Durée indicative : phase 1 (1 jour), phase 2 (2 jours), phase 3 (3 à 4 jours), phase 4 (4 à 5 jours).
