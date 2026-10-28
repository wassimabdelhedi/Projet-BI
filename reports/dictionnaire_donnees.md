# Dictionnaire de données

## 1. Fichier brut : `online_retail_II.xlsx`

Deux feuilles (`Year 2009-2010`, `Year 2010-2011`) avec les mêmes 8 attributs. Environ 1,07 million de lignes (à confirmer par le notebook 01).

| Attribut brut | Nom standardisé | Type | Description et pièges |
|---|---|---|---|
| `Invoice` (`InvoiceNo`) | `Invoice` | texte | Numéro de facture à 6 chiffres. Préfixe **`C`** = annulation/retour. Préfixe **`A`** = ajustement comptable (à exclure). Une facture a plusieurs lignes. |
| `StockCode` | `StockCode` | texte | Code produit alphanumérique. Codes non-produits : `POST`, `D`, `M`, `BANK CHARGES`, `DOT`, `CRUK`, `TEST001`… Un même code peut avoir plusieurs descriptions. |
| `Description` | `Description` | texte | Libellé du produit. Valeurs manquantes (souvent avec prix 0), casse incohérente, annotations de stock (« damaged »…). |
| `Quantity` | `Quantity` | entier | Négative pour les retours et ajustements. Valeurs extrêmes (± 80 995). |
| `InvoiceDate` | `InvoiceDate` | datetime | Précision à la minute. |
| `Price` (`UnitPrice`) | `Price` | décimal | Prix unitaire en livres sterling (GBP). Contient des 0 et quelques valeurs négatives. |
| `Customer ID` (`CustomerID`) | `CustomerID` | entier nullable (`Int64`) | Lu en `float64` par pandas (13085.0) à cause des manquants ; converti. Environ 20 à 23 % de lignes sans identifiant. |
| `Country` | `Country` | texte | Pays du client. Royaume-Uni très dominant. Valeurs ambiguës : `Unspecified`, `European Community`. |
| (ajoutée) | `SourceSheet` | texte | Feuille d'origine, pour auditer le chevauchement. Supprimée au nettoyage. |

## 2. Colonnes dérivées (`transactions_clean`)

| Colonne | Définition |
|---|---|
| `IsReturn` | `Invoice` commence par `C` |
| `Revenue` | `Quantity × Price` (négatif pour un retour) |
| `HasCustomer` | `CustomerID` non nul |
| `Description` | Description canonique : la plus fréquente par `StockCode`, en majuscules, espaces normalisés |
| `Category` | Catégorie dérivée de la `Description` par mots-clés (approximative) : `CHRISTMAS`, `BAGS`, `CANDLES_LIGHTING`, `JEWELLERY_ACCESSORIES`, `STATIONERY_PAPER`, `TOYS_GAMES`, `GARDEN_OUTDOOR`, `KITCHEN_DINING`, `STORAGE`, `HOME_DECOR`, `OTHER` |
| `InvoiceDay`, `Year`, `Month`, `YearMonth`, `DayOfWeek`, `Hour` | Extraits de `InvoiceDate` |
| `IsQtyOutlier`, `IsPriceOutlier` | Valeur au-dessus du percentile 99,9 des ventes (marquage, pas de suppression) |

## 3. Variables client (`customer_features`, `churn_dataset`)

Toutes calculées avec les lignes **antérieures à T** uniquement.

| Variable | Définition |
|---|---|
| `CustomerID` | Identifiant client |
| `recency` | Jours entre le dernier achat et T |
| `frequency` | Nombre de factures de vente |
| `monetary` | CA net = ventes - retours |
| `gross_revenue` | CA des ventes seules |
| `return_amount`, `return_rate` | Montant retourné ; retours / ventes (borné à [0, 1]) |
| `tenure_days` | Jours depuis le premier achat |
| `avg_basket` | CA moyen par facture |
| `avg_items_per_invoice` | Articles moyens par facture |
| `avg_qty_per_line` | Quantité moyenne par ligne |
| `n_unique_products`, `n_unique_categories` | Diversité des produits et des catégories |
| `ipt_mean` | Intervalle moyen entre deux jours d'achat (manquant si un seul achat) |
| `is_single_purchase` | 1 si un seul jour d'achat |
| `recency_over_ipt` | `recency / ipt_mean` (> 1 : en retard sur son rythme habituel) |
| `revenue_recent`, `revenue_prev` | CA des 90 derniers jours / des 90 jours précédents |
| `trend` | `(recent - prev) / (|recent| + |prev|)`, dans [-1, 1] |
| `is_wholesale` | 1 si `avg_qty_per_line` ≥ seuil fixe (config) : heuristique B2B |
| `country_group`, `is_uk` | UK / Europe / Rest of World ; indicateur UK |
| `cat_share_*` | Part du CA par catégorie dérivée |
| `snapshot_month`, `snapshot_date` | Mois et date de T |

## 4. Cible et colonnes de découpage (`churn_dataset` uniquement)

| Colonne | Définition |
|---|---|
| `churn` | 1 si aucun achat (vente) dans `[T, T + horizon)`, sinon 0 |
| `split` | `train`, `validation` ou `test`, selon le calendrier de `config.yaml` |

## 5. Fichiers produits

| Fichier | Contenu |
|---|---|
| `data/processed/transactions_clean.parquet` | Une ligne par ligne de facture, après nettoyage |
| `data/processed/customer_features.parquet` | Un client par ligne, T = fin des données + 1 jour (segmentation) |
| `data/processed/churn_dataset.parquet` | Une ligne par (client, snapshot) avec features, `churn`, `split` |
| `outputs/audit_summary.csv` | Chiffres clés de qualité avant nettoyage |
| `outputs/exclusion_log.csv` | Lignes exclues, règle, % et CA concerné |
| `outputs/validation_categories_sample.csv` | Échantillon de produits à étiqueter à la main |
