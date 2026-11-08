"""Phase 5 : Segmentation client par K-Means.

Principe :
- Preprocessing (log1p sur vars asymetriques, winsorisation au 99e pct, StandardScaler)
  ajuste UNE SEULE FOIS sur un seul jeu, applique sur tout autre (pas de fuite).
- Le pays est INTERDIT dans le clustering (ValueError si present) ; il sert uniquement
  au profilage apres coup.
- Choix de K : Elbow (inertie), Silhouette, Davies-Bouldin.
- Comparaison : Agglomerative Clustering, GMM.
- RFM par quantiles (reference metier, interpretabilite garantie).
- Stabilite : bootstrap + ARI ; comparaison entre deux snapshots.
- Nommage automatique des segments base sur les profils reels (mediane de recency,
  frequency, monetary) ; ecrasable via NAME_OVERRIDES.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------- constantes
COUNTRY_COLS = {"country_group", "is_uk"}  # colonnes de pays : interdites dans le clustering
META_COLS = {"snapshot_date", "snapshot_month", "CustomerID"}

# Variables continues a log-transformer avant winsorisation
DEFAULT_LOG_VARS = [
    "recency", "frequency", "monetary", "gross_revenue", "avg_basket",
    "tenure_days", "return_amount", "avg_qty_per_line", "avg_items_per_invoice",
    "n_unique_products", "n_unique_categories", "ipt_mean", "recency_over_ipt",
    "revenue_recent", "revenue_prev",
]

# Nommage automatique des segments (peut etre ecrase par NAME_OVERRIDES)
# Logique : on classe les segments par recency croissante et monetary decroissant.
_AUTO_NAMES = {
    "champion": "Champions",
    "at_risk": "A risque",
    "loyal": "Fideles",
    "lost_high": "Perdus (CA eleve)",
    "lost_low": "Perdus (CA faible)",
    "new": "Nouveaux",
    "occasional": "Occasionnels",
}


# ====================================================================== preprocessing
class SegmentationPreprocessor:
    """Transforme un DataFrame client en matrice numerique prete pour le clustering.

    Etapes :
        1. Verification : aucune colonne pays dans les features.
        2. Imputation de ipt_mean (NaN = client a achat unique) par 0 en log-espace.
        3. log1p sur les variables asymetriques definies dans log_vars.
        4. Winsorisation au quantile winsor_quantile.
        5. StandardScaler (zero-mean, unit-variance).

    Toutes les etapes sont ajustees sur le meme jeu (fit) et appliquees ulterieurement
    (transform). Ne jamais ré-ajuster sur un jeu de test ou de validation.
    """

    def __init__(
        self,
        log_vars: Optional[List[str]] = None,
        winsor_quantile: float = 0.99,
        exclude_cols: Optional[List[str]] = None,
    ):
        self.log_vars = log_vars or DEFAULT_LOG_VARS
        self.winsor_quantile = winsor_quantile
        self.exclude_cols = set(exclude_cols or []) | COUNTRY_COLS | META_COLS
        self.feature_names_: List[str] = []
        self.winsor_limits_: Dict[str, Tuple[float, float]] = {}
        self._scaler = StandardScaler()
        self._fitted = False

    # ---- helpers privees
    def _check_no_country(self, df: pd.DataFrame) -> None:
        bad = COUNTRY_COLS & set(df.columns)
        if bad:
            raise ValueError(
                f"Les colonnes de pays {bad} ne doivent PAS etre dans les features "
                "du clustering. Utilisez-les seulement pour profiler les segments apres coup."
            )

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_no_country(df)
        cols = [c for c in df.columns if c not in self.exclude_cols]
        # Garder uniquement les colonnes numeriques
        num = df[cols].select_dtypes(include=[np.number])
        return num

    def _log_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.log_vars:
            if col not in X.columns:
                continue
            # ipt_mean : NaN = client a achat unique → aucune info sur l'intervalle → 0 apres log
            if col == "ipt_mean":
                X[col] = X[col].fillna(0.0)
            # monetary peut etre negatif (retours) : shift pour rester positif
            vals = X[col]
            if vals.min() <= 0:
                shift = abs(vals.min()) + 1e-6
                X[col] = np.log1p(vals + shift)
            else:
                X[col] = np.log1p(vals)
        # recency_over_ipt : peut etre NaN (meme raison que ipt_mean)
        if "recency_over_ipt" in X.columns:
            X["recency_over_ipt"] = X["recency_over_ipt"].fillna(0.0)
        return X

    def _winsorize(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if fit:
                lo = float(X[col].quantile(1 - self.winsor_quantile))
                hi = float(X[col].quantile(self.winsor_quantile))
                self.winsor_limits_[col] = (lo, hi)
            lo, hi = self.winsor_limits_.get(col, (None, None))
            if lo is not None:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X

    # ---- API publique
    def fit(self, df: pd.DataFrame) -> "SegmentationPreprocessor":
        num = self._select_features(df)
        self.feature_names_ = list(num.columns)
        X = self._log_transform(num)
        X = self._winsorize(X, fit=True)
        self._scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Appelez fit() avant transform().")
        # Verifier l'absence de colonnes pays AVANT d'extraire feature_names_
        self._check_no_country(df)
        num = df[self.feature_names_].copy()
        num = self._log_transform(num)
        num = self._winsorize(num, fit=False)
        return self._scaler.transform(num)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)


# ====================================================================== choix de K
def evaluate_k_range(
    X: np.ndarray,
    k_range: List[int],
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Calcule l'inertie, Silhouette et Davies-Bouldin pour chaque K.

    Args:
        X: matrice normalise (n_samples, n_features).
        k_range: liste de valeurs de K a tester.
        random_state: graine.
        n_init: nombre d'initialisations K-Means.

    Returns:
        DataFrame avec colonnes [k, inertia, silhouette, davies_bouldin].
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=random_state)
        db = davies_bouldin_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "davies_bouldin": db})
    return pd.DataFrame(rows)


def suggest_k(metrics_df: pd.DataFrame) -> int:
    """Renvoie le K suggere automatiquement (meilleur Silhouette).

    Note : le choix final peut legitimement differir du K suggere si les profils
    metier sont plus interpretables a un autre K (documenter le choix).
    """
    return int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])


# ====================================================================== K-Means
def fit_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeans:
    """Ajuste un K-Means et renvoie le modele ajuste."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    km.fit(X)
    return km


def kmeans_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Renvoie {silhouette, davies_bouldin, inertia} pour un jeu de labels."""
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    db = davies_bouldin_score(X, labels)
    return {"silhouette": round(sil, 4), "davies_bouldin": round(db, 4)}


# ====================================================================== comparaison algorithmes
def compare_algorithms(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare K-Means, Agglomerative et GMM sur les memes donnees.

    Returns:
        DataFrame [algo, silhouette, davies_bouldin, ari_vs_kmeans].
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels_km = km.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_agg = agg.fit_predict(X)

    gmm = GaussianMixture(n_components=k, n_init=5, random_state=random_state)
    labels_gmm = gmm.fit_predict(X)

    rows = []
    for name, lbl in [("KMeans", labels_km), ("Agglomerative (Ward)", labels_agg), ("GMM", labels_gmm)]:
        sil = silhouette_score(X, lbl, sample_size=min(5000, len(X)), random_state=random_state)
        db = davies_bouldin_score(X, lbl)
        ari = adjusted_rand_score(labels_km, lbl) if name != "KMeans" else 1.0
        rows.append({"algo": name, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
                     "ari_vs_kmeans": round(ari, 4)})
    return pd.DataFrame(rows)


# ====================================================================== RFM par quantiles
def rfm_segments(df: pd.DataFrame, n_quantiles: int = 4) -> pd.Series:
    """Segmentation RFM par quantiles (reference metier, sans ML).

    Chaque dimension est decoupee en n_quantiles niveaux :
        recency  : score inverse (1 = recent, n = ancien)
        frequency: score direct  (1 = rare,   n = frequent)
        monetary : score direct  (1 = faible,  n = eleve)

    Le segment final est le label texte du score RFM concatene (ex: '444', '111').
    """
    r = pd.qcut(df["recency"], n_quantiles, labels=False, duplicates="drop")
    # Recency : 1 = meilleur (le plus recent) => inverser
    r_score = (r.max() - r + 1).rename("R_score")

    f = pd.qcut(df["frequency"].rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    f_score = (f + 1).rename("F_score")

    m = pd.qcut(df["monetary"].rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    m_score = (m + 1).rename("M_score")

    rfm = pd.concat([r_score, f_score, m_score], axis=1).fillna(1).astype(int)
    rfm["rfm_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["rfm_total"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    # Segments metier RFM (heuristique classique)
    def label(row):
        r_, f_, m_ = row["R_score"], row["F_score"], row["M_score"]
        total = row["rfm_total"]
        maxq = n_quantiles
        if r_ == maxq and f_ >= maxq - 1 and m_ >= maxq - 1:
            return "Champions RFM"
        if r_ >= maxq - 1 and f_ >= maxq - 1:
            return "Fideles RFM"
        if r_ <= 2 and f_ <= 2:
            return "Perdus RFM"
        if r_ <= 2:
            return "A risque RFM"
        if f_ == 1:
            return "Nouveaux RFM"
        return "Potentiels RFM"

    rfm["rfm_segment"] = rfm.apply(label, axis=1)
    return rfm


# ====================================================================== nommage des segments
def name_segments(
    df_with_labels: pd.DataFrame,
    label_col: str = "segment_id",
    name_overrides: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """Genere des noms de segments bases sur les profils reels (mediane de RFM).

    Les segments sont d'abord tries par recency croissante (actif d'abord) puis
    monetary decroissant pour les egalites. Les 5 premiers recoit un nom canonique,
    les suivants ont un nom generique.

    Args:
        df_with_labels: DataFrame avec colonnes [recency, frequency, monetary, segment_id].
        label_col: nom de la colonne contenant les IDs de segment (entiers).
        name_overrides: {id_segment: nom_souhaite} pour ecraser les noms automatiques.

    Returns:
        {id_segment: nom_string}
    """
    profiles = (
        df_with_labels.groupby(label_col)[["recency", "frequency", "monetary"]]
        .median()
        .sort_values(["recency", "monetary"], ascending=[True, False])
    )
    canonical = ["Champions", "A risque", "Fideles", "Perdus (CA eleve)", "Perdus (CA faible)"]
    names = {}
    for rank, (seg_id, _) in enumerate(profiles.iterrows()):
        if rank < len(canonical):
            # Affiner : si recency eleve, c'est probablement "Perdus"
            rec = profiles.loc[seg_id, "recency"]
            freq = profiles.loc[seg_id, "frequency"]
            mon = profiles.loc[seg_id, "monetary"]
            # Heuristique de nommage
            if rec <= 30 and freq >= 5 and mon >= profiles["monetary"].quantile(0.6):
                names[seg_id] = "Champions"
            elif rec > 200:
                names[seg_id] = "Perdus (CA eleve)" if mon >= profiles["monetary"].median() else "Perdus (CA faible)"
            elif rec > 90 and freq <= 3:
                names[seg_id] = "A risque"
            elif freq >= 3 and rec <= 60:
                names[seg_id] = "Fideles"
            else:
                names[seg_id] = canonical[rank]
        else:
            names[seg_id] = f"Segment {seg_id}"
    if name_overrides:
        names.update(name_overrides)
    return names


# ====================================================================== stabilite bootstrap
def bootstrap_stability(
    df: pd.DataFrame,
    preprocessor: SegmentationPreprocessor,
    k: int,
    n_bootstrap: int = 50,
    sample_frac: float = 0.80,
    random_state: int = 42,
) -> Dict[str, object]:
    """Evalue la stabilite du clustering par bootstrap.

    Protocole :
        1. Ajuste K-Means sur le jeu complet (labels de reference).
        2. Pour chaque tirage : echantillonne sample_frac des clients, retransfo,
           ajuste un nouveau K-Means, calcule ARI entre les labels sur l'echantillon.
        3. Calcule aussi le Jaccard par segment.

    Returns:
        dict avec ari_mean, ari_std, ari_values (liste), jaccard_per_segment (dict).
    """
    rng = np.random.default_rng(random_state)
    X_full = preprocessor.transform(df)
    km_ref = fit_kmeans(X_full, k, random_state=random_state)
    labels_ref = km_ref.predict(X_full)

    ari_values = []
    # jaccard[seg] = liste de Jaccard sur chaque bootstrap
    jaccard_dict: Dict[int, List[float]] = {s: [] for s in range(k)}

    for i in range(n_bootstrap):
        idx = rng.choice(len(df), size=int(len(df) * sample_frac), replace=False)
        X_b = X_full[idx]
        km_b = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e6)))
        lbl_b = km_b.fit_predict(X_b)
        lbl_ref_b = labels_ref[idx]
        ari_values.append(adjusted_rand_score(lbl_ref_b, lbl_b))

        # Jaccard par segment (correspondance par mode)
        for seg in range(k):
            mask_ref = lbl_ref_b == seg
            if mask_ref.sum() == 0:
                continue
            # Chercher le label bootstrap dominant pour ce segment
            mode_b = pd.Series(lbl_b[mask_ref]).mode()
            if len(mode_b) == 0:
                continue
            mode_b = mode_b.iloc[0]
            mask_b = lbl_b == mode_b
            intersection = (mask_ref & mask_b).sum()
            union = (mask_ref | mask_b).sum()
            jaccard_dict[seg].append(intersection / union if union > 0 else 0.0)

    jaccard_summary = {
        seg: round(float(np.mean(vals)), 4) if vals else float("nan")
        for seg, vals in jaccard_dict.items()
    }
    return {
        "ari_mean": round(float(np.mean(ari_values)), 4),
        "ari_std": round(float(np.std(ari_values)), 4),
        "ari_values": [round(v, 4) for v in ari_values],
        "jaccard_per_segment": jaccard_summary,
    }


# ====================================================================== comparaison snapshots
def compare_snapshots(
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame,
    preprocessor: SegmentationPreprocessor,
    k: int,
    random_state: int = 42,
) -> Dict[str, object]:
    """Compare la partition du snapshot T1 appliquee a T2 vs un modele reajuste sur T2.

    Args:
        df_t1: features clients au snapshot T1.
        df_t2: features clients au snapshot T2 (memes colonnes).
        preprocessor: ajuste sur df_t1.
        k: nombre de clusters.

    Returns:
        dict avec ari, pct_same_segment, n_common_customers.
    """
    X1 = preprocessor.transform(df_t1)
    km1 = fit_kmeans(X1, k, random_state=random_state)

    X2 = preprocessor.transform(df_t2)
    km2 = fit_kmeans(X2, k, random_state=random_state)

    # Clients communs
    ids1 = set(df_t1["CustomerID"].values)
    ids2 = set(df_t2["CustomerID"].values)
    common = sorted(ids1 & ids2)
    if len(common) == 0:
        return {"ari": float("nan"), "pct_same_segment": float("nan"), "n_common_customers": 0}

    idx1 = df_t1[df_t1["CustomerID"].isin(common)].index
    idx2 = df_t2[df_t2["CustomerID"].isin(common)].index

    # Labels T1 (modele T1 applique sur T2 via predict)
    # On applique le preprocessor T1 (ajuste sur T1) sur les donnees T2
    lbl1_on2 = km1.predict(preprocessor.transform(df_t2.loc[idx2]))
    # Labels T2 (modele T2 ajuste sur T2)
    lbl2_on2 = km2.predict(X2[df_t2.index.isin(idx2)])

    ari = adjusted_rand_score(lbl1_on2, lbl2_on2)
    # % de clients au meme segment (en matchant les labels par best correspondance)
    # Simple : aligner par ARI n'est pas direct, on calcule le max matching
    from scipy.optimize import linear_sum_assignment
    conf = pd.crosstab(lbl1_on2, lbl2_on2).values
    row_ind, col_ind = linear_sum_assignment(-conf)
    n_matched = conf[row_ind, col_ind].sum()
    pct_same = n_matched / len(lbl1_on2)

    return {
        "ari": round(float(ari), 4),
        "pct_same_segment": round(float(pct_same), 4),
        "n_common_customers": len(common),
    }


# ====================================================================== export
def build_customer_segments(
    df: pd.DataFrame,
    preprocessor: SegmentationPreprocessor,
    km: KMeans,
    name_overrides: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Genere la table customer_segments (une ligne par client).

    Colonnes : CustomerID, segment_id, segment_name + X_pca_1, X_pca_2.
    df peut contenir country_group/is_uk (elles seront automatiquement exclues avant clustering).
    """
    # Retirer les colonnes pays avant de passer au preprocessor
    df_safe = df.drop(columns=[c for c in COUNTRY_COLS if c in df.columns])
    X = preprocessor.transform(df_safe)
    labels = km.predict(X)
    # Colonnes de base disponibles
    base_cols = ["CustomerID", "recency", "frequency", "monetary", "gross_revenue", "is_wholesale"]
    available = [c for c in base_cols if c in df.columns]
    out = df[available].copy()
    if "country_group" in df.columns:
        out["country_group"] = df["country_group"].values
    out["segment_id"] = labels
    names = name_segments(out, name_overrides=name_overrides)
    out["segment_name"] = out["segment_id"].map(names)
    # Coordonnees PCA 2D pour la visualisation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out["X_pca_1"] = coords[:, 0]
    out["X_pca_2"] = coords[:, 1]
    return out.reset_index(drop=True)


def save_artifacts(
    preprocessor: SegmentationPreprocessor,
    km: KMeans,
    segment_names: Dict[int, str],
    metrics: Dict[str, float],
    cfg: dict,
) -> Dict[str, Path]:
    """Sauvegarde preprocessor, kmeans et metadonnees dans models/."""
    models_dir = Path(cfg["paths"].get("outputs_dir", "outputs")).parent / cfg.get("segmentation", {}).get("models_dir", "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["scaler"] = models_dir / "scaler.joblib"
    joblib.dump(preprocessor, paths["scaler"])
    paths["kmeans"] = models_dir / "kmeans.joblib"
    joblib.dump(km, paths["kmeans"])
    meta = {
        "k": int(km.n_clusters),
        "feature_names": preprocessor.feature_names_,
        "segment_names": {str(k): v for k, v in segment_names.items()},
        "metrics": metrics,
    }
    paths["meta"] = models_dir / "segment_meta.json"
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return paths


def load_artifacts(cfg: dict) -> Tuple[SegmentationPreprocessor, KMeans, dict]:
    """Charge les artefacts sauvegardes (preprocessor, kmeans, meta)."""
    models_dir = Path(cfg["paths"].get("outputs_dir", "outputs")).parent / cfg.get("segmentation", {}).get("models_dir", "models")
    preprocessor = joblib.load(models_dir / "scaler.joblib")
    km = joblib.load(models_dir / "kmeans.joblib")
    with open(models_dir / "segment_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return preprocessor, km, meta
