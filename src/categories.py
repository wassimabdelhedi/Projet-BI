"""Categories PRODUIT derivees de la Description (le dataset n'a pas de categorie native).

Methode : regles par mots-cles, appliquees dans l'ORDRE (la premiere regle qui correspond gagne).
Cette categorisation est APPROXIMATIVE : elle doit etre validee sur un echantillon etiquete a la main
(voir export_validation_sample) et signalee comme limite dans le rapport.
"""
import re

import pandas as pd

# Ordre important : "CHRISTMAS CARD" -> CHRISTMAS (pas STATIONERY), "LUNCH BAG" -> BAGS (pas KITCHEN).
CATEGORY_KEYWORDS = {
    "CHRISTMAS": ["CHRISTMAS", "XMAS", "SANTA", "ADVENT", "REINDEER", "SNOWMAN", "TINSEL", "BAUBLE", "SNOWFLAKE"],
    "BAGS": ["BAG", "SHOPPER", "TOTE", "PURSE", "WALLET", "SATCHEL", "RUCKSACK"],
    "CANDLES_LIGHTING": ["CANDLE", "LIGHT", "LAMP", "LANTERN", "TEALIGHT", "T-LIGHT", "LAMPSHADE", "FAIRY LIGHTS"],
    "JEWELLERY_ACCESSORIES": ["NECKLACE", "BRACELET", "EARRING", "RING", "BROOCH", "JEWELLERY", "HAIR", "SCARF", "BEAD", "PENDANT"],
    "STATIONERY_PAPER": ["CARD", "WRAP", "PAPER", "NOTEBOOK", "PENCIL", "PEN", "STICKER", "RIBBON", "TAG", "ENVELOPE", "STATIONERY", "ERASER", "RUBBER"],
    "TOYS_GAMES": ["TOY", "GAME", "PUZZLE", "DOLL", "PLAY", "CHILDREN", "KIDS", "SKITTLES", "TEDDY", "DOMINOES", "BALLOON"],
    "GARDEN_OUTDOOR": ["GARDEN", "PLANT", "BIRD", "SEED", "WATERING", "TROWEL", "BUNTING", "PICNIC", "PARASOL"],
    "KITCHEN_DINING": ["MUG", "CUP", "TEAPOT", "PLATE", "BOWL", "JAR", "TIN", "CAKE", "SPOON", "KITCHEN", "LUNCH", "BOTTLE",
                       "TEA", "COFFEE", "BAKING", "APRON", "TRAY", "GLASS", "NAPKIN", "COASTER", "JUG", "FORK", "KNIFE"],
    "STORAGE": ["BOX", "BASKET", "STORAGE", "DRAWER", "CABINET", "ORGANISER", "TIDY", "HOLDER", "RACK", "SHELF", "CRATE"],
    "HOME_DECOR": ["FRAME", "CUSHION", "CLOCK", "MIRROR", "DOORMAT", "HEART", "SIGN", "DECORATION", "WALL", "HOOK",
                   "HANGING", "DOOR", "CURTAIN", "VASE", "BLANKET", "THROW", "PILLOW", "ORNAMENT", "STAR"],
}
OTHER = "OTHER"


def _compile():
    return {cat: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")S?\b") for cat, kws in CATEGORY_KEYWORDS.items()}


_PATTERNS = _compile()


def normalize_description(s: pd.Series) -> pd.Series:
    """Majuscules, espaces multiples reduits, bords nettoyes."""
    return s.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def category_of(description: str) -> str:
    if not isinstance(description, str):
        return OTHER
    for cat, pat in _PATTERNS.items():
        if pat.search(description):
            return cat
    return OTHER


def assign_category(descriptions: pd.Series) -> pd.Series:
    """Applique les regles sur les descriptions UNIQUES puis rebroadcast (rapide sur 1 M de lignes)."""
    uniques = descriptions.dropna().unique()
    mapping = {d: category_of(d) for d in uniques}
    return descriptions.map(mapping).fillna(OTHER).astype(str)


def export_validation_sample(df: pd.DataFrame, n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Echantillon de produits uniques a etiqueter MANUELLEMENT (colonne Category_manual) pour mesurer la precision."""
    prod = df[["StockCode", "Description", "Category"]].drop_duplicates("StockCode")
    sample = prod.sample(min(n, len(prod)), random_state=seed).rename(columns={"Category": "Category_auto"})
    sample["Category_manual"] = ""
    return sample.reset_index(drop=True)
