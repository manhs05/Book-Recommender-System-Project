"""
Goodreads Discover — Streamlit App
Powered by Hybrid Recommender System v3 (Dual-Embedding NLP)
"""

import os
import re
import warnings
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Goodreads · Discover",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Goodreads aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

.gr-ai-box {
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 15px;
    margin-top: 15px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.6;
    color: #2c3e50;
    font-style: italic; /* Italic for explanation */
}

.gr-stars {
    display: inline !important; /* Show stars */
    color: #e07c41;
    font-size: 15px;
    letter-spacing: 1px;
    margin-right: 4px;
}
.gr-ai-header {
    font-weight: bold;
    color: #007bff;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 5px;
}
                       
/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f4f1ea !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}

/* ── Navbar ── */
.gr-navbar {
    background: #2D4A3E;
    padding: 12px 32px;
    display: flex;
    border-radius: 12px 12px 0px 0px;        
    align-items: center;
    gap: 14px;
    margin-bottom: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

.gr-logo {
    font-family: 'Lora', serif;
    font-size: 22px;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: -0.3px;
}

.gr-logo span { color: #e07c41; }

.gr-badge {
    background: #e07c41;
    color: #fff;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ── Hero section ── */
.gr-hero {
    background: #ffffff;
    padding: 40px 48px 32px;
    border-bottom: 1px solid #d8cfc4;
}

.gr-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #767676;
    margin-bottom: 12px;
}

.gr-headline {
    font-family: 'Lora', serif;
    font-size: 38px;
    color: #1a1a1a;
    margin-bottom: 28px;
    line-height: 1.2;
    font-weight: 400;
}

.gr-headline em {
    color: #e07c41;
    font-style: italic;
}

.stTextInput div div input {
    border: 1.0px solid #c8bfb0;
    background-color: #ffffff !important;
    }

.stForm {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

.stButton button {
     background-color: #2D4A3E !important;
     color: white !important;
     font-weight: 700 !important;
     transition: background-color 0.3s !important;
    }

.stButton button:hover {
    background-color: #251e19 !important;
    color: #d1a45d !important; 
    }

button {
    background-color: #2D4A3E !important;
    color: white !important;
}



/* ── Prompt chips ── */
.gr-chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.gr-chip {
    background: #fff;
    border: 1.5px solid #c8bfb0;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    color: #4a4a4a;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.18s;
    font-family: 'Inter', sans-serif;
}

.gr-chip:hover {
    background: #382110;
    color: #fff;
    border-color: #382110;
}

/* ── NLP Badges ── */
.gr-nlp-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: #f0ece4;
    border-radius: 8px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}

.gr-nlp-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #767676;
}

.gr-badge-genre {
    background: #382110;
    color: #fff;
    font-size: 12px;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 12px;
}

.gr-badge-trope {
    background: #e07c41;
    color: #fff;
    font-size: 12px;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 12px;
}

/* ── Results header ── */
.gr-results-header {
    font-size: 13px;
    color: #767676;
    margin-bottom: 20px;
    padding: 0 2px;
}

.gr-results-header strong {
    color: #1a1a1a;
    font-weight: 600;
}

/* ── Book Cards ── */
.gr-card {
    background: #ffffff;
    border: 1px solid #ddd5c8;
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
    transition: box-shadow 0.2s, border-color 0.2s;
    position: relative;
}

.gr-card:hover {
    box-shadow: 0 4px 20px rgba(56,33,16,0.10);
    border-color: #c8a882;
}

.gr-card-left { flex: 1; min-width: 0; }

.gr-card-title {
    font-family: 'Lora', serif;
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
    text-decoration: none;
    line-height: 1.3;
    display: block;
    margin-bottom: 4px;
}

.gr-card-title:hover { color: #e07c41; text-decoration: underline; }

.gr-card-author_clean {
    font-size: 14px;
    color: #767676;
    margin-bottom: 10px;
    font-weight: 400;
    font-style: italic; /* Make author names italic */
}

.gr-stars {
    color: #e07c41;
    font-size: 15px;
    letter-spacing: 1px;
}

.gr-rating-text {
    font-size: 13px;
    color: #767676;
    margin-left: 8px;
}

.gr-card-meta {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 12px;
}

.gr-genre-pill {
    font-size: 11px;
    background: #f0ece4;
    color: #5a4a3a;
    padding: 2px 10px;
    border-radius: 10px;
    font-weight: 500;
}

.gr-explanation {
    font-size: 14px;
    color: #2d5a3d;
    background-color: #e8f5e9;
    line-height: 1.7;
    margin-top: 12px;
    padding: 12px;
    border-left: 4px solid #4caf50;
    border-radius: 4px;
    font-style: normal;
}

.gr-card-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
    flex-shrink: 0;
}

.gr-match-badge {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    background: #382110;
    padding: 5px 14px;
    border-radius: 20px;
    white-space: nowrap;
    letter-spacing: 0.3px;
}

.gr-match-badge.high { background: #2d6a2d; }
.gr-match-badge.med  { background: #7a5c2e; }
.gr-match-badge.low  { background: #767676; }

.gr-goodreads-link {
    font-size: 12px;
    color: #e07c41;
    text-decoration: none;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
}

.gr-goodreads-link:hover { text-decoration: underline; }

/* ── Loading spinner override ── */
.stSpinner > div {
    border-top-color: #e07c41 !important;
}

/* ── Divider ── */
.gr-divider {
    border: none;
    border-top: 1px solid #ddd5c8;
    margin: 24px 0;
}

/* ── Streamlit element hiding ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }

  
[data-testid="column"] .stTextInput,
[data-testid="column"] .stButton {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (from notebook)
# ─────────────────────────────────────────────────────────────────────────────
TROPE_KEYWORDS = {
    "survival":           ["survival", "survive", "wilderness", "stranded", "rescue", "endurance"],
    "romance":            ["love", "romance", "relationship", "heartbreak", "passion"],
    "redemption":         ["redemption", "forgiveness", "second chance", "overcome",
                           "overcoming", "obstacles", "finding yourself", "self discovery", "growth"],
    "mystery":            ["mystery", "detective", "clue", "investigation", "thriller"],
    "coming_of_age":      ["coming of age", "growing up", "youth", "self-discovery",
                           "finding yourself", "who you are", "identity"],
    "conspiracy":         ["conspiracy", "cover up", "secret", "hidden truth"],
    "adventure":          ["adventure", "quest", "journey", "expedition"],
    "war":                ["war", "battle", "soldier", "combat", "military"],
    "friendship":         ["friendship", "loyalty", "bond", "companion"],
    "historical_fiction": ["historical fiction", "historical", "history",
                           "period drama", "set in the past", "historical novel"],
    "heist":              ["heist", "robbery", "steal", "caper"],
    "dystopia":           ["dystopia", "dystopian", "post-apocalyptic", "totalitarian"],
    "magic":              ["magic", "wizard", "spell", "sorcery", "enchant"],
    "revenge":            ["revenge", "vengeance", "payback", "retaliation"],
    "time_travel":        ["time travel", "time machine", "paradox"],
    "unrequited_love":    ["unrequited", "not returned", "one sided", "one-sided", "unrequited love"],
}

TONE_KEYWORDS = {
    "emotional":   ["emotional", "moving", "touching", "heartfelt", "deep", "poignant"],
    "funny":       ["funny", "humor", "comedy", "witty", "lighthearted", "light-hearted"],
    "dark":        ["dark", "grim", "bleak", "disturbing", "haunting"],
    "inspiring":   ["inspiring", "uplifting", "motivational", "hopeful", "inspired"],
    "educational": ["informative", "educational", "insightful", "detailed"],
    "tense":       ["tense", "suspense", "gripping", "thrilling", "intense"],
    "cozy":        ["cozy", "warm", "comfort", "wholesome", "gentle"],
    "romantic":    ["romantic", "swoon", "butterflies", "sweet"],
}

GENRE_KEYWORDS = {
    "fantasy":            ["magic", "dragon", "wizard", "spell", "quest", "realm", "enchant", "prophecy"],
    "science fiction":    ["space", "robot", "alien", "future", "dystopia", "technology", "cyberpunk"],
    "mystery":            ["detective", "murder", "crime", "clue", "suspect", "investigation", "thriller"],
    "romance":            ["love", "romance", "relationship", "passion", "heart", "kiss", "attraction"],
    "horror":             ["horror", "scary", "ghost", "haunted", "terror", "supernatural", "demon"],
    "historical fiction": ["historical", "history", "war", "century", "ancient", "kingdom", "empire"],
    "self-help":          ["self-help", "growth", "improvement", "motivation", "habit", "mindset"],
    "literary fiction":   ["literary", "prose", "character study", "coming of age", "family"],
}

GENRE_ALIAS = {
    "self help":"self-help","selfhelp":"self-help","sci fi":"science fiction",
    "scifi":"science fiction","sf":"science fiction","ya":"young adult",
    "non fiction":"nonfiction","non-fiction":"nonfiction",
    "hist fiction":"historical fiction","historical":"historical fiction",
    "literary fiction":"literary","lgbtq":"lgbtq","lgbt":"lgbtq",
    "childrens":"children's","picture books":"children's","paranormal":"paranormal romance"
}

GENRE_BLACKLIST = {"favorites","owned","kindle","to-read","default",
    "library","read","books","my books","to read","currently reading",
    "wish list","series","recommended",
}

KEYWORD_BLACKLIST = {
    # Meta words
    "book", "books", "novel", "story", "read", "reading", "want",
    "need", "like", "give", "recommend", "find", "tell", "know",
    "please", "something", "genre", "type", "kind",
    # Pronouns & articles
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "she", "they", "them",
    "a", "an", "the", "this", "that", "these", "those",
    # Common verbs & conjunctions
    "is", "are", "be", "have", "has", "do", "does", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "with", "from", "by", "of", "about", "as", "can", "could", "would", "should",
    "where", "what", "which", "who", "when", "why", "how",
    # Generic filler
    "one", "two", "three", "all", "each", "get", "take", "make", "go", "come", "see", "way",
    "thing", "stuff", "etc", "so", "just", "really", "very", "quite", "more", "some", "any",
}

NEGATION_CUES = {"not", "no", "without", "never", "don't", "doesnt",
                 "isn't", "aren't", "wasn't", "weren't", "hardly", "barely"}

SYNONYM_MAP = {
    "sad": "emotional", "cry": "emotional", "crying": "emotional",
    "lgbt": "lgbtq", "gay": "lgbtq", "lesbian": "lgbtq", "queer": "lgbtq",
    "scary": "horror", "creepy": "horror",
    "space": "science fiction", "futuristic": "science fiction",
    "medieval": "fantasy", "magic": "fantasy",
    "funny": "humor", "comedy": "humor", "hilarious": "humor",
    "ww2": "world war 2", "wwii": "world war 2",
}
CLUSTER_LABEL_MAP = {
     0:"anthology and short stories",2:"design and architecture",3:"superhero comics",
     4:"family drama",6:"historical romance",7:"epic fantasy",8:"fairy tale retelling",
     9:"poetry collection",12:"sports story",13:"asian history and culture",
    14:"cookbook and food",15:"relationships and sexuality",16:"self-help guide",
    18:"manga",19:"christian and religious fiction",20:"personal development",
    21:"world war military history",22:"american political history",24:"nature and wildlife",
    25:"graphic novel and comics",26:"crafts and hobbies",27:"science and nature for kids",
    28:"civil war history",29:"jewish history and culture",30:"holiday and seasonal",
    31:"biblical commentary",32:"children's picture book",34:"quilting and sewing",
    35:"film and cinema",36:"animals and nature",37:"russian and soviet history",
    38:"pets and animal care",39:"mathematics and textbook",40:"murder mystery detective",
    42:"contemporary romance",43:"art and visual culture",44:"literary fiction",
    45:"health diet and nutrition",46:"business and management",47:"pirates and nautical adventure",
    48:"language and linguistics",49:"royalty and medieval kingdom",50:"amish and community fiction",
    51:"ancient egypt and classics",52:"tudor and british history",53:"manga romance",
    54:"theatre and plays",55:"how-to guide and reference",56:"prayer and devotional",
    57:"japanese history and culture",58:"roman and ancient history",
    59:"physics and philosophy of science",
}
TROPE_LABELS = [
    "enemies to lovers","forbidden love","second chance romance","love triangle",
    "slow burn romance","fake dating","marriage of convenience","unrequited love",
    "soulmates","friends to lovers","redemption arc","coming of age","chosen one",
    "antihero","villain protagonist","reluctant hero","found family","mentor and student",
    "underdog story","identity crisis","revenge plot","heist","mystery","conspiracy",
    "survival","quest","adventure","time travel","prophecy","tournament","road trip",
    "fish out of water","mistaken identity","secret identity","betrayal",
    "unreliable narrator","twist ending","whodunit","dystopia","post-apocalyptic",
    "magic system","political intrigue","war","historical setting","small town",
    "school or academy","royalty and nobility","space exploration","parallel worlds",
    "cozy setting","grief and loss","trauma and healing","mental health",
    "loneliness and isolation","hope and resilience","dark and gritty",
    "lighthearted and cozy","bittersweet","philosophical themes","class struggle",
    "coming out story","immigrant experience","female friendship and sisterhood",
    "multiple POV","epistolary format","nonlinear timeline","dual timeline",
    "vampire","werewolf","fae and fairy tale","demons and angels","detective noir",
    "cozy mystery","legal thriller","medical drama","sports story","superhero","friendship",
]
PROMPT_SUGGESTIONS = [
    "dark academia with enemies-to-lovers",
    "cozy mystery with found family…",
    "epic fantasy with morally grey hero",
    "literary fiction about grief and loss",
    "sci-fi thriller with unreliable narrator",
    "historical romance set in Victorian era",
    "coming-of-age with self-discovery",
    "survival thriller gripping suspense",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_query(query: str) -> str:
    q = query.lower().strip()
    for src, tgt in SYNONYM_MAP.items():
        q = re.sub(r"\b" + src + r"\b", tgt, q)
    return q


def normalize_genre_tag(tag: str) -> str:
    tag = str(tag).lower().strip()
    tag = re.sub(r"[^a-z0-9 ]", " ", tag)
    tag = re.sub(r"\s+", " ", tag).strip()
    # Apply GENRE_ALIAS mapping first
    tag = GENRE_ALIAS.get(tag, tag)
    # Then convert any remaining spaces to hyphens (e.g., "personal development" → "personal-development")
    tag = tag.replace(' ', '-')
    return tag


def parse_genres(genre_str) -> list:
    if pd.isna(genre_str) or str(genre_str).strip() in ("", "nan"):
        return []
    tags = [normalize_genre_tag(t) for t in str(genre_str).split("|")]
    seen, result = set(), []
    for t in tags:
        if t and t not in seen and t not in GENRE_BLACKLIST:
            seen.add(t); result.append(t)
    return result


def classify_genre_fallback(text: str) -> str:
    text = str(text).lower()
    scores = {g: sum(1 for kw in kws if kw in text) for g, kws in GENRE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def stars_html(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def match_class(pct: int) -> str:
    if pct >= 75:
        return "high"
    elif pct >= 55:
        return "med"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# NLP FUNCTIONS (ported from notebook)
# ─────────────────────────────────────────────────────────────────────────────
def _get_negated_indices_simple(query: str) -> set:
    """Enhanced negation detection with multi-word patterns."""
    negated = set()
    words = query.lower().split()
    
    # Multi-word negation patterns
    negation_patterns = [
        ["not", "returned"],
        ["not", "be"],
        ["not", "the"],
        ["not", "love"],
        ["not", "that"],
        ["are", "not"],
    ]
    
    # Check for multi-word negation patterns
    for pattern in negation_patterns:
        for i in range(len(words) - len(pattern) + 1):
            if words[i:i+len(pattern)] == pattern:
                for j in range(i, min(i + len(pattern) + 3, len(words))):
                    negated.add(j)
    
    # Standard negation detection
    for i, w in enumerate(words):
        if w in NEGATION_CUES:
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j] not in NEGATION_CUES:
                    negated.add(j)
                    break
    
    return negated


def extract_intent_final(query: str, model=None, trope_embeddings=None) -> dict:
    """
    Negation-aware intent extractor (v4.0++ - production lite).
    **IMPROVED (v4.0++):** Better pattern detection + auto-semantic filtering
    - Fixed: Extract negated word itself (not just after cue)
    - Fixed: Look ahead up to 5 words for negated patterns
    - Fixed: Exclude negated words from positive keywords
    """
    from sentence_transformers import util as st_util
    import torch
    
    query_lower = normalize_query(query)
    words = query_lower.split()
    negated_idx = _get_negated_indices_simple(query_lower)

    # ── IMPROVED: Better negated word extraction (look ahead 5 words) ────────
    negated_words = []
    for i, w in enumerate(words):
        if w in NEGATION_CUES:
            # Look up to 5 words ahead (not just 3-4)
            for j in range(i + 1, min(i + 6, len(words))):
                w_next = words[j]
                # Stop at conjunctions
                if w_next in ("but", "and", "or", "although"):
                    break
                # Capture content words
                if len(w_next) > 2 and w_next not in NEGATION_CUES and w_next not in KEYWORD_BLACKLIST:
                    negated_words.append(w_next)
                    break
    
    # Auto-detect negated themes using semantic similarity
    negated_theme_keywords = set()
    if model is not None and trope_embeddings is not None:
        for neg_word in set(negated_words):
            try:
                neg_emb = model.encode(f"not {neg_word}", convert_to_tensor=True)
                trope_scores_neg = st_util.cos_sim(neg_emb, trope_embeddings)[0]
                top_indices = torch.topk(trope_scores_neg, min(5, len(TROPE_LABELS))).indices
                nearby_tropes = [TROPE_LABELS[i] for i in top_indices.cpu().numpy()]
                
                # Keep only tropes that contain the negated word
                related = [t for t in nearby_tropes if neg_word in t.lower()]
                negated_theme_keywords.update(related)
            except:
                pass

    # ── FIXED: Keyword extraction - EXCLUDE negated words ──────────────────
    positive_kws, negative_kws = [], []
    processed = set()
    
    for i, w in enumerate(words):
        if i in processed:
            continue
        if w in NEGATION_CUES or w in KEYWORD_BLACKLIST:
            continue
        if len(w) <= 2:
            continue
        
        # Multi-word concepts
        multi_word = None
        if i + 1 < len(words):
            next_word = words[i + 1]
            if (next_word not in KEYWORD_BLACKLIST and 
                next_word not in NEGATION_CUES and 
                len(next_word) > 2):
                multi_word = f"{w} {next_word}"
                processed.add(i + 1)
        
        is_negated = i in negated_idx
        keyword = multi_word if multi_word else w
        
        # ← FIXED: Don't include negated words in positive keywords
        if is_negated:
            negative_kws.append(f"not_{keyword.replace(' ', '_')}")
        else:
            positive_kws.append(keyword)

    # ── Trope matching (with negation filtering) ───────────────────────────
    tropes = []
    for trope, keywords in TROPE_KEYWORDS.items():
        triggered = [kw for kw in keywords if kw in query_lower]
        if triggered:
            all_neg = all(
                any(w in kw for w_i, w in enumerate(words) if w_i in negated_idx)
                for kw in triggered
            )
            if not all_neg:
                tropes.append(trope)
    
    # Filter tropes using auto-detected negated themes (NO hardcoded mapping!)
    tropes = [
        t for t in tropes 
        if not any(kw in t.lower() for kw in negated_theme_keywords)
    ]

    # Tone matching
    tones = []
    for tone, keywords in TONE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            tones.append(tone)

    # Genre detection
    genres = []
    for genre, keywords in GENRE_KEYWORDS.items():
        if sum(1 for kw in keywords if kw in query_lower) >= 1:
            genres.append(genre)

    return {
        "original_query": query,
        "keywords": positive_kws,
        "negative_keywords": negative_kws,
        "tropes": tropes,
        "tones": tones,
        "genres": genres,
    }


def remove_negated_base_tropes(query: str, tropes: list) -> list:
    """
    Post-processor for negation handling.
    Extracts negated words from query and removes any tropes that match.
    Uses bidirectional substring matching and smart word extraction.
    
    Example: query="not romance" → removes any trope containing "romance"
    """
    query_lower = query.lower()
    
    # Words to skip when extracting negated content
    skip_words = {"a", "an", "the", "any", "some", "all", "this", "that", "these", "those"}
    
    # Extract negated words via regex patterns with lookahead to skip articles
    negated_words = set()
    
    # Extended patterns that capture the negated concept, not just articles
    patterns = [
        r'not\s+(?:a\s+|an\s+|the\s+)?(\w+)',
        r'no\s+(?:a\s+|an\s+|the\s+)?(\w+)',
        r'without\s+(?:a\s+|an\s+|the\s+)?(?:any\s+)?(\w+)',
        r'definitely\s+not\s+(?:a\s+|an\s+|the\s+)?(\w+)',
        r'never\s+(?:a\s+|an\s+|the\s+)?(\w+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query_lower)
        negated_words.update(m for m in matches if m not in skip_words)
    
    # If no negated words found, return original tropes
    if not negated_words:
        return tropes
    
    # Filter tropes using bidirectional substring matching
    # Also check if underscores versions match (e.g., "science_fiction")
    filtered_tropes = []
    for trope in tropes:
        trope_lower = trope.lower()
        trope_underscored = trope_lower.replace(" ", "_")  # Handle space-separated tropes
        
        # Keep trope only if NO negated word matches it (bidirectional)
        is_negated = any(
            (neg_word in trope_lower or trope_lower in neg_word or
             neg_word in trope_underscored or trope_underscored in neg_word)
            for neg_word in negated_words
        )
        
        if not is_negated:
            filtered_tropes.append(trope)
    
    return filtered_tropes


def build_query_text(intent: dict) -> str:
    parts = []
    kws = intent.get("keywords", [])
    if kws:
        parts.append(" ".join(kws))
    tropes = intent.get("tropes", [])
    if tropes:
        parts.extend([t.replace("_", " ") for t in tropes])
    tones = intent.get("tones", [])
    if tones:
        parts.extend(tones)
    return " ".join(parts) if parts else intent.get("original_query", "")


# ─────────────────────────────────────────────────────────────────────────────
# TITLE CASE FORMATTER (Proper capitalization with exceptions)
# ─────────────────────────────────────────────────────────────────────────────
def apply_title_case(text: str) -> str:
    """Convert text to title case with exceptions for small words."""
    if not text or not isinstance(text, str):
        return "Unknown"
    
    exceptions = {'a', 'an', 'the', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'by', 'for', 'with'}
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in exceptions:
            result.append(word.capitalize())
        else:
            result.append(word.lower())
    
    return ' '.join(result)


def capitalize_author_name(name: str) -> str:
    """Capitalize first letter of author name, handling multiple authors."""
    if not name or not isinstance(name, str):
        return "Unknown"
    
    # Handle multiple authors separated by comma
    authors = [a.strip() for a in name.split(',')]
    capitalized = []
    
    for author in authors:
        # Split by spaces and capitalize each word
        parts = author.split()
        capitalized_parts = [part.capitalize() for part in parts]
        capitalized.append(' '.join(capitalized_parts))
    
    return ', '.join(capitalized)


def extract_top_genres(genres_str: str, top_n: int = 5) -> list:
    """Extract top N unique genres from a pipe-separated genres string (ONLY split by pipe)."""
    if not genres_str or pd.isna(genres_str):
        return []
    
    # Convert to string and clean pandas artifacts
    genres_str = str(genres_str).strip()
    
    # Remove 'genres_list' text and pandas dtype information
    genres_str = genres_str.replace('genres_list', '').strip()
    if 'Name:' in genres_str:
        genres_str = genres_str.split('Name:')[0].strip()
    genres_str = genres_str.replace('Dtype: object', '').replace('dtype:', '').strip()
    
    # ONLY split by pipe delimiter - never by space (preserves hyphenated genres like "self-help")
    if '|' in genres_str:
        genres = [normalize_genre_tag(g.strip()) for g in genres_str.split('|')]
    else:
        # Single genre - return as-is after normalization
        genres = [normalize_genre_tag(genres_str)] if genres_str and genres_str != 'nan' else []
    
    # Clean and filter - remove empty strings and '...', keep unique
    seen = set()
    unique_genres = []
    for g in genres:
        g_lower = g.lower()
        if (g and g_lower != 'nan' and '...' not in g and 
            len(g) > 0 and g_lower not in seen):
            seen.add(g_lower)
            unique_genres.append(g)
    
    return unique_genres[:top_n]  # Return top N unique genres

# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION GENERATION (FREE - No API required)
# ─────────────────────────────────────────────────────────────────────────────
def generate_explanation_pattern(book_title: str, author: str, genres_list: list,
                                  user_query: str, tropes: list, tones: list, seed_idx: int = 0) -> str:
    """
    5 carefully crafted explanation templates - distributed across recommendations.
    Uses seed_idx to ensure variety without repetition.
    """
    import random
    
    # Clean inputs
    def clean_list(items):
        cleaned = []
        for t in items:
            if not t:
                continue
            s = str(t).strip().lower()
            if any(bad in s for bad in ['dtype', 'name:', 'object', 'float', 'int64', 'nan']):
                continue
            if s and len(s) > 2:
                cleaned.append(str(t).strip())
        return cleaned
    
    tropes = clean_list(tropes)
    tones = clean_list(tones)
    genres_clean = clean_list(genres_list)
    
    # Build references
    if genres_clean:
        genre_ref = ", ".join([g for g in genres_clean[:2]]) if len(genres_clean) >= 2 else genres_clean[0]
    else:
        genre_ref = "fiction"
    
    if not tropes and not tones:
        base_templates = [
            f"A captivating {genre_ref} that perfectly matches your search.",
            f"Just what you're looking for in {genre_ref}.",
            f"Highly recommended {genre_ref} for your reading list.",
            f"A standout choice in {genre_ref}.",
            f"The perfect {genre_ref} to discover.",
        ]
        return base_templates[seed_idx % len(base_templates)]
    
    trope_str = tropes[0] if tropes else None
    tone_str = tones[0] if tones else None
    
    # 5 Master Templates - rotating through with seed
    if trope_str and tone_str:
        templates = [
            f"Exactly this: {genre_ref} featuring {trope_str} with a {tone_str} narrative.",
            f"Your perfect match—{genre_ref} that masterfully delivers {trope_str} in a {tone_str} way.",
            f"The rare gem: {genre_ref} that combines {trope_str} with {tone_str} storytelling.",
            f"{genre_ref} at its finest. Exceptional {trope_str} wrapped in a {tone_str} tale.",
            f"What you need: {genre_ref} where {trope_str} shines through a {tone_str} lens.",
        ]
    elif trope_str:
        templates = [
            f"Brilliant {genre_ref} centered on {trope_str}.",
            f"Outstanding {genre_ref} featuring {trope_str}—exactly what you wanted.",
            f"{genre_ref} that excels at {trope_str}. Highly compelling.",
            f"The {trope_str} in this {genre_ref} is absolutely masterful.",
            f"Perfect find: {genre_ref} with superb {trope_str} execution.",
        ]
    elif tone_str:
        templates = [
            f"A distinctly {tone_str} {genre_ref} you'll love.",
            f"{genre_ref} delivered with {tone_str} brilliance.",
            f"Exceptional {genre_ref}—{tone_str} and utterly captivating.",
            f"The {tone_str} tone of this {genre_ref} is perfectly crafted.",
            f"Outstanding {genre_ref}: {tone_str}, gripping, unforgettable.",
        ]
    else:
        templates = [
            f"Excellent {genre_ref} recommendation.",
            f"Outstanding {genre_ref} choice.",
            f"Highly recommended {genre_ref}.",
            f"Superb {genre_ref} selection.",
            f"Perfect {genre_ref} match.",
        ]
    
    return templates[seed_idx % len(templates)]


def get_explanation(book_title: str, author: str, genres_list: list,
                    user_query: str, tropes: list, tones: list, seed_idx: int = 0) -> str:
    """
    Generate personalized explanation using 5 rotating templates.
    seed_idx ensures variety across recommendations.
    """
    return generate_explanation_pattern(book_title, author, genres_list, user_query, tropes, tones, seed_idx)


def display_load_more_button(total_recs: int) -> None:
    """
    Display 'Load More Books' button with pagination counter.
    Shows button only if there are more books to load.
    Optimized for smooth loading without full page rerun.
    """
    if st.session_state.results_displayed < total_recs:
        # Spacing before button
        st.markdown("<div style='padding: 50px 0 20px 0;'></div>", unsafe_allow_html=True)
        st.markdown("<div id='load_more_anchor'></div>", unsafe_allow_html=True)
        
        # Button in centered columns
        col1, col2, col3 = st.columns([1, 2.5, 1])
        with col2:
            button_clicked = st.button(
                "Load More Books",
                use_container_width=True,
                key=f"load_more_{st.session_state.results_displayed}",
                type="secondary"
            )
            if button_clicked:
                new_count = min(
                    st.session_state.results_displayed + 20,
                    total_recs
                )
                st.session_state.results_displayed = new_count
                # Scroll to button location
                st.markdown("""
                <script>
                    setTimeout(() => {
                        document.getElementById('load_more_anchor').scrollIntoView({behavior: 'smooth', block: 'center'});
                    }, 100);
                </script>
                """, unsafe_allow_html=True)
                st.rerun()
        
        # Counter text below button
        st.markdown(
            f"<div style='text-align: center; color: #a09080; font-size: 13px; margin-top: 16px;'>"
            f"<strong>Showing {min(st.session_state.results_displayed, total_recs)} of {total_recs} books</strong>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Bottom spacing
        st.markdown("<div style='padding: 20px 0;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_data(books_path: str, reviews_path: str, interactions_path: str):
    books        = pd.read_csv(books_path,        low_memory=False)
    reviews      = pd.read_csv(reviews_path,      low_memory=False)
    interactions = pd.read_csv(interactions_path, low_memory=False)

    for df_, col in [(books, "book_id"), (reviews, "book_id"),
                     (interactions, "book_id")]:
        df_[col] = df_[col].astype(str)

    if "desc_clean" in books.columns:
        books["desc_clean"] = books["desc_clean"].apply(clean_text)

    return books, reviews, interactions


@st.cache_data(show_spinner=False)
def build_book_profiles(_books, _reviews):
    # Filter reviews
    mask = _reviews["rating"] > 0
    if "is_spam_flag"   in _reviews.columns: mask &= (_reviews["is_spam_flag"]   == False)
    if "is_boilerplate" in _reviews.columns: mask &= (_reviews["is_boilerplate"] == False)
    df_clean = _reviews[mask].copy()

    books_meta = _books[["book_id", "genres_list", "desc_clean", "author_clean"]].drop_duplicates("book_id")
    df_clean   = df_clean.merge(books_meta, on="book_id", how="left")

    rev_col = "review_normalized" if "review_normalized" in df_clean.columns else "review_text"
    agg = df_clean.groupby("book_id").agg(
        review_text  = (rev_col, lambda x: " ".join(x.astype(str))),
        genres_list  = ("genres_list",  "first"),
        desc_clean   = ("desc_clean",   "first"),
        author_clean = ("author_clean", "first"),
        title        = ("title",        "first"),
    ).reset_index()

    missing = _books[["book_id", "title", "genres_list", "desc_clean", "author_clean"]][
        ~_books["book_id"].isin(agg["book_id"])
    ].copy()
    missing["review_text"] = ""

    book_profiles = pd.concat([agg, missing], ignore_index=True)

    # Genre parsing - convert list back to pipe-separated string for later extraction
    book_profiles["genres_list"]   = book_profiles["genres_list"].apply(parse_genres)
    book_profiles["genres_list"] = book_profiles["genres_list"].apply(
        lambda lst: "|".join(lst) if lst else None
    )
    combined_fallback = book_profiles["review_text"].fillna("") + " " + book_profiles["desc_clean"].fillna("")
    missing_mask = book_profiles["genres_list"].isna()
    book_profiles.loc[missing_mask, "genres_list"] = combined_fallback[missing_mask].apply(classify_genre_fallback)

    return book_profiles, df_clean


@st.cache_data(show_spinner=False)
def build_collab_scores(_books, _reviews_clean, _interactions):
    # Sentiment aggregation
    sentiment_agg = _reviews_clean.groupby("book_id").agg(
        avg_polarity     = ("sentiment_polarity",     "mean") if "sentiment_polarity"     in _reviews_clean.columns else ("rating", "mean"),
        avg_subjectivity = ("sentiment_subjectivity", "mean") if "sentiment_subjectivity" in _reviews_clean.columns else ("rating", "mean"),
    ).reset_index().round(4)
    sentiment_agg.columns = ["book_id", "avg_polarity", "avg_subjectivity"]

    # Rating source
    books_rating = _books[["book_id", "rating", "reviews"]].copy()
    books_rating.columns = ["book_id", "rating", "totalratings"]
    books_rating["rating"]   = pd.to_numeric(books_rating["rating"],   errors="coerce")
    books_rating["totalratings"] = pd.to_numeric(books_rating["totalratings"], errors="coerce")

    # Interactions
    int_agg = _interactions.groupby("book_id").agg(
        total_saves     = ("save",     "sum") if "save"     in _interactions.columns else ("book_id", "count"),
        total_clicks    = ("click",    "sum") if "click"    in _interactions.columns else ("book_id", "count"),
        total_purchases = ("purchase", "sum") if "purchase" in _interactions.columns else ("book_id", "count"),
        total_shares    = ("share",    "sum") if "share"    in _interactions.columns else ("book_id", "count"),
    ).reset_index()

    collab_raw = (
        books_rating
        .merge(int_agg,      on="book_id", how="left")
        .merge(sentiment_agg, on="book_id", how="left")
    )
    collab_raw.fillna({"total_saves": 0, "total_clicks": 0,
                        "total_purchases": 0, "total_shares": 0,
                        "avg_polarity": 0.0, "avg_subjectivity": 0.5}, inplace=True)

    # Bayesian score
    C = collab_raw["rating"].mean()
    m = collab_raw["totalratings"].quantile(0.25)
    collab_raw["bayesian_score"] = (
        (collab_raw["totalratings"] * collab_raw["rating"] + m * C) /
        (collab_raw["totalratings"] + m)
    )

    # Interaction score
    int_cols = [c for c in ["total_saves", "total_clicks", "total_purchases", "total_shares"] if c in collab_raw.columns]
    if int_cols:
        collab_raw["interaction_raw"] = collab_raw[int_cols].sum(axis=1)
        mx = collab_raw["interaction_raw"].max()
        collab_raw["interaction_score"] = collab_raw["interaction_raw"] / (mx + 1e-9)
    else:
        collab_raw["interaction_score"] = 0.0

    # Sentiment score
    collab_raw["sentiment_score"] = (
        0.5 * collab_raw["avg_polarity"].clip(-1, 1).add(1).div(2) +
        0.5 * (1 - collab_raw["avg_subjectivity"].clip(0, 1))
    )

    # Normalize sub-scores
    for col in ["bayesian_score", "interaction_score", "sentiment_score"]:
        mn, mx = collab_raw[col].min(), collab_raw[col].max()
        collab_raw[col] = (collab_raw[col] - mn) / (mx - mn + 1e-9)

    collab_raw["collab_score"] = (
        0.5 * collab_raw["bayesian_score"] +
        0.3 * collab_raw["interaction_score"] +
        0.2 * collab_raw["sentiment_score"]
    ).round(4)

    collab_raw = collab_raw.set_index("book_id")
    return collab_raw


@st.cache_data(show_spinner=False)
def build_embeddings(_book_profiles, _st_model):
    # Try to load pre-computed embeddings first
    desc_emb_path = "desc_emb.npy"
    review_emb_path = "review_emb.npy"
    
    if os.path.exists(desc_emb_path) and os.path.exists(review_emb_path):
        desc_emb = np.load(desc_emb_path)
        review_emb = np.load(review_emb_path)
        return desc_emb, review_emb
    
    # Fall back to encoding if .npy files don't exist
    descriptions = _book_profiles["desc_clean"].fillna("").astype(str).tolist()
    reviews_text = _book_profiles["review_text"].fillna("").astype(str).tolist()

    desc_emb   = sk_normalize(_st_model.encode(descriptions, batch_size=64, show_progress_bar=False, convert_to_numpy=True), norm="l2")
    review_emb = sk_normalize(_st_model.encode(reviews_text, batch_size=64, show_progress_bar=False, convert_to_numpy=True), norm="l2")
    return desc_emb, review_emb


@st.cache_data(show_spinner=False)
def build_trope_embeddings(_st_model):
    """Pre-encode TROPE_LABELS for v4.0-lite semantic filtering."""
    return _st_model.encode(TROPE_LABELS, convert_to_tensor=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (from Cells 28, 37, 39)
# ─────────────────────────────────────────────────────────────────────────────
def process_query_vec(query, model):
    q = re.sub(r"[^a-z0-9\s]", "", str(query).lower())
    q = re.sub(r"\s+", " ", q).strip()
    return sk_normalize(model.encode([q], convert_to_numpy=True), norm="l2")

def compute_similarity(query_vec, desc_emb, review_emb, dw=0.7, rw=0.3):
    return dw * np.dot(query_vec, desc_emb.T)[0] + rw * np.dot(query_vec, review_emb.T)[0]

def compute_content_score(query_text, intent, book_profiles, desc_emb, review_emb, model, boost=0.05):
    if not query_text.strip():
        return pd.Series(0.0, index=book_profiles["book_id"])
    qv     = process_query_vec(query_text, model)
    scores = compute_similarity(qv, desc_emb, review_emb)
    detected = set(intent.get("tropes",[]) + intent.get("tones",[]))
    if detected:
        for i, row in book_profiles.iterrows():
            g = str(row.get("genre_primary") or "").lower()
            t = str(row.get("mined_trope")   or "").lower()
            if any(d in g or d in t for d in detected):
                scores[i] = min(1.0, scores[i] + boost)
    mn,mx = scores.min(), scores.max()
    return pd.Series(((scores-mn)/(mx-mn+1e-9)).round(4), index=book_profiles["book_id"])

def compute_collab_score_fn(user_id, collab_scores, user_sim, user_item_matrix):
    base = collab_scores["collab_score"].copy()
    if not user_id or user_id not in user_sim.index:
        return base
    neighbours = user_sim[user_id].drop(user_id, errors="ignore").nlargest(10)
    if neighbours.empty or neighbours.sum() == 0:
        return base
    nb_ids = neighbours.index.intersection(user_item_matrix.index)
    if len(nb_ids) == 0:
        return base
    n_rat  = user_item_matrix.loc[nb_ids]
    w_vec  = neighbours.loc[nb_ids].values
    w_sum  = n_rat.multiply(w_vec, axis=0).sum(axis=0)
    w_cnt  = n_rat.notna().multiply(w_vec, axis=0).sum(axis=0)
    cf     = (w_sum / w_cnt.replace(0, np.nan)).fillna(0)
    mn,mx  = cf.min(), cf.max()
    cf_n   = (cf-mn)/(mx-mn+1e-9)
    blended = pd.Series(0.0, index=base.index)
    blended.update(0.6*cf_n + 0.4*base.reindex(cf_n.index, fill_value=0))
    blended[blended == 0] = base[blended == 0]
    return blended.round(4)


def recommend(query, top_n, book_profiles, desc_emb, review_emb, books_df,
               collab_scores, model=None, user_id=None, user_sim=None, user_item_matrix=None, 
               user_texts=None, user_read=None, trope_embeddings=None,
               w_content=0.6, w_collab=0.4, neg_penalty=0.10, exclude_read=True):
    if model is None:
        model = load_sentence_transformer()
    if trope_embeddings is None:
        trope_embeddings = build_trope_embeddings(model)
    intent     = extract_intent_final(query, model=model, trope_embeddings=trope_embeddings)
    
    # Apply post-processor: remove negated tropes
    intent["tropes"] = remove_negated_base_tropes(query, intent.get("tropes", []))
    
    query_text = build_query_text(intent)

    # Initialize defaults for optional parameters
    if user_texts is None:
        user_texts = {}
    if user_read is None:
        user_read = {}
    if user_sim is None:
        user_sim = pd.Series(dtype=float)
    if user_item_matrix is None:
        user_item_matrix = pd.DataFrame()

    is_known = bool(user_id and user_id in user_texts)
    if is_known:
        taste = user_texts[user_id]
        query_text = (query_text + " " + taste).strip() if query_text else taste

    c_series = compute_content_score(query_text, intent, book_profiles, desc_emb, review_emb, model)
    k_series = compute_collab_score_fn(user_id or "", collab_scores, user_sim, user_item_matrix)

    all_ids = book_profiles["book_id"].values
    c_arr   = c_series.reindex(all_ids, fill_value=0).values
    k_arr   = k_series.reindex(all_ids, fill_value=0).values
    final   = w_content * c_arr + w_collab * k_arr

    neg_kws = intent.get("negative_keywords", [])
    if neg_kws:
        neg_text = " ".join(nk.replace("not_","") for nk in neg_kws)
        neg_vec  = process_query_vec(neg_text, model)
        neg_sim  = compute_similarity(neg_vec, desc_emb, review_emb)
        mn,mx    = neg_sim.min(), neg_sim.max()
        neg_norm = (neg_sim-mn)/(mx-mn+1e-9)
        final   -= neg_penalty * neg_norm * len(neg_kws)
        final    = np.clip(final, 0, 1)

    # Build results
    result = book_profiles[["book_id", "title", "author_clean",
                              "genres_list", "genres_list"]].copy()
    result["content_score"] = c_arr.round(4)
    result["collab_score"]  = k_arr.round(4)
    result["final_score"]   = final.round(4)

    meta = collab_scores[["rating", "totalratings"]].reset_index()
    result = result.merge(meta, on="book_id", how="left")

    if exclude_read and user_id and user_id in user_read:
        result = result[~result["book_id"].isin(user_read[user_id])]

    # Attach link from books_df
    if "link" in books_df.columns:
        link_df = books_df[["book_id", "link"]].drop_duplicates("book_id")
        result  = result.merge(link_df, on="book_id", how="left")
    else:
        result["link"] = ""

    result = (result
              .sort_values("final_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    return result, intent


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    books_path        = st.text_input("books.csv path",        value="books.csv")
    reviews_path      = st.text_input("reviews CSV path",      value="review_clean.csv")
    interactions_path = st.text_input("interactions CSV path", value="interactions.csv")
    top_n             = st.slider("Number of recommendations", 40, 200, 40)

    st.divider()
    st.caption("Hybrid Recommender v3 · Dual-Embedding NLP")


# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gr-navbar">
    <span class="gr-logo">good<span>reads</span></span>
    <span class="gr-badge">Discover AI</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
data_ok = False

# Check file existence
files_exist = all(os.path.exists(p) for p in [books_path, reviews_path, interactions_path])

if not files_exist:
    missing_files = [p for p in [books_path, reviews_path, interactions_path] if not os.path.exists(p)]
    st.markdown("""
    <div class="gr-hero">
        <div class="gr-eyebrow">Intent-Based Discovery</div>
        <div class="gr-headline">What kind of story are you <em>in the mood</em> for?</div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"⚠️ Data files not found: `{'`, `'.join(missing_files)}`\n\nPlease update the file paths in the sidebar (☰).")
    st.stop()

try:
    with st.spinner("Loading model and data…"):
        st_model = load_sentence_transformer()
        books_df, reviews_df, interactions_df = load_data(books_path, reviews_path, interactions_path)

    with st.spinner("Building book profiles…"):
        book_profiles, df_clean = build_book_profiles(books_df, reviews_df)

    with st.spinner("Computing collaborative scores…"):
        collab_scores = build_collab_scores(books_df, df_clean, interactions_df)

    with st.spinner("Encoding book embeddings (first run may take ~1-2 min)…"):
        desc_emb, review_emb = build_embeddings(book_profiles, st_model)

    data_ok = True

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gr-hero">
    <div class="gr-eyebrow">Intent-Based Discovery</div>
    <div class="gr-headline">What kind of story are you <em>in the mood</em> for?</div>
</div>
""", unsafe_allow_html=True)

# Search bar with form for Enter key support
st.markdown("<div style='padding: 0 0 4px 0'></div>", unsafe_allow_html=True)

with st.form(key="search_form", clear_on_submit=False):
    col_input, col_btn = st.columns([7, 1])
    
    with col_input:
        query = st.text_input(
            label="search_query",
            placeholder="e.g. I need a book where love is not returned and the main character has to overcome obstacles to find themselves",
            label_visibility="collapsed",
            key="main_query",
        )
    
    with col_btn:
        search_clicked = st.form_submit_button("Search", use_container_width=True)

# Prompt chips
chips_html = '<div class="gr-chips-row">' + "".join(
    f'<span class="gr-chip">{s}</span>' for s in PROMPT_SUGGESTIONS
) + "</div>"
st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("<hr class='gr-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
search_triggered = search_clicked or st.session_state.get("auto_search", False)

def render_results(recs, intent, tropes_detected, query):
    """Render book cards from cached recs — no recompute needed."""
    genres_detected = intent.get("genres", [])

    if genres_detected or tropes_detected:
        badges_html = '<div class="gr-nlp-row"><span class="gr-nlp-label">NLP detected:</span>'
        for g in genres_detected[:3]:
            badges_html += f'<span class="gr-badge-genre">{g}</span>'
        for t in tropes_detected[:3]:
            label = t.replace("_", " ")
            badges_html += f'<span class="gr-badge-trope">{label}</span>'
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown(f'<div class="gr-results-header"><strong>{len(recs)} books found</strong></div>', unsafe_allow_html=True)

    n_show = st.session_state.get("results_displayed", min(20, len(recs)))

    for idx, (_, row) in enumerate(recs.iterrows()):
        if idx >= n_show:
            break

        # Extract and clean data - prefer title_clean if available
        title_raw = str(row.get("title_clean", row.get("title", "Unknown")))
        original_title = str(row.get("title", ""))
        if title_raw == str(row.get("title", "Unknown")):
            has_special = any(ord(c) > 127 for c in original_title)
            if has_special and "title_clean" in row.index:
                title_raw = str(row.get("title_clean", title_raw))

        title = apply_title_case(title_raw)
        author_raw = str(row.get("author_clean", "Unknown"))
        author = capitalize_author_name(author_raw)

        genre_raw = str(row.get("genre", "")).strip()
        if genre_raw and genre_raw.lower() != "nan":
            genres_list = [g.strip() for g in genre_raw.split(",")][:3]
        else:
            genres_list = []

        avg_rating = float(row.get("rating", 0) or 0)
        total_ratings = int(row.get("totalratings", 0) or 0)
        final_score = float(row.get("final_score", 0))
        link = str(row.get("link", "") or "")

        match_pct = int(round(final_score * 100))
        stars = stars_html(avg_rating)

        explanation = get_explanation(
            book_title=title, author=author, genres_list=genres_list,
            user_query=query, tropes=tropes_detected,
            tones=intent.get("tones", []),
            seed_idx=idx
        )

        card_html = f"""
        <div class="gr-card">
            <div class="gr-card-left">
                <a href="{link}" target="_blank" class="gr-card-title">{title}</a>
                <div class="gr-card-author_clean">{author}</div>
                <div class="gr-card-meta">
                    <span class="gr-stars">{stars}</span>
                    <span class="gr-rating-text"><strong>{avg_rating:.2f}</strong> · {total_ratings:,} ratings</span>
                </div>
                <div class="gr-ai-box">
                    <span class="gr-ai-label">✨ Why this book:</span> {explanation}
                </div>
            </div>
            <div class="gr-card-right">
                <span class="gr-match-badge {match_class(match_pct)}">{match_pct}% match</span>
                <a href="{link}" target="_blank" class="gr-goodreads-link">View on Goodreads ↗</a>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    display_load_more_button(len(recs))


# CASE 1: New search triggered
if query.strip() and search_triggered:
    with st.spinner("Finding your next read…"):
        recs, intent = recommend(
            query        = query,
            top_n        = top_n,
            book_profiles= book_profiles,
            desc_emb     = desc_emb,
            review_emb   = review_emb,
            books_df     = books_df,
            collab_scores= collab_scores,
            model        = st_model,
        )

    tropes_detected = intent.get("tropes", [])

    # Reset pagination only on new query
    if st.session_state.get("last_query") != query:
        st.session_state.results_displayed = min(20, len(recs))
        st.session_state.last_query = query

    # Cache results
    st.session_state.recs_cache = recs
    st.session_state.tropes_detected_cache = tropes_detected
    st.session_state.intent_cache = intent

    render_results(recs, intent, tropes_detected, query)

# CASE 2: Load More rerun — use cached results, skip recompute
elif (
    not search_triggered
    and "recs_cache" in st.session_state
    and st.session_state.get("last_query")
):
    recs            = st.session_state.recs_cache
    intent          = st.session_state.intent_cache
    tropes_detected = st.session_state.tropes_detected_cache
    cached_query    = st.session_state.last_query
    render_results(recs, intent, tropes_detected, cached_query)

# CASE 3: Empty state
elif not query.strip() and "recs_cache" not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #9a9080;">
        <div style="font-size: 48px; margin-bottom: 16px;">📖</div>
        <div style="font-size: 18px; font-family: 'Lora', serif; color: #5a4a3a;">
            Describe the book you're looking for...
        </div>
        <div style="font-size: 14px; margin-top: 8px;">
            Try searching: "a thrilling mystery with a heartwarming ending"
        </div>
    </div>
    """, unsafe_allow_html=True)