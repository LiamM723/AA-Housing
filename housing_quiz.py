"""
housing_quiz.py  –  Pairwise preference quiz for student housing
Run with:  streamlit run housing_quiz.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Find Your Place",
    page_icon="🏠",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F4EF;
    color: #1A1A1A;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

/* Title */
.quiz-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1.15;
    color: #1A1A1A;
    margin-bottom: 0.2rem;
}
.quiz-subtitle {
    font-size: 1rem;
    color: #666;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Progress */
.progress-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.4rem;
}

/* Listing card */
.listing-card {
    background: #FFFFFF;
    border: 1.5px solid #E5E0D8;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s;
}
.listing-card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.08);
}
.listing-address {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1A1A1A;
    margin-bottom: 0.8rem;
}
.listing-price {
    font-size: 1.7rem;
    font-weight: 600;
    color: #2D6A4F;
    margin-bottom: 0.6rem;
}
.listing-meta {
    display: flex;
    gap: 1.2rem;
    font-size: 0.88rem;
    color: #555;
    margin-bottom: 0.4rem;
}
.listing-tag {
    display: inline-block;
    background: #F0EDE8;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: #666;
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}
.vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #C5BAA8;
    padding: 0 1rem;
    margin-top: 3.5rem;
}

/* Result card */
.result-card {
    background: #FFFFFF;
    border: 1.5px solid #E5E0D8;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
}
.result-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #888;
}
.result-score-bar-bg {
    background: #F0EDE8;
    border-radius: 99px;
    height: 6px;
    margin-top: 0.4rem;
}
.weight-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #F0EDE8;
}
.weight-label { color: #444; }
.weight-value { font-weight: 600; color: #2D6A4F; }
</style>
""", unsafe_allow_html=True)

# distance formula using Haversine formula--------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance
#-------------------------------------------------------------------------------

# ── Sample data (replace with your RentCast API data) ────────────────────────
# Each listing must have: price, bedrooms, bathrooms, distance (miles from campus)

with open("listings.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# add distance to the dataframe
df["distance"] = df.apply(lambda row: round(haversine(row["latitude"], row["longitude"], 42.2770, -83.7382), 1), axis=1)

df = df.dropna(subset=["price", "bedrooms", "bathrooms", "distance"])

# Assign a clean integer id if one doesn't already exist
if "id" not in df.columns:
    df["id"] = range(len(df))

SAMPLE_LISTINGS = df[["id", "formattedAddress", "price", "bedrooms", "bathrooms", "distance"]].to_dict(orient="records")

FEATURES = ["price", "bedrooms", "bathrooms", "distance"]
FEATURE_LABELS = {
    "price":     "Monthly Rent",
    "bedrooms":  "Bedrooms",
    "bathrooms": "Bathrooms",
    "distance":  "Distance from Campus",
}
# Sign: negative means "less is better" (price, distance), positive means "more is better"
FEATURE_DIRECTION = {"price": -1, "bedrooms": 1, "bathrooms": 1, "distance": -1}

MIN_COMPARISONS_TO_TRAIN = 6
QUIZ_LENGTH = 15  # stop accepting new comparisons after this many


# ── Feature helpers ───────────────────────────────────────────────────────────
def feature_vector(listing):
    return np.array([listing[f] for f in FEATURES], dtype=float)

def feature_diff(a, b):
    """Difference vector: positive value means A > B for each feature."""
    return feature_vector(a) - feature_vector(b)

def normalize_features(listings):
    """Return a scaler fitted on all listings — used for diverse-pair selection."""
    X = np.array([feature_vector(l) for l in listings])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ── Pair selection ────────────────────────────────────────────────────────────
def pick_diverse_pair(listings, seen_pairs, scaler):
    """
    Early-quiz strategy: pick the pair that is most different from each other
    (maximum Euclidean distance in normalised feature space), excluding pairs
    already shown.
    """
    X_norm = scaler.transform(np.array([feature_vector(l) for l in listings]))
    best_pair, best_dist = None, -1
    candidates = list(combinations(range(len(listings)), 2))
    random.shuffle(candidates)  # avoid always picking the same extreme pair
    for i, j in candidates:
        if (listings[i]["id"], listings[j]["id"]) in seen_pairs:
            continue
        d = np.linalg.norm(X_norm[i] - X_norm[j])
        if d > best_dist:
            best_dist = d
            best_pair = (listings[i], listings[j])
    return best_pair

def pick_uncertain_pair(listings, model, scaler, seen_pairs):
    """
    Active-learning strategy: pick the pair the current model is LEAST confident
    about — i.e. the predicted probability is closest to 0.5.
    """
    X_norm = scaler.transform(np.array([feature_vector(l) for l in listings]))
    candidates = list(combinations(range(len(listings)), 2))
    random.shuffle(candidates)
    best_pair, best_uncertainty = None, float("inf")
    for i, j in candidates:
        if (listings[i]["id"], listings[j]["id"]) in seen_pairs:
            continue
        diff = (X_norm[i] - X_norm[j]).reshape(1, -1)
        prob = model.predict_proba(diff)[0][1]  # P(A preferred over B)
        uncertainty = abs(prob - 0.5)
        if uncertainty < best_uncertainty:
            best_uncertainty = uncertainty
            best_pair = (listings[i], listings[j])
    return best_pair

def select_pair(listings, comparisons, model, scaler):
    seen_pairs = {
        (c["a"]["id"], c["b"]["id"]) for c in comparisons
    } | {
        (c["b"]["id"], c["a"]["id"]) for c in comparisons
    }
    n = len(comparisons)
    if n < MIN_COMPARISONS_TO_TRAIN or model is None:
        return pick_diverse_pair(listings, seen_pairs, scaler)
    else:
        return pick_uncertain_pair(listings, model, scaler, seen_pairs)


# ── Model training ────────────────────────────────────────────────────────────
def build_training_data(comparisons, scaler, listings):
    """
    Each comparison → two rows (original + flipped) in normalised feature space.
    """
    X_norm = {
        l["id"]: scaler.transform(feature_vector(l).reshape(1, -1))[0]
        for l in listings
    }
    X, y = [], []
    for c in comparisons:
        va = X_norm[c["a"]["id"]]
        vb = X_norm[c["b"]["id"]]
        diff = va - vb
        label = 1 if c["chosen"] == "a" else 0
        X.append(diff);       y.append(label)
        X.append(-diff);      y.append(1 - label)   # flipped
    return np.array(X), np.array(y)

def train_model(comparisons, scaler, listings):
    if len(comparisons) < MIN_COMPARISONS_TO_TRAIN:
        return None
    X, y = build_training_data(comparisons, scaler, listings)
    if len(np.unique(y)) < 2:
        return None  # need both classes
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def rank_listings(listings, model, scaler):
    """Score every listing vs. the average listing; sort descending."""
    X_norm = scaler.transform(np.array([feature_vector(l) for l in listings]))
    avg_norm = X_norm.mean(axis=0)
    scores = []
    for i, listing in enumerate(listings):
        diff = (X_norm[i] - avg_norm).reshape(1, -1)
        score = model.predict_proba(diff)[0][1]
        scores.append((score, listing))
    return sorted(scores, key=lambda x: x[0], reverse=True)

def interpret_weights(model, scaler):
    """
    Translate logistic regression coefficients back into human-readable weights,
    adjusting for direction (lower price/distance is better).
    """
    coef = model.coef_[0]  # one coef per feature in normalised space
    # Adjust sign so that "higher weight = better for the user"
    adjusted = np.array([
        coef[i] * FEATURE_DIRECTION[f] for i, f in enumerate(FEATURES)
    ])
    # Normalise to sum to 1 for display
    total = np.abs(adjusted).sum()
    if total == 0:
        return {f: 0.0 for f in FEATURES}
    normalised = adjusted / total
    return {f: float(normalised[i]) for i, f in enumerate(FEATURES)}


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "comparisons": [],
        "current_pair": None,
        "model": None,
        "scaler": normalize_features(SAMPLE_LISTINGS),
        "quiz_done": False,
        "listings": SAMPLE_LISTINGS,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
s = st.session_state


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_listing_card(listing, side_label):
    deal_score = ""
    if s.model:
        X_norm = s.scaler.transform(feature_vector(listing).reshape(1, -1))
        avg_norm = s.scaler.transform(
            np.array([feature_vector(l) for l in s.listings]).mean(axis=0).reshape(1, -1)
        )
        prob = s.model.predict_proba(X_norm - avg_norm)[0][1]
        deal_score = f'<span class="listing-tag">Match score: {prob:.0%}</span>'
 
    st.markdown(f"""
    <div class="listing-card">
        <div class="listing-address">{listing['formattedAddress']}</div>
        <div class="listing-price">${listing['price']:,}<span style="font-size:0.9rem;font-weight:300;color:#888">/mo</span></div>
        <div class="listing-meta">
            <span>🛏 {listing['bedrooms']} bed</span>
            <span>🚿 {listing['bathrooms']} bath</span>
            <span>📍 {listing['distance']} mi from campus</span>
        </div>
        {deal_score}
    </div>
    """, unsafe_allow_html=True)


def choose(side):
    """Record a comparison and advance the quiz."""
    pair = s.current_pair
    s.comparisons.append({"a": pair[0], "b": pair[1], "chosen": side})
    s.model = train_model(s.comparisons, s.scaler, s.listings)
    if len(s.comparisons) >= QUIZ_LENGTH:
        s.quiz_done = True
    else:
        s.current_pair = None  # force re-selection on next render


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="quiz-title">Find Your Place</div>', unsafe_allow_html=True)
st.markdown('<div class="quiz-subtitle">Answer a few quick comparisons and we\'ll learn what matters most to you.</div>', unsafe_allow_html=True)

# ── Results view ──────────────────────────────────────────────────────────────
if s.quiz_done and s.model:
    st.markdown("---")
    st.markdown("### 🎯 Your Personalised Rankings")

    col_r, col_w = st.columns([2, 1])

    with col_r:
        ranked = rank_listings(s.listings, s.model, s.scaler)
        for rank, (score, listing) in enumerate(ranked, 1):
            bar_width = int(score * 100)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-rank">#{rank}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.1rem">{listing['formattedAddress']}</div>
                <div style="display:flex;align-items:center;gap:1rem;margin-top:0.3rem">
                    <span style="font-weight:600;color:#2D6A4F">${listing['price']:,}/mo</span>
                    <span style="font-size:0.82rem;color:#888">🛏 {listing['bedrooms']}  🚿 {listing['bathrooms']}  📍 {listing['distance']} mi</span>
                </div>
                <div class="result-score-bar-bg">
                    <div style="width:{bar_width}%;background:#2D6A4F;height:6px;border-radius:99px"></div>
                </div>
                <div style="font-size:0.75rem;color:#aaa;margin-top:0.2rem">Match: {score:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_w:
        st.markdown("**What you care about**")
        weights = interpret_weights(s.model, s.scaler)
        for feature, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
            direction = "↓ less" if FEATURE_DIRECTION[feature] == -1 else "↑ more"
            pct = f"{abs(weight):.0%}"
            st.markdown(f"""
            <div class="weight-row">
                <span class="weight-label">{FEATURE_LABELS[feature]} <span style="color:#bbb;font-size:0.75rem">({direction})</span></span>
                <span class="weight-value">{pct}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Retake quiz"):
            for k in ["comparisons", "current_pair", "model", "quiz_done"]:
                del st.session_state[k]
            st.rerun()

# ── Quiz view ─────────────────────────────────────────────────────────────────
else:
    n = len(s.comparisons)
    progress = n / QUIZ_LENGTH

    st.markdown(f'<div class="progress-label">Comparison {n + 1} of {QUIZ_LENGTH}</div>', unsafe_allow_html=True)
    st.progress(progress)

    # Show active learning status once it kicks in
    if n >= MIN_COMPARISONS_TO_TRAIN and s.model:
        st.caption("✨ Model trained — now showing pairs it's least sure about (active learning)")

    # Select pair if needed
    if s.current_pair is None:
        pair = select_pair(s.listings, s.comparisons, s.model, s.scaler)
        if pair is None:
            st.warning("All pairs have been shown! Showing results.")
            s.quiz_done = True
            st.rerun()
        s.current_pair = pair

    listing_a, listing_b = s.current_pair

    # Render cards + buttons
    col_a, col_vs, col_b = st.columns([5, 1, 5])

    with col_a:
        render_listing_card(listing_a, "a")
        if st.button("I'd prefer this one →", key="choose_a", use_container_width=True):
            choose("a")
            st.rerun()

    with col_vs:
        st.markdown('<div class="vs-divider">vs</div>', unsafe_allow_html=True)

    with col_b:
        render_listing_card(listing_b, "b")
        if st.button("← I'd prefer this one", key="choose_b", use_container_width=True):
            choose("b")
            st.rerun()

    # Skip button
    st.markdown("<br>", unsafe_allow_html=True)
    skip_col, _ = st.columns([1, 4])
    with skip_col:
        if st.button("Skip – too similar to judge", use_container_width=True):
            s.current_pair = None
            st.rerun()

    # Early results option
    if n >= MIN_COMPARISONS_TO_TRAIN and s.model:
        st.markdown("---")
        if st.button("✅ I've answered enough – show my results"):
            s.quiz_done = True
            st.rerun()