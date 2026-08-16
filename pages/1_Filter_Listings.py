import streamlit as st # type: ignore
import json
import pandas as pd # type: ignore
import math

st.set_page_config(page_title="Filter Listings", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

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

# Load data
with open("listings.json") as f:
    raw = json.load(f)
df = pd.DataFrame(raw)
# add distance to the dataframe
df["distance"] = df.apply(lambda row: round(haversine(row["latitude"], row["longitude"], 42.2770, -83.7382), 1), axis=1)
df = df.dropna(subset=["price", "bedrooms", "bathrooms", "distance"])

st.markdown('<div class="quiz-title">Filter Listings</div>', unsafe_allow_html=True)
st.markdown('<div class="quiz-subtitle">Narrow down the dataset before starting the quiz.</div>', unsafe_allow_html=True)

# ── Filters ───────────────────────────────────────────────────────────────────
bed_options = sorted(df["bedrooms"].dropna().unique().astype(int).tolist())
bath_options = sorted(df["bathrooms"].dropna().unique().tolist())

# Initialize filter state if not already set
filter_defaults = {
    "filter_price_min": int(df["price"].quantile(0.05)),
    "filter_price_max": int(df["price"].quantile(0.95)),
    "filter_dist_max": 3.0,
    "filter_beds": [int(x) for x in bed_options],
    "filter_baths": [float(x) for x in bath_options],
}
for k, v in filter_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

col1, col2 = st.columns(2)

with col1:
    price_min, price_max = st.slider(
        "Monthly rent ($)",
        min_value=int(df["price"].min()),
        max_value=int(df["price"].max()),
        value=(st.session_state["filter_price_min"], st.session_state["filter_price_max"]),
        step=50,
        key="filter_price",
    )
    dist_max = st.slider(
        "Max distance from campus (miles)",
        min_value=0.0,
        max_value=float(df["distance"].max()),
        value=st.session_state["filter_dist_max"],
        step=0.1,
    )

with col2:
    selected_beds = st.multiselect(
        "Bedrooms",
        options=[int(x) for x in bed_options],
        default=[int(x) for x in st.session_state["filter_beds"]],
    )
    selected_baths = st.multiselect(
        "Bathrooms",
        options=[float(x) for x in bath_options],
        default=[float(x) for x in st.session_state["filter_baths"]],
    )

# Only update session state if the value actually changed
if price_min != st.session_state["filter_price_min"]:
    st.session_state["filter_price_min"] = price_min
    st.rerun()
if price_max != st.session_state["filter_price_max"]:
    st.session_state["filter_price_max"] = price_max
    st.rerun()
if dist_max != st.session_state["filter_dist_max"]:
    st.session_state["filter_dist_max"] = dist_max
    st.rerun()
if selected_beds != st.session_state["filter_beds"]:
    st.session_state["filter_beds"] = selected_beds
    st.rerun()
if selected_baths != st.session_state["filter_baths"]:
    st.session_state["filter_baths"] = selected_baths
    st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[
    (df["price"] >= price_min) &
    (df["price"] <= price_max) &
    (df["distance"] <= dist_max) &
    (df["bedrooms"].isin(selected_beds)) &
    (df["bathrooms"].isin(selected_baths))
]

st.markdown(f"**{len(filtered)}** listings match your filters (out of {len(df)} total)")

# ── Save to session state so the quiz uses filtered data ──────────────────────
if st.button("✅ Apply filters and go to quiz", width="content"):
    st.session_state["filtered_listings"] = filtered.to_dict(orient="records")
    st.switch_page("pages/2_Quiz.py")

# ── Preview filtered listings ─────────────────────────────────────────────────
st.dataframe(
    filtered[["formattedAddress", "price", "bedrooms", "bathrooms", "distance"]]
    .rename(columns={
        "formattedAddress": "Address",
        "price": "Price ($)",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "distance": "Distance (mi)",
    })
    .sort_values("Price ($)")
    .reset_index(drop=True),
    width="stretch",
)