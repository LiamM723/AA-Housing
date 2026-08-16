"""
Home.py  –  Landing page for the student housing finder
Run with:  streamlit run Home.py
"""

import streamlit as st # type: ignore

st.set_page_config(
    page_title="Find Your Place",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F4EF;
    color: #1A1A1A;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
button[data-testid="collapsedControl"] { display: block !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hero */
.hero {
    background: #00274C;
    padding: 5rem 4rem 4rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(45,106,79,0.4) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 10%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(45,106,79,0.2) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #F7F4EF;
    margin-bottom: 1.2rem;
    font-weight: 500;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    line-height: 1.05;
    color: #F7F4EF;
    margin-bottom: 1.5rem;
    max-width: 700px;
}
.hero-title em {
    font-style: italic;
    color: #FFCB05;
}
.hero-body {
    font-size: 1.1rem;
    color: #F7F4EF;
    font-weight: 300;
    max-width: 540px;
    line-height: 1.7;
    margin-bottom: 2.5rem;
}
.hero-stats {
    display: flex;
    gap: 3rem;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #333;
}
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #FFCB05;
    line-height: 1;
}
.stat-label {
    font-size: 0.8rem;
    color: #F7F4EF;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
}

/* How it works */
.section {
    padding: 4rem;
    background: #F7F4EF;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1A1A1A;
    margin-bottom: 2.5rem;
}
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
    margin-bottom: 3rem;
}
.step-card {
    background: #FFFFFF;
    border: 1.5px solid #E5E0D8;
    border-radius: 16px;
    padding: 1.8rem;
    position: relative;
}
.step-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #E5E0D8;
    line-height: 1;
    margin-bottom: 0.8rem;
}
.step-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1A1A1A;
    margin-bottom: 0.5rem;
}
.step-body {
    font-size: 0.88rem;
    color: #666;
    line-height: 1.6;
    font-weight: 300;
}
.step-icon {
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
}

/* ML explainer */
.ml-section {
    background: #00274C;
    padding: 4rem;
}
.ml-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #F7F4EF;
    margin-bottom: 1rem;
}
.ml-body {
    font-size: 1rem;
    color: #F7F4EF;
    font-weight: 300;
    line-height: 1.8;
    max-width: 680px;
    margin-bottom: 2rem;
}
.ml-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    max-width: 700px;
}
.ml-card {
    background: #0B3257;
    border: 1px solid #0B3257;
    border-radius: 12px;
    padding: 1.4rem;
}
.ml-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #FFCB05;
    margin-bottom: 0.4rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.ml-card-body {
    font-size: 0.85rem;
    color: #F7F4EF;
    line-height: 1.6;
}

/* CTA */
.cta-section {
    padding: 4rem;
    background: #F7F4EF;
    text-align: center;
}
.cta-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1A1A1A;
    margin-bottom: 0.8rem;
}
.cta-body {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
    font-weight: 300;
}

[data-testid="stPageLink"] a {
    text-align: center !important;
    justify-content: center !important;
}

/* Animate in */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-eyebrow { animation: fadeUp 0.5s ease both; }
.hero-title   { animation: fadeUp 0.5s ease 0.1s both; }
.hero-body    { animation: fadeUp 0.5s ease 0.2s both; }
.hero-stats   { animation: fadeUp 0.5s ease 0.3s both; }
.step-card:nth-child(1) { animation: fadeUp 0.5s ease 0.1s both; }
.step-card:nth-child(2) { animation: fadeUp 0.5s ease 0.2s both; }
.step-card:nth-child(3) { animation: fadeUp 0.5s ease 0.3s both; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Student Housing · Ann Arbor</div>
    <div class="hero-title">Find a place that <em>actually</em> fits you.</div>
    <div class="hero-body">
        Stop scrolling through hundreds of listings. Answer a few quick comparisons
        and let the model learn what you care about — then see every rental ranked
        just for you.
    </div>
    <div class="hero-stats">
        <div>
            <div class="stat-number">500+</div>
            <div class="stat-label">Live listings</div>
        </div>
        <div>
            <div class="stat-number">~15</div>
            <div class="stat-label">Comparisons to rank</div>
        </div>
        <div>
            <div class="stat-number">4</div>
            <div class="stat-label">Features learned</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section">
    <div class="section-title">How it works</div>
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-icon">🎛️</div>
            <div class="step-number">01</div>
            <div class="step-title">Set your filters</div>
            <div class="step-body">
                Start by narrowing the dataset to listings that make sense for you —
                set a price range, max distance from campus, and bedroom count.
            </div>
        </div>
        <div class="step-card">
            <div class="step-icon">⚖️</div>
            <div class="step-number">02</div>
            <div class="step-title">Compare listings</div>
            <div class="step-body">
                You'll be shown pairs of listings and asked which you'd prefer.
                The algorithm learns from your choices, not from you manually
                entering weights.
            </div>
        </div>
        <div class="step-card">
            <div class="step-icon">🏆</div>
            <div class="step-number">03</div>
            <div class="step-title">See your rankings</div>
            <div class="step-body">
                Once the model has learned your preferences, every listing is
                scored and ranked for you — with a breakdown of what you
                seem to care about most.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ML explainer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="ml-section">
    <div class="ml-title">Powered by preference learning</div>
    <div class="ml-body">
        Rather than asking you to assign weights to features — which is hard to do
        accurately — this app infers your preferences from pairwise comparisons.
        Each choice you make is a training example. The model updates in real time
        and uses active learning to show you the most informative pairs first.
    </div>
    <div class="ml-grid">
        <div class="ml-card">
            <div class="ml-card-title">Pairwise ranking</div>
            <div class="ml-card-body">
                Logistic regression on feature difference vectors, trained on
                your comparison choices.
            </div>
        </div>
        <div class="ml-card">
            <div class="ml-card-title">Active learning</div>
            <div class="ml-card-body">
                After initial training, the model selects the most
                informative pairs to show next — or stops early if
                it's already confident in your preferences.
            </div>
        </div>
        <div class="ml-card">
            <div class="ml-card-title">Price regression</div>
            <div class="ml-card-body">
                A separate linear regression flags listings that are above or
                below their expected rent given their features.
            </div>
        </div>
        <div class="ml-card">
            <div class="ml-card-title">Convergence detection</div>
            <div class="ml-card-body">
                The quiz ends automatically when the model's uncertainty across
                all pairs drops below a threshold — no fixed question count.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cta-section">
    <div class="cta-title">Ready to find your place?</div>
    <div class="cta-body">Start by filtering the listings, then take the quiz.</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    st.page_link("pages/1_Filter_Listings.py", label="Get started →", width="stretch")

st.markdown("<div style='height: 4rem'></div>", unsafe_allow_html=True)