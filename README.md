# AAHousing — Student Housing Finder

A Streamlit web app that helps University of Michigan students find rental listings that actually match their preferences, using pairwise comparisons and active learning.

<video>src="/Users/liammiller/Downloads/My Movie 1.mp4"</video>

## How it works

1. **Filter listings** — Narrow ~500 live Ann Arbor rental listings (obtained from the RentCast API) by price, distance from campus, bedrooms, and bathrooms before starting the quiz.
2. **Compare listings pairwise** — Rather than asking users to assign explicit weights to features, the app shows pairs of listings — each with an embedded interactive Google Street View panorama — and asks which one the user prefers. Each choice becomes a training example.
3. **Learn preferences with logistic regression** — A logistic regression model is trained in real time on feature-difference vectors derived from each comparison, learning what the user implicitly values.
4. **Active learning for efficient questioning** — After an initial batch of diverse comparisons, the model selects the most *informative* pairs to show next (the ones it's least certain about), and the quiz ends automatically once the model's average confidence across all pairs clears a threshold — rather than asking a fixed number of questions regardless of how confident it already is.
5. **Live deal detection** — Each listing card shows a real-time "above/below predicted rent" indicator, powered by a linear regression trained on bedrooms, bathrooms, and distance from a predetermined landmark.
6. **Rank every listing** — Once the quiz concludes, every filtered listing is scored and ranked according to the user's learned preferences, with a breakdown of which features mattered most and a map of the top 10 results.

## Tech stack

- **Python** — core pipeline and ML
- **Streamlit** — multi-page web interface
- **scikit-learn** — logistic regression (preference learning) and linear regression (price prediction)
- **pandas / numpy** — data handling
- **folium** — results map of top-ranked listings
- **Google Maps Street View API** — interactive per-listing panoramas during the quiz

## Project structure

```
AAHousing/
├── listings.json              # Rental listing dataset
├── Home.py                    # Landing page
├── pages/
│   ├── 1_Filter_Listings.py   # Filter listings by price, distance, beds, baths
│   └── 2_Quiz.py              # Pairwise comparison quiz + preference model + rankings
└── .streamlit/
    ├── config.toml            # App theming
```

## Setup

Clone the repo and install dependencies:
```bash
git clone https://github.com/yourusername/AAHousing.git
cd AAHousing
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API keys

The comparison quiz uses the [Google Maps Street View API](https://developers.google.com/maps/documentation/streetview) to show an interactive panorama for each listing. Create `.streamlit/secrets.toml` in the project root:
```toml
GOOGLE_MAPS_API_KEY = "your_key_here"
```
Without this key, the quiz page will fail when trying to render listing cards, since every card checks Street View availability and requests a panorama.

Additionally, listings.json was originally obtained during development using the [RentCast API](https://developers.rentcast.io/reference/introduction) and was not updated further due to the proof-of-concept nature of this project. If you would like to work with current data, obtain an API key from RentCast and place it in `.streamlit/secrets.toml` as with the Google Maps API:
```toml
RC_API_KEY = "your_key_here"
```
Then request the new data and write it into listings.json. The quiz will now cover the new set of listings.


## Usage

Run the Streamlit app locally:
```bash
streamlit run Home.py
```
Start on the landing page, filter listings to a relevant subset, then take the comparison quiz to get a personalized ranking.

## Known limitations

- As of August 17 2026, The interactive street view was discovered to be non-functional. Since it is not part of the core functionality of the ranking algorithm and this project was not developed with the intent to become a fully public website, this issue is low on my list of priorities and will be left alone for the near future.
- Listing data is a static snapshot (`listings.json`) rather than a live feed.
- The price-prediction models use a small, fixed feature set (bedrooms, bathrooms, and distance to a predetermined landmark) and don't account for other factors that affect rent (amenities, building age, etc.).
- Preference learning starts cold each session — the model doesn't currently persist or reuse a user's preferences across visits.
- The results map only plots the top 10 ranked listings, not the full ranked list.
- Street View availability isn't guaranteed for every address; listings without coverage fall back to a text notice or blank gray card.
- The active-learning confidence threshold (`CONFIDENCE_THRESHOLD`) is manually tuned rather than derived analytically.

## Possible future improvements

- Persist learned user preferences across sessions
- Live listing data via consistently requesting new data from RentCast- not enough to trigger a rate limit, but enough to remain current
- Incorporate additional features (amenities, walkability, noise) into both the price model and preference learning
- Extend the results map to show more (or all) ranked listings, not just the top 10
- Analytically derive the active-learning confidence threshold instead of manual tuning

## Credits

Built by Liam Miller. Uses features from the [Google Maps API](https://developers.google.com/maps/documentation) and listing data from [RentCast](https://developers.rentcast.io/reference/introduction).