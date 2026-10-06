# AI Food Recommendation System

A complete college-level AI food recommendation project built with Python, Streamlit, Pandas and Scikit-learn.

## Features
- Content-based food recommendation
- TF-IDF vectorization
- Cosine similarity
- Diet filtering
- Cuisine filtering
- Calorie filtering
- Spice-level filtering
- Ingredient search
- Recommendation score
- Interactive Streamlit dashboard

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Algorithm

1. Load food dataset.
2. Combine food name, cuisine, diet, ingredients and spice level.
3. Convert text features to TF-IDF vectors.
4. Calculate cosine similarity.
5. Apply user preferences.
6. Rank matching foods.
7. Display the top recommendations.

## Files

- `app.py` - Streamlit user interface
- `recommender.py` - recommendation algorithm
- `data.csv` - food dataset
- `requirements.txt` - dependencies
