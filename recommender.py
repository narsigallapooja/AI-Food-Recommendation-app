import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FoodRecommender:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path).fillna("")
        self.data["combined_features"] = (
            self.data["name"].astype(str) + " " +
            self.data["cuisine"].astype(str) + " " +
            self.data["diet"].astype(str) + " " +
            self.data["ingredients"].astype(str) + " " +
            self.data["spice"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(
            self.data["combined_features"]
        )

    def get_cuisines(self):
        return sorted(self.data["cuisine"].dropna().unique().tolist())

    def recommend(self, diet="Any", cuisine="Any", max_calories=1000,
                  spice="Any", ingredient="", top_n=5):
        filtered = self.data.copy()

        if diet != "Any":
            filtered = filtered[filtered["diet"].str.lower() == diet.lower()]

        if cuisine != "Any":
            filtered = filtered[filtered["cuisine"].str.lower() == cuisine.lower()]

        filtered = filtered[
            pd.to_numeric(filtered["calories"], errors="coerce") <= max_calories
        ]

        if spice != "Any":
            filtered = filtered[filtered["spice"].str.lower() == spice.lower()]

        if ingredient.strip():
            term = ingredient.strip().lower()
            filtered = filtered[
                filtered["ingredients"].str.lower().str.contains(term, na=False)
            ]

        if filtered.empty:
            return pd.DataFrame()

        indexes = filtered.index.tolist()
        selected_matrix = self.feature_matrix[indexes]
        similarity = cosine_similarity(selected_matrix, selected_matrix)

        if len(filtered) == 1:
            scores = [1.0]
        else:
            scores = [
                (similarity[i].sum() - 1) / (len(filtered) - 1)
                for i in range(len(filtered))
            ]

        filtered = filtered.copy()
        filtered["score"] = scores
        return filtered.sort_values("score", ascending=False).head(top_n)
