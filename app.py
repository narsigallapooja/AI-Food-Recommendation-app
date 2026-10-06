import streamlit as st
import pandas as pd
from recommender import FoodRecommender

st.set_page_config(page_title="AI Food Recommendation System", page_icon="🍴", layout="wide")

@st.cache_resource
def load_recommender():
    return FoodRecommender("data.csv")

recommender = load_recommender()

st.title("🍴 AI Food Recommendation System")
st.write("Get personalized food recommendations using AI and content-based machine learning.")

st.sidebar.header("🔍 Your Preferences")

diet = st.sidebar.selectbox("Dietary Preference", ["Any", "Vegetarian", "Non-Vegetarian", "Vegan"])
cuisine = st.sidebar.selectbox("Cuisine", ["Any"] + recommender.get_cuisines())
max_calories = st.sidebar.slider("Maximum Calories", 100, 1000, 700, 50)
spice = st.sidebar.selectbox("Spice Level", ["Any", "Low", "Medium", "High"])
ingredient = st.sidebar.text_input("Preferred Ingredient", placeholder="Example: Paneer")
top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

if st.sidebar.button("🍽️ Recommend Food", use_container_width=True):
    results = recommender.recommend(
        diet=diet, cuisine=cuisine, max_calories=max_calories,
        spice=spice, ingredient=ingredient, top_n=top_n
    )

    st.subheader("✨ Recommended Foods")

    if results.empty:
        st.warning("No matching foods found. Try changing your preferences.")
    else:
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"# {row['emoji']}")
                with col2:
                    st.markdown(f"### {row['name']}")
                    st.write(
                        f"**Cuisine:** {row['cuisine']} | "
                        f"**Diet:** {row['diet']} | "
                        f"**Calories:** {row['calories']} kcal"
                    )
                    st.write(
                        f"**Protein:** {row['protein']} g | "
                        f"**Spice:** {row['spice']}"
                    )
                    st.write(f"**Ingredients:** {row['ingredients']}")
                    st.info(f"AI Recommendation Score: {row['score']:.2f}")
else:
    st.subheader("🍽️ Explore Food")
    st.write("Select your preferences from the sidebar and click **Recommend Food**.")
    st.dataframe(
        recommender.data[["name", "cuisine", "diet", "calories", "protein", "spice"]],
        use_container_width=True
    )

st.divider()
st.caption("AI Food Recommendation System | Python • Streamlit • Pandas • Scikit-learn")
