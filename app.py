# app.py

import streamlit as st
import pandas as pd

import os, sys
sys.path.append(os.path.abspath("."))

from src.data_loader import load_data
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from src.content_recommender import ContentRecommender

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 MovieLens Recommender System")

# Load data
ratings, movies, df = load_data()

# Train SVD model on full data
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)
content_rec = ContentRecommender(movies)

# Sidebar: user selection
user_ids = df['userId'].unique().tolist()
selected_user = st.sidebar.selectbox("Select a User ID", user_ids)

# Recommend for selected user
user_rated = df[df['userId'] == selected_user]['movieId'].tolist()
all_movies = df['movieId'].unique()
unrated_movies = [m for m in all_movies if m not in user_rated]

preds = [model.predict(selected_user, mid) for mid in unrated_movies]
top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:10]

top_movie_ids = [p.iid for p in top_preds]
recs = pd.DataFrame({
    "movieId": top_movie_ids,
    "predicted_rating": [p.est for p in top_preds]
}).merge(movies, on='movieId')[['title', 'predicted_rating']]

st.subheader("🎬 Recommend Similar Movies (Content-Based)")

movie_titles = movies["title"].sort_values().tolist()
selected_movie = st.selectbox("Pick a movie:", movie_titles)

if st.button("Get Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = content_rec.get_similar_movies(selected_movie, top_n=5)

    if recommendations:
        st.success("You might also like:")
        for title in recommendations:
            st.write(f"- {title}")
    else:
        st.warning("No recommendations found.")


# Display results
st.subheader(f"Top 10 Recommendations for User {selected_user}")
st.table(recs)
