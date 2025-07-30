import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentRecommender:
    def __init__(self, movie_df):
        self.movie_df = movie_df.copy()
        self.movie_df['genres'] = self.movie_df['genres'].fillna('')

        # Vectorize the genres column
        self.vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movie_df['genres'])

        # Compute cosine similarity
        self.similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.movie_df['title'].values:
            return f"'{movie_title}' not found in dataset."

        idx = self.movie_df[self.movie_df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movie_df.iloc[movie_indices][['title', 'genres']]
