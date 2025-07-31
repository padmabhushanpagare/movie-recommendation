import numpy as np
import pandas as pd

class HybridRecommender:
    def __init__(self, svd_model, content_model, df, movies, alpha=0.7):
        self.svd_model = svd_model
        self.content_model = content_model
        self.df = df
        self.movies = movies
        self.alpha = alpha

    def recommend(self, user_id, top_n=10):
        user_rated = self.df[self.df['userId'] == user_id]['movieId'].tolist()
        all_movies = self.movies['movieId'].unique()
        unrated_movies = [mid for mid in all_movies if mid not in user_rated]

        predictions = []
        for movie_id in unrated_movies:
            # Collaborative
            svd_score = self.svd_model.predict(user_id, movie_id).est

            # Content: average similarity to movies the user has already rated
            content_scores = [self.content_model.get_similarity_score(movie_id, rated)
                              for rated in user_rated]
            if content_scores:
                content_score = np.mean(content_scores)
            else:
                content_score = 0

            # Final hybrid score
            final_score = self.alpha * svd_score + (1 - self.alpha) * content_score
            predictions.append((movie_id, final_score))

        # Sort and return top N
        top = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top, columns=['movieId', 'hybrid_score']) \
                 .merge(self.movies, on='movieId')[['title', 'hybrid_score']]
