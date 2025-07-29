import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_item_user_matrix(df):
    """
    Creates a matrix where rows are movies and columns are users.
    Cells are ratings (NaN = not rated).
    """
    return df.pivot_table(index='movieId', columns='userId', values='rating')

def get_similar_movies(movie_id, ratings_matrix, top_n=10):
    """
    Given a movie_id and a movie-user matrix, return the most similar movies
    based on cosine similarity.
    """
    similarity = cosine_similarity(ratings_matrix.fillna(0))
    similarity_df = pd.DataFrame(similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

    similar_scores = similarity_df[movie_id].sort_values(ascending=False).drop(movie_id)
    return similar_scores.head(top_n)

