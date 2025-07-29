import pandas as pd

def get_top_n_popular(df, n=10, min_ratings=50):
    """
    Returns top N most popular movies by average rating,
    considering only those with at least `min_ratings` ratings.
    """
    movie_stats = df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    })
    movie_stats.columns = ['avg_rating', 'rating_count']
    movie_stats = movie_stats.reset_index()

    # Filter by number of ratings
    popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]

    
    # Sort by avg_rating and return top N
    top_movies = popular_movies.sort_values(
        by=['avg_rating', 'rating_count'], 
        ascending=[False, False]
    ).head(n)

    return top_movies