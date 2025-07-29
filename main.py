import sys
sys.path.append('src')

 from src.data_loader import load_ratings, load_movies, merge_datasets

ratings = load_ratings()
movies = load_movies()
df = merge_datasets(ratings, movies)

print(df.head())
