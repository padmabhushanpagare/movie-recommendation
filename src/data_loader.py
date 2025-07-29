# src/data_loader.py

import pandas as pd
import os

def load_data(data_path="../data"):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    data_path = os.path.abspath(data_path)
    print("Looking for data in:", data_path)
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    df = pd.merge(ratings, movies, on='movieId')
    return ratings, movies, df
