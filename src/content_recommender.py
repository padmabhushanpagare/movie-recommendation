import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, movies_df:pd.DataFrame):
        self.movies = movies_df.copy()
        self._prepare()

    def _prepare(self):
        #Preprocess genres
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ", regex=False )

        # TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies["genres"])

        # Cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Title to index mapping
        self.movie_indices = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()

    def get_similar_movies(self, title: str, top_n: int = 10):
        if title not in self.movie_indices:
            return []
        
        idx = self.movie_indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices_sorted = [i[0] for i in sim_scores]
        return self.movies["title"].iloc[movie_indices_sorted].tolist()
