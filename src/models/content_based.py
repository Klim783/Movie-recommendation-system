import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """Рекомендательная система на основе содержания (Content-Based Filtering)

    Использует TF-IDF вектора жанров и косинусное сходство между фильмами.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.movies_df = None
        self.similarity_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, movies_df: pd.DataFrame):
        """Обучение TF-IDF на тексте жанров и расчет матрицы сходства."""
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Маппинг ID фильма -> Индекс в датафрейме
        self.movie_id_to_idx = {
            row['movie_id']: idx for idx, row in self.movies_df.iterrows()
        }
        self.idx_to_movie_id = {
            idx: row['movie_id'] for idx, row in self.movies_df.iterrows()
        }

        # Вычисляем TF-IDF матрицу по строкам жанров
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['genres_text'])

        # Косинусное сходство (длина векторов равна 1, скалярного произведения достаточно)
        self.similarity_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Возвращает top_n наиболее похожих фильмов на заданный movie_id."""
        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"Movie ID {movie_id} не найден в базе данных.")

        idx = self.movie_id_to_idx[movie_id]

        # Оценки сходства для данного фильма со всеми остальными
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Сортируем по убыванию сходства
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Пропускаем индекс 0, так как это сам фильм
        sim_scores = sim_scores[1 : top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = scores

        return recommendations[['movie_id', 'title', 'genres_text', 'similarity_score']]