import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender


class HybridRecommender:
	"""Гибридная рекомендательная система.

	Объединяет персональные предсказания SVD (Collaborative)
	и сходство по жанрам TF-IDF (Content-Based).
	"""

	def __init__(self, alpha: float = 0.5, n_factors: int = 20):
		self.alpha = alpha
		self.cb_model = ContentBasedRecommender()
		self.cf_model = CollaborativeRecommender(n_factors=n_factors)
		self.scaler = MinMaxScaler()

	def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
		"""Обучение обоих модулей."""
		self.cb_model.fit(movies_df)
		self.cf_model.fit(ratings_df)
		self.movies_df = movies_df
		self.ratings_df = ratings_df

	def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
		"""Генерирует гибридные рекомендации для пользователя."""
		if user_id not in self.cf_model.predicted_ratings_df.index:
			raise ValueError(f"User ID {user_id} не найден.")

		# 1. Получаем предсказания SVD для всех фильмов
		user_cf_scores = self.cf_model.predicted_ratings_df.loc[user_id].copy()

		# Нормализуем оценки SVD в диапазон [0, 1]
		cf_values = user_cf_scores.values.reshape(-1, 1)
		cf_scaled = self.scaler.fit_transform(cf_values).flatten()
		cf_scores_series = pd.Series(cf_scaled, index=user_cf_scores.index)

		# 2. Получаем просмотренные пользователем фильмы с высокими оценками (>= 4)
		user_history = self.ratings_df[self.ratings_df['user_id'] == user_id]
		liked_movies = user_history[user_history['rating'] >= 4]['movie_id'].tolist()

		# Если пользователь еще ничего не оценил высоко, берем просто его просмотренные
		if not liked_movies:
			liked_movies = user_history['movie_id'].tolist()

		# 3. Считаем средний профиль контентного сходства на основе любимых фильмов
		cb_scores_dict = {m_id: 0.0 for m_id in self.movies_df['movie_id']}

		if liked_movies:
			for m_id in liked_movies:
				if m_id in self.cb_model.movie_id_to_idx:
					idx = self.cb_model.movie_id_to_idx[m_id]
					sim_scores = self.cb_model.similarity_matrix[idx]
					for target_idx, score in enumerate(sim_scores):
						target_m_id = self.cb_model.idx_to_movie_id[target_idx]
						cb_scores_dict[target_m_id] += score

			# Усредняем
			for m_id in cb_scores_dict:
				cb_scores_dict[m_id] /= len(liked_movies)

		cb_scores_series = pd.Series(cb_scores_dict)

		# 4. Взвешенное суммирование (Hybrid Score)
		hybrid_scores = (self.alpha * cf_scores_series) + ((1 - self.alpha) * cb_scores_series)

		# Исключаем уже просмотренные фильмы
		watched_movie_ids = user_history['movie_id'].tolist()
		hybrid_scores = hybrid_scores[~hybrid_scores.index.isin(watched_movie_ids)]

		# Топ-N рекомендаций
		top_movie_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index

		recommendations = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)].copy()
		recommendations['hybrid_score'] = recommendations['movie_id'].map(hybrid_scores)

		return recommendations[['movie_id', 'title', 'genres_text', 'hybrid_score']].sort_values(
			by='hybrid_score', ascending=False
		)