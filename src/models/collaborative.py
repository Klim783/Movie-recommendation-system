import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class CollaborativeRecommender:
	"""Рекомендательная система на основе коллаборативной фильтрации (SVD).

	Разлагает матрицу взаимодействий User-Item на скрытые факторы (Latent Features).
	"""

	def __init__(self, n_factors: int = 20):
		self.n_factors = n_factors
		self.user_item_matrix = None
		self.user_ids = None
		self.movie_ids = None
		self.predicted_ratings_df = None

	def fit(self, ratings_df: pd.DataFrame):
		"""Обучение SVD: построение матрицы и сингулярное разложение."""
		# 1. Строим сводную таблицу User-Item
		self.user_item_matrix = ratings_df.pivot(
			index='user_id',
			columns='movie_id',
			values='rating'
		).fillna(0)

		self.user_ids = self.user_item_matrix.index
		self.movie_ids = self.user_item_matrix.columns

		# 2. Нормализуем матрицы (вычитаем среднюю оценку пользователя)
		user_ratings_mean = np.mean(self.user_item_matrix.values, axis=1)
		matrix_demeaned = self.user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

		# 3. Применяем SVD (Singular Value Decomposition)
		U, sigma, Vt = svds(matrix_demeaned, k=self.n_factors)
		sigma = np.diag(sigma)

		# 4. Восстанавливаем матрицу предсказанных оценок
		all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

		self.predicted_ratings_df = pd.DataFrame(
			all_user_predicted_ratings,
			columns=self.movie_ids,
			index=self.user_ids
		)

	def recommend_for_user(
			self,
			user_id: int,
			ratings_df: pd.DataFrame,
			movies_df: pd.DataFrame,
			top_n: int = 10
	) -> pd.DataFrame:
		"""Возвращает топ персональных рекомендаций для указанного пользователя."""
		if user_id not in self.predicted_ratings_df.index:
			raise ValueError(f"User ID {user_id} не найден в обучающей выборке.")

		# Получаем предсказанные оценки пользователя
		user_predictions = self.predicted_ratings_df.loc[user_id].sort_values(ascending=False)

		# Фильтруем фильмы, которые пользователь уже оценивал
		user_history = ratings_df[ratings_df['user_id'] == user_id]['movie_id']
		recommendations = user_predictions[~user_predictions.index.isin(user_history)].head(top_n)

		# Формируем итоговый DataFrame
		rec_df = movies_df[movies_df['movie_id'].isin(recommendations.index)].copy()
		rec_df['predicted_rating'] = rec_df['movie_id'].map(recommendations)

		return rec_df[['movie_id', 'title', 'genres_text', 'predicted_rating']].sort_values(
			by='predicted_rating', ascending=False
		)