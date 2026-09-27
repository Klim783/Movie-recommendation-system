from pathlib import Path
import pandas as pd


class DataLoader:
    """Класс для загрузки и первичной обработки датасета MovieLens 100K."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Автоматически определяем корень проекта от файла loader.py
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "data" / "raw" / "ml-100k"
        else:
            self.data_dir = Path(data_dir).resolve()

        self.genre_cols = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

    def load_ratings(self) -> pd.DataFrame:
        ratings_path = self.data_dir / 'u.data'

        if not ratings_path.exists():
            raise FileNotFoundError(f"Файл не найден: {ratings_path}")

        cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(ratings_path, sep='\t', names=cols, engine='python')
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        return ratings

    def load_movies(self) -> pd.DataFrame:
        movies_path = self.data_dir / 'u.item'

        if not movies_path.exists():
            raise FileNotFoundError(f"Файл не найден: {movies_path}")

        cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + self.genre_cols
        movies = pd.read_csv(movies_path, sep='|', names=cols, encoding='latin-1')
        movies.drop(columns=['video_release_date', 'IMDb_URL'], inplace=True)

        movies['genres_text'] = movies.apply(
            lambda row: " ".join([g for g in self.genre_cols if row[g] == 1]),
            axis=1
        )

        return movies

    def load_merged_data(self) -> pd.DataFrame:
        ratings = self.load_ratings()
        movies = self.load_movies()
        return pd.merge(ratings, movies, on='movie_id')