from pathlib import Path
import pandas as pd


class DataLoader:
    """Класс для загрузки и первичной обработки датасета MovieLens 100K."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
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

        # Обход бага Python 3.14: чтение файла напрямую через pure Python
        records = []
        with open(ratings_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    records.append((int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])))

        ratings = pd.DataFrame(records, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        return ratings

    def load_movies(self) -> pd.DataFrame:
        movies_path = self.data_dir / 'u.item'

        if not movies_path.exists():
            raise FileNotFoundError(f"Файл не найден: {movies_path}")

        records = []
        with open(movies_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 24:
                    movie_id = int(parts[0])
                    title = parts[1]
                    release_date = parts[2]
                    video_release = parts[3]
                    imdb_url = parts[4]
                    genres = [int(x) for x in parts[5:24]]
                    records.append([movie_id, title, release_date, video_release, imdb_url] + genres)

        cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + self.genre_cols
        movies = pd.DataFrame(records, columns=cols)
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