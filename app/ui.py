import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
	sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.models.hybrid import HybridRecommender

# Page Configuration
st.set_page_config(
	page_title="Movie Recommender System",
	layout="wide"
)

st.title("Hybrid Movie Recommendation System")
st.caption("Combining Content-Based Filtering (TF-IDF) & Collaborative Filtering (SVD)")


# Load Data and Cache Model Initialization
@st.cache_resource
def load_and_train_model():
	loader = DataLoader()
	ratings = loader.load_ratings()
	movies = loader.load_movies()
	return loader, ratings, movies


with st.spinner("Loading MovieLens dataset..."):
	loader, ratings_df, movies_df = load_and_train_model()

# Sidebar Settings
st.sidebar.header("Recommendation Parameters")

user_ids = sorted(ratings_df['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_ids, index=0)

top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

alpha = st.sidebar.slider(
	"Collaborative vs Content Weight (α)",
	min_value=0.0,
	max_value=1.0,
	value=0.6,
	step=0.05,
	help="Higher values give more weight to user ratings (SVD), lower values prioritize genre similarity (TF-IDF)."
)

n_factors = st.sidebar.number_input("SVD Latent Factors", min_value=5, max_value=50, value=20)


# Instantiate and Train Hybrid Model
@st.cache_resource
def get_fitted_hybrid_model(alpha_val, factors):
	model = HybridRecommender(alpha=alpha_val, n_factors=factors)
	model.fit(ratings_df=ratings_df, movies_df=movies_df)
	return model


hybrid_model = get_fitted_hybrid_model(alpha, n_factors)

# Main UI Columns
col1, col2 = st.columns([1, 2])

with col1:
	st.subheader(f"Top Rated Movies by User #{selected_user}")
	user_history = ratings_df[ratings_df['user_id'] == selected_user].sort_values(
		by='rating', ascending=False
	).head(5)

	user_history_df = user_history.merge(movies_df, on='movie_id')[['title', 'genres_text', 'rating']]
	st.dataframe(user_history_df, hide_index=True, use_container_width=True)

with col2:
	st.subheader(f"Recommended Movies for User #{selected_user}")

	if st.button("Generate Recommendations", type="primary"):
		with st.spinner("Computing hybrid recommendation scores..."):
			recs = hybrid_model.recommend(user_id=selected_user, top_n=top_n)

			# Format display table
			recs_display = recs.rename(columns={
				'title': 'Movie Title',
				'genres_text': 'Genres',
				'hybrid_score': 'Match Score'
			})
			recs_display['Match Score'] = recs_display['Match Score'].apply(lambda x: f"{x:.4f}")

			st.dataframe(
				recs_display[['Movie Title', 'Genres', 'Match Score']],
				hide_index=True,
				use_container_width=True
			)