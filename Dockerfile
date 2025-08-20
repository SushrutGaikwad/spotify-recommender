FROM python:3.12

WORKDIR /app

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY data/interim/Music_Info_cleaned.csv            data/interim/
COPY data/interim/collab_filtered_songs.csv         data/interim/
COPY data/interim/collab_filtered_songs_sorted.csv  data/interim/
COPY data/processed/Music_Info_cleaned_transformed_CBF.npz data/processed/
COPY data/processed/collab_track_ids.npy            data/processed/
COPY data/processed/collab_interaction_matrix.npz   data/processed/
COPY data/processed/hybrid_transformed_cbf.npz      data/processed/

COPY spotify_recommender/__init__.py                               spotify_recommender/
COPY spotify_recommender/config.py                                 spotify_recommender/
COPY spotify_recommender/content_based_filtering/__init__.py       spotify_recommender/content_based_filtering/
COPY spotify_recommender/content_based_filtering/recommendation.py spotify_recommender/content_based_filtering/
COPY spotify_recommender/collaborative_filtering/__init__.py       spotify_recommender/collaborative_filtering/
COPY spotify_recommender/collaborative_filtering/recommendation.py spotify_recommender/collaborative_filtering/
COPY spotify_recommender/hybrid_recommendation/__init__.py         spotify_recommender/hybrid_recommendation/
COPY spotify_recommender/hybrid_recommendation/recommendation.py   spotify_recommender/hybrid_recommendation/

COPY app.py .

EXPOSE 8000
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
