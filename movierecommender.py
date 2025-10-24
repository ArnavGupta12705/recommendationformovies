import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# =========================
# TMDB API Setup
# =========================
api_key = "f1637e77b7ca1ee7c4ce0db6268be28e"  # Get API key from https://www.themoviedb.org/
poster_url_base = "https://image.tmdb.org/t/p/w500/"

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"{poster_url_base}{poster_path}"
    return None


# =========================
# Data Loading and Scoring
# =========================
def read_data():
    df = pd.read_csv(r"tmdb_5000_movies.csv")
    return df

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

def calculate_weighted_rating(df):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.9)
    q_movies = df.copy().loc[df['vote_count'] > m]
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, m=m, C=C)
    return q_movies

def get_top5_movies_by_WRScore(q_movies):
    return q_movies[['title', 'score']].sort_values('score', ascending=False).head(5)

def get_top5_movies_by_popularity(df):
    return df[['title', 'popularity']].sort_values('popularity', ascending=False).head(5)


# =========================
# Similarity Based Recommendation
# =========================
def calculate_simscores(df):
    df['overview'] = df['overview'].fillna('')
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df['overview'])
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return sim, indices

def get_personalized_recommendations(title):
    sim, indices = calculate_simscores(df)
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sort_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sort_sim_scores = sort_sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sort_sim_scores]
    return df['title'].iloc[movie_indices]


# =========================
# Streamlit UI
# =========================
df = read_data()
q_movies = calculate_weighted_rating(df)

menu = st.sidebar.radio(
    "Choose a Recommender Type",
    ["Top 5 Movies by Weighted Score", "Top 5 Popular Movies", "Get Personalized Recommendation"]
)

# --------------------------
# 1. Weighted Score Recommender
# --------------------------
if menu == "Top 5 Movies by Weighted Score":
    st.title("🎬 Top 5 Movies by Weighted Rating Score")
    wr = get_top5_movies_by_WRScore(q_movies)
    st.dataframe(wr)

    st.write("### Movie Posters:")
    cols = st.columns(5)
    for idx, row in enumerate(wr.itertuples()):
        movie_id = df.loc[df['title'] == row.title, 'id'].values[0]
        poster = fetch_poster(movie_id)
        with cols[idx % 5]:
            if poster:
                st.image(poster, width=150, caption=row.title)
            else:
                st.write(row.title)

    plt.figure(figsize=(8, 5))
    plt.barh(wr['title'], wr['score'])
    plt.gca().invert_yaxis()
    plt.title("Top 5 Movies by Weighted Score")
    plt.xlabel("Weighted Score")
    plt.ylabel("Movie Title")
    st.pyplot(plt)

# --------------------------
# 2. Popularity-Based Recommender
# --------------------------
elif menu == "Top 5 Popular Movies":
    st.title("🔥 Top 5 Movies by Popularity")
    pop = get_top5_movies_by_popularity(df.copy())
    st.dataframe(pop)

    st.write("### Movie Posters:")
    cols = st.columns(5)
    for idx, row in enumerate(pop.itertuples()):
        movie_id = df.loc[df['title'] == row.title, 'id'].values[0]
        poster = fetch_poster(movie_id)
        with cols[idx % 5]:
            if poster:
                st.image(poster, width=150, caption=row.title)
            else:
                st.write(row.title)

    plt.figure(figsize=(8, 5))
    plt.barh(pop['title'], pop['popularity'])
    plt.gca().invert_yaxis()
    plt.title("Top 5 Movies by Popularity")
    plt.xlabel("Popularity Score")
    plt.ylabel("Movie Title")
    st.pyplot(plt)

# --------------------------
# 3. Personalized Recommender
# --------------------------
else:
    st.title("🎯 Movies Similar to Your Favorite One")
    selected_movies = [
        "Harry Potter and the Chamber of Secrets",
        "Harry Potter and the Philosopher's Stone",
        "The Hobbit: The Desolation of Smaug", "Avatar", "Spider-Man 3",
        "Avengers: Age of Ultron", "Iron Man", "Iron Man 2",
        "X-Men: The Last Stand", "Star Trek Beyond", "The Fast and the Furious",
        "How to Train Your Dragon", "Mission: Impossible - Rogue Nation",
        "Minions"
    ]
    fav_movie = st.selectbox("Select a movie", selected_movies)
    rec = get_personalized_recommendations(fav_movie)
    st.write("### Top 5 Recommended Movies:")

    cols = st.columns(5)
    for idx, title in enumerate(rec):
        movie_id = df.loc[df['title'] == title, 'id'].values[0]
        poster = fetch_poster(movie_id)
        with cols[idx % 5]:
            if poster:
                st.image(poster, width=150, caption=title)
            else:
                st.write(title)
