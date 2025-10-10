import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def read_data():
    df = pd.read_csv("/Users/arnavgupta/Downloads/sadnesspython/chatbot/tmdb_5000_movies.csv")
    return df
def weighted_rating(x,m,C):
    v=x['vote_count']
    R=x['vote_average']
    return(v/(v+m)*R)+(m/(v+m)*C)
def calculated_weighted_rating(df):
    C=df['vote_average'].mean()
    m=df['vote_count'].quantile(0.9)
    q_movies = df.copy().loc[df['vote_count']>m]
    q_movies['score']=q_movies.apply(weighted_rating, axis=1, m = m, C=C)
    return(q_movies)
def get_top5_movies_by_WrScore(q_movies):
    return q_movies[['title', 'score']].sort_values('score', ascending = False).head(5)
df = read_data()
def calculate_simscores(df):
    df['overview']=df['overview'].fillna('')
    vectorizer = TfidfVectorizer(stop_words = "english")
    tfidf = vectorizer.fit_transform(df['overview'])
    sim = cosine_similarity(tfidf, tfidf)
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()
    return sim,indices
def get_top5_movies_by_popularity(df):
    return df[['title','popularity']].sort_values('popularity', ascending=False).head(5)
def get_personalized_recommendations(title):
    sim,indices = calculate_simscores(df)
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sort_sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    sort_sim_scores = sort_sim_scores[1:6]
    movie_indices = []
    for i in sort_sim_scores:
        movie_indices.append(i[0])
    return df['title'].iloc[movie_indices]
get_personalized_recommendations("Pirates of the Caribbean: At World's End")
q_movies=calculated_weighted_rating(df)
menu = st.sidebar.radio("Choose a recommender type", ["Top 5 movies by weighted score", "Top 5 popular movies", "Get Personalized Recommendation"])
if menu=="Top 5 movies by weighted score":
    
    st.title("Top 5 movies based on Weighted rating score")
    wr = get_top5_movies_by_WrScore(q_movies)
    st.dataframe(wr)
    plt.figure(figsize=(10, 6))
    plt.barh(wr['title'], wr['score'])
    plt.gca().invert_yaxis()
    plt.title("Top 5 movies based on weighted score")
    plt.xlabel("Weighted score")
    plt.ylabel("Movie title")
    st.pyplot(plt)
elif menu == "Top 5 popular movies":
    st.title("Top 5 movies based on popularity")
    pop = get_top5_movies_by_popularity(df.copy())
    st.dataframe(pop)
    plt.figure(figsize=(10, 6))
    plt.barh(pop['title'], pop['popularity'])
    plt.gca().invert_yaxis()
    plt.title("Top 5 movies based on popularity")
    plt.xlabel("Popularity score")
    plt.ylabel("Movie title")
    st.pyplot(plt)
else:
    st.title("Top 5 movies similar to your favorite movie")
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
    st.dataframe(rec)