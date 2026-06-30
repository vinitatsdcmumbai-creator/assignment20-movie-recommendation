import os
import re
import string

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get 5 similar movie recommendations.")


# -------------------------------
# Clean Text Function
# -------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------
# Load Dataset
# -------------------------------
file_path = os.path.join(os.path.dirname(__file__), "tmdb_5000_movies.csv")
df = pd.read_csv(file_path)

df["overview"] = df["overview"].fillna("")
df["clean_overview"] = df["overview"].apply(clean_text)


# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000
)

tfidf_matrix = vectorizer.fit_transform(df["clean_overview"])

similarity = cosine_similarity(tfidf_matrix)


# -------------------------------
# Movie Selection
# -------------------------------
movie_list = sorted(df["title"].unique())

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)


# -------------------------------
# Recommendation
# -------------------------------
if st.button("Recommend Movies"):

    movie_index = df[df["title"] == selected_movie].index[0]

    distances = list(enumerate(similarity[movie_index]))

    movies = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Top 5 Recommended Movies")

    count = 0

    for movie in movies[1:]:

        index = movie[0]

        st.write("✅", df.iloc[index]["title"])

        count += 1

        if count == 5:
            break
