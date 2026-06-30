# ==========================================================
# Task 6: Streamlit Movie Recommendation App
# ==========================================================

# Import required libraries

import os
import re
import string
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords only if needed
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ----------------------------------------------------------
# Function to clean text
# ----------------------------------------------------------

def clean_text(text):

    if pd.isnull(text):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

file_path = os.path.join(os.path.dirname(__file__), "tmdb_5000_movies.csv")
df = pd.read_csv(file_path)

# Fill missing values
df["overview"] = df["overview"].fillna("")

# Clean text
df["clean_overview"] = df["overview"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["clean_overview"])

# Cosine Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------

st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox(
    "Select a Movie",
    sorted(df["title"].unique())
)

if st.button("Recommend Movies"):

    movie_index = df[df["title"] == movie_name].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Top 5 Recommended Movies")

    count = 0

    for movie in similarity_scores[1:]:

        index = movie[0]

        st.write("✅", df.iloc[index]["title"])

        count += 1

        if count == 5:
            break
