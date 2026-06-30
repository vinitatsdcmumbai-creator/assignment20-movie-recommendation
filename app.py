# ==========================================================
# Task 6: Streamlit Movie Recommendation App
# ==========================================================

# Import required libraries

import streamlit as st
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Function to clean text
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

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Clean overview column
df["clean_overview"] = df["overview"].apply(clean_text)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["clean_overview"])

# Calculate similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Streamlit title
st.title("Movie Recommendation System")

# Movie selection
movie_name = st.selectbox("Select a Movie", df["title"])

# Recommendation button
if st.button("Recommend Movies"):

    movie_index = df[df["title"] == movie_name].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Top 5 Recommended Movies")

    count = 0

    for movie in sorted_movies[1:]:

        index = movie[0]

        st.write(df.iloc[index]["title"])

        count += 1

        if count == 5:
            break