import os
import re
import string
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Download stopwords
# -----------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "tmdb_5000_movies.csv"
    )

    df = pd.read_csv(file_path)
    df["overview"] = df["overview"].fillna("")
    df["clean_overview"] = df["overview"].apply(clean_text)

    return df


# -----------------------------
# Create TF-IDF
# -----------------------------
@st.cache_resource
def create_model(df):

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    matrix = tfidf.fit_transform(df["clean_overview"])

    return matrix


df = load_data()
tfidf_matrix = create_model(df)

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get 5 similar movie recommendations.")

movie_name = st.selectbox(
    "Choose a Movie",
    sorted(df["title"].unique())
)

if st.button("Recommend Movies"):

    movie_index = df[df["title"] == movie_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    movie_list = list(enumerate(similarity_scores))

    movie_list = sorted(
        movie_list,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Top 5 Recommended Movies")

    for movie in movie_list[1:6]:
        st.success(df.iloc[movie[0]]["title"])
