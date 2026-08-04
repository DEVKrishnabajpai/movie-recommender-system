import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


@st.cache_resource
def load_data_and_similarity():
    with open("moviedict.pkl", "rb") as f:
        movies_dict = pickle.load(f)
    movies = pd.DataFrame(movies_dict)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity


movies, similarity = load_data_and_similarity()


def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_indices]


st.title("🎬 Movie Recommender System")
st.write("Content-based recommendations using genres, cast, crew, and overview.")

selected_movie = st.selectbox("Pick a movie you like:", movies["title"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("You might also like:")
    for title in recommendations:
        st.write(f"- {title}")