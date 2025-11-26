import streamlit as st
import pandas as pd
import pickle
import json
from scipy import sparse
import nltk
nltk.download('punkt')
from pathlib import Path

fitted_count = Path(__file__).parents[0] / 'models/count_vectorizer_v01.pkl'
fitted_count_matrix = Path(__file__).parents[0] / 'data/processed/count_matrix_v01.npz'
tmdb_training_data = Path(__file__).parents[0] / 'data/processed/updated_tmdb_training_data_v01.csv'
language_dictionary = Path(__file__).parents[0] / 'data/raw/iso639_language_codes.json'

from modules.similar_movies_count import similar_movies_count

st.set_page_config(
    page_title="Elevator Pitch",
    page_icon=":clapper:",
    layout="wide"
)

st.subheader("Movie Recommendations based on Your Elevator Pitch")

genre_options = ['Action', 'Adventure', 'Animation', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family',
       'Fantasy', 'History', 'Horror', 'Music',
       'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
       'Thriller', 'War', 'Western']

left_gen, gap_gen, right_gen = st.columns([5,2,6], vertical_alignment="top")
with left_gen:
    st.markdown("What are your movie's genres ?")
    given_genres = st.multiselect(
        label="", label_visibility="collapsed",
        options=genre_options,
        max_selections=3, width="stretch",
        accept_new_options=False,
        default=None,
        placeholder="Choose max. 3 genres that best represent your movie"
    )

# TF-IDF Vectorizer
with open(fitted_count, "rb") as file:
    count = pickle.load(file)

# ISO639-1 Language Codes
with open(language_dictionary, "rb") as file:
    langdict = json.load(file)

df = pd.read_csv(tmdb_training_data, parse_dates=['release_date'])
count_matrix = sparse.load_npz(fitted_count_matrix)

if 'similardf' not in st.session_state:
    st.session_state['similardf'] = pd.DataFrame()

st.markdown("\n\n")
st.markdown("What is your movie about ?")
form = st.form(key="user_form")
given_plot = form.text_area(label="", label_visibility="collapsed",
                            placeholder="Please summarize the plot of your movie in a few sentences")
submitted = form.form_submit_button("See similar movies")

if submitted:
    st.markdown("\n\n")
    st.markdown("Top 10 Most Similar Movies Based on the Summary")
    similardf = similar_movies_count(df, count, count_matrix, given_plot, given_genres, langdict)
    st.session_state["similardf"] = similardf
    st.dataframe(similardf, hide_index=True)
else:
    st.markdown("\n\n")
    st.markdown("Top 10 Most Similar Movies Based on the Summary")
    similardf = st.session_state["similardf"]
    st.dataframe(similardf, hide_index=True)
