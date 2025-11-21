import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import sparse
from pathlib import Path

trained_model = Path(__file__).parents[0] / 'models/weighted_rating_nlp_runtime_3genres_v01.pkl'
fitted_tokenizer = Path(__file__).parents[0] / 'models/weighted_rating_tokenizer_v01.pkl'
fitted_encoder = Path(__file__).parents[0] / 'models/weighted_rating_3genres_encoder_v01.pkl'
fitted_scaler = Path(__file__).parents[0] / 'models/weighted_rating_3genres_scaler_v01.pkl'
fitted_tfidf = Path(__file__).parents[0] / 'models/weighted_rating_tfidf_vectorizer_v01.pkl'
fitted_tfidf_matrix = Path(__file__).parents[0] / 'data/processed/tfidf_matrix_v01.npz'
tmdb_training_data = Path(__file__).parents[0] / 'data/processed/tmdb_training_data_v01.csv'
language_dictionary = Path(__file__).parents[0] / 'data/raw/iso639_language_codes.json'

from modules.clean_sentences import clean_sentences
from modules.similar_movies import similar_movies

st.set_page_config(
    page_title="Elevator Pitch",
    page_icon=":clapper:",
    layout="wide"
)

st.subheader("Movie Review Score Predictions")

st.markdown("What is your movie about ?")
form = st.form(key="user_form")
given_plot = form.text_area(label="", label_visibility="collapsed",
                            placeholder="Please summarize the plot of your movie in a few sentences")
submitted = form.form_submit_button("See similar movies")

genre_options = ['Action', 'Adventure', 'Animation', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family',
       'Fantasy', 'History', 'Horror', 'Music',
       'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
       'Thriller', 'War', 'Western']

st.markdown("\n\n")
st.markdown("\n\n")
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
    st.markdown("How long is your movie ?")
    left_num, gap_num, right_num = st.columns([2, 1, 2], vertical_alignment="top")
    with left_num:
        given_runtime_hours = st.number_input(label="Runtime Hours",
                                        min_value=0, max_value=5, step=1)
    with right_num:
        given_runtime_mins = st.number_input(label="Runtime Minutes",
                                        min_value=0, max_value=60, step=1)

# NLP + Numerical Features Combined Model
with open(trained_model, "rb") as model_file:
    model = pickle.load(model_file)

# Tokenizer
with open(fitted_tokenizer, "rb") as file:
    tk = pickle.load(file)

# OneHotEncoder
with open(fitted_encoder, "rb") as file:
    ohe = pickle.load(file)

# StandardScaler
with open(fitted_scaler, "rb") as file:
    sc = pickle.load(file)

# TF-IDF Vectorizer
with open(fitted_tfidf, "rb") as file:
    tfidf = pickle.load(file)

# ISO639-1 Language Codes
with open(language_dictionary, "rb") as file:
    langdict = json.load(file)

df = pd.read_csv(tmdb_training_data, parse_dates=['release_date'])
tfidf_matrix = sparse.load_npz(fitted_tfidf_matrix)

if 'similardf' not in st.session_state:
    st.session_state['similardf'] = pd.DataFrame()

if submitted:
    st.markdown("\n\n")
    st.markdown("Top 10 Most Similar Movies Based on the Summary")
    similardf = similar_movies(df, tfidf, tfidf_matrix, given_plot, langdict)
    st.session_state["similardf"] = similardf
    st.dataframe(similardf, hide_index=True)
else:
    st.markdown("\n\n")
    st.markdown("Top 10 Most Similar Movies Based on the Summary")
    similardf = st.session_state["similardf"]
    st.dataframe(similardf, hide_index=True)

# max_length_padding
maxlen = 80
# embedding size
embedding_size = 40

if len(given_plot) > 0 and len(given_genres) > 0:
    given_runtime = given_runtime_hours*60 + given_runtime_mins
    if given_runtime > 0:
        with right_gen:
            for i in range(3-len(given_genres)):
                given_genres.append('No')
            clean_plot = clean_sentences(given_plot)
            X_pred_tokens = tk.texts_to_sequences(np.array([clean_plot]))
            X_pred_text_pad = pad_sequences(X_pred_tokens, dtype='float32', padding='post', maxlen=maxlen)

            genresdf = pd.DataFrame({'g1':[given_genres[0]], 'g2':[given_genres[1]], 'g3':[given_genres[2]]})

            X_num_pred = np.hstack([np.array([[given_runtime]]), ohe.transform(genresdf)])
            X_num_pred = sc.transform(X_num_pred)

            prediction = model.predict(x=[X_pred_text_pad, pd.DataFrame(X_num_pred)])

            st.markdown("Predicted Movie Review Score")
            left_sl, slider_sl, right_sl, gap_sl = st.columns([0.25, 9, 0.25, 0.5], vertical_alignment="top")
            with left_sl:
                st.markdown("0")
            with slider_sl:
                st.slider(label="", label_visibility="collapsed", min_value=0.0, max_value=10.0,
                      value=prediction[0][0], format="%0.2f", disabled=True)
            with right_sl:
                st.markdown("10")

            left_score, right_score = st.columns([5, 5], vertical_alignment="top")
            with left_score:
                st.markdown("\n\n")
                st.markdown("""Score = IMDB's weighted rating""")
            with right_score:
                st.latex(r'''\left(\frac{v}{v+m}.R\right) + \left(\frac{m}{v+m}.C\right)''')

            st.markdown("""v is number of votes / m is min. number of votes for training data""")
            st.markdown("""R is avg. rating / C is avg. rating of all in the training data""")
