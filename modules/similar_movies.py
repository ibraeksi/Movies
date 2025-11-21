import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

def similar_movies(df, tfidf, tfidf_matrix, new_plot, langdict):
    """
    Using cosine similarity to find the top 10 most similar movies to the user input
    df = tmdb dataset used in training the deep learning model
    tfidf = Fitted TF-IDF Vectorizer
    tfidf_matrix = Fitted TF-IDF matrix to the dataset
    new_plot = User input for movie plot
    """
    # Add the new plot to TF-IDF matrix
    new_tfidf_matrix = sparse.vstack((tfidf_matrix, tfidf.transform(pd.DataFrame({'overview':[new_plot]})['overview'])))

    # Compute the updated cosine similarity matrix
    cosine_sim = linear_kernel(new_tfidf_matrix, new_tfidf_matrix)

    # Get the pairwsie similarity scores of all movies with the added plot
    sim_scores = list(enumerate(cosine_sim[-1]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    df['release_date'] = df['release_date'].dt.strftime('%d.%m.%Y')
    df['original_language'] = df['original_language'].map(langdict)

    # Columns to be displayed
    cols = ['title', 'score', 'vote_count', 'release_date', 'revenue', 'runtime',
            'budget', 'original_language', 'original_title', 'overview',
            'genres', 'production_countries', 'spoken_languages']
    # Return the top 10 most similar movies without the added plot
    return df[cols].iloc[movie_indices[1:]]
