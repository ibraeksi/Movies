import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from modules.clean_sentences import clean_sentences


def similar_movies_count(df, count, count_matrix, new_plot, new_genres, langdict):
    """
    Using cosine similarity to find the top 10 most similar movies to the user input
    df = tmdb dataset used in training the deep learning model
    count = Fitted Count Vectorizer
    count_matrix = Fitted Count matrix to the dataset
    new_plot = User input for movie plot
    new_genres = User input for movie genres
    """
    clean_plot = clean_sentences(new_plot)
    user_input = clean_plot.split(" ")
    genre_input = []
    for genre in new_genres:
        if genre != 'No':
            genre_input.append(genre.lower().replace(" ", ""))
    user_input += genre_input

    # Add the new plot to count matrix
    new_count_matrix = sparse.vstack((count_matrix, count.transform(pd.DataFrame({'soup':[' '.join(user_input)]})['soup'])))

    # Compute the updated cosine similarity matrix
    cosine_sim = cosine_similarity(new_count_matrix, new_count_matrix)

    # Similarity scores with the added plot
    sim_scores = list(enumerate(cosine_sim[-1]))

    # The 10 most similar movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    df['release_date'] = df['release_date'].dt.strftime('%d.%m.%Y')
    df['original_language'] = df['original_language'].map(langdict)

    # Columns to be displayed
    cols = ['title', 'vote_average', 'vote_count', 'release_date', 'revenue', 'runtime',
            'budget', 'original_language', 'original_title', 'overview',
            'genres', 'production_countries', 'spoken_languages']
    output = df[cols].iloc[movie_indices[1:]].reset_index(drop=True)

    return output
