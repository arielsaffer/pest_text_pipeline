"""
This script contains functions for text analysis data processing.
"""

## Libraries

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import tomotopy as tp
import nltk
import en_core_web_md

# Load language data products

nlp = en_core_web_md.load()
stopwords = nltk.corpus.stopwords.words('english')

### LDA functions

# Preprocess text data into list of lists of strings

def clean_tweet(text):
    """
    Clean the given text by removing URLs, mentions, and "RT ".

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.replace(r"http\S+", "")
    text = text.replace(r"@\S+", "")
    text = text.replace(r"RT ", "")
    return text


def preprocess_text(text_data, lang = "english"):
    """
    Preprocess the given text data into a list of strings.

    Args:
        data (pd.Series): A pandas Series containing text data.
        lang (str): The language of the data. Default is "english".

    Returns:
        list: A list of strings.
    """
    # Tokenize text data
    documents = text_data.str.lower().str.split()
    # Remove stopwords and punctuation
    documents = documents.apply(lambda x: [tok for tok in x if tok.isalnum() and tok not in stopwords])
    print("Text data preprocessed (tokenized, lowercased, stopwords removed).\n\n")
    return documents


def model_topics(data, num_topics = 20, num_iter = 10):
    """
    Train an LDA model on the given data.

    Args:
        data (list): A list of lists of strings, where each inner list represents a document.
        num_topics (int): The number of topics to train the model on.
        num_iter (int): The number of iterations to train the model for.
        lang (str): The language of the data. Default is "english".

    Returns:
        model: The trained LDA model.
    """
    model = tp.LDAModel(k=num_topics)
    for doc in data:
        model.add_doc(doc)
    print("Topic Model Training...\n\n")
    for i in range(0, num_iter):
        model.train(num_iter)
        print(f'Iteration: {i}\tLog-likelihood: {model.ll_per_word}')
    return model


def get_topics(model, num_words = 10):
    """
    Get the top words for each topic in the given model.

    Args:
        model: The trained LDA model.
        num_words (int): The number of top words to return for each topic.

    Returns:
        pd.DataFrame: A DataFrame containing the top words for each topic.
    """
    topic_words = []
    for k in range(model.k):
        topic_words.append(model.get_topic_words(k, top_n = num_words))

    # Keep just the words
    topic_words = [[word for word, prob in topic] for topic in topic_words]

    topic_number = [i for i in range(model.k)]

    print(f"Top {num_words} words for each topic extracted.")

    return pd.DataFrame({"Topic Number": topic_number, "Top Words": topic_words})

def text_to_topics(text_data, lang = "english", num_topics = 20, num_iter = 10):
    """
    This function takes in text data and returns the topics for each document.

    Args:
        data (pd.Series): A pandas Series containing text data.

    Returns:
        pd.DataFrame: A DataFrame containing the topics for each document.
    """
    # Preprocess text data
    data = preprocess_text(text_data, lang = lang)
    # Train LDA model
    model = model_topics(data, num_topics = num_topics, num_iter = num_iter)
    # Get the topics and their proportions
    topic_table = get_topics(model)
    return topic_table


### Keyword search functions

def keyword_search(text_data, keywords):
    """
    Search for the given keywords in the text data.

    Args:
        text_data (pd.Series): A pandas Series containing text data.
        keywords (list): A list of keywords to search for.

    Returns:
        pd.DataFrame: A DataFrame containing the text data with a column indicating whether the keywords were found.
    """
    # Create a DataFrame with the text data
    df = pd.DataFrame({"Text": text_data})
    # Search for the keywords in the text data
    df["Keywords Found"] = df["Text"].apply(lambda x: any(keyword in x for keyword in keywords))
    return df.loc[df["Keywords Found"] == True]


### Pest event functions

# Detecting reports of pest events


# Extracting locations and geoparsing

def get_loc_ents(clean_text):
    """
    Extract location entities (GPE and LOC) from the given text.

    Args:
        clean_text (str): Cleaned text to extract entities from.

    Returns:
        list: A list of location entities.
    """
    NLP = nlp(clean_text)
    EntText = [ent.text for ent in NLP.ents if ent.label_ in ["GPE","LOC"]]
    return EntText

# Function to get country, state, or city from geocoder result
# with KeyError and TypeError handling

def get_loc_level(result, level):
    """
    Get the country, state, or city from the geocoder result.

    Args:
        result (dict): The geocoder result.
        level (str): The level of location to retrieve (e.g., 'country', 'state', 'city').

    Returns:
        str or None: The location at the specified level, or None if not found.
    """
    try:
        return result[f'addr:{level}']
    except (KeyError, TypeError):
        return None

# Function to get X and Y coordinates from geocoder result
# with KeyError and TypeError handling

def get_coords(result):
    """
    Get the X and Y coordinates from the geocoder result.

    Args:
        result (dict): The geocoder result.

    Returns:
        tuple or None: The X and Y coordinates as a tuple (x, y), or None if not found.
    """
    try:
        return result['x'], result['y']
    except (KeyError, TypeError):
        return None, None
