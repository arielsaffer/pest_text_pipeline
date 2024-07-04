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
import pytesseract
from pdf2image import pdfinfo_from_path, convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Load language data products

nlp = en_core_web_md.load()
stopwords = nltk.corpus.stopwords.words('english')

### OCR functions

import pytesseract
from pdf2image import pdfinfo_from_path, convert_from_path

# Create 

def pdf_to_text_files(pdf_path, lang = "eng"):
    """
    Convert a pdf to text files for each page.
    
    Args:
    pdf_path (str): The path to the pdf file.
    lang (str): The language of the text in the pdf. Default is "eng" for English.

    Returns:
        None
    """
    print(f"Processing {pdf_path} to images...")
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
    maxPages = info["Pages"]
    print(f"Found {maxPages} pages. This document will be processed in {maxPages//10+1} batches of 10 pages.")
    for page in range(1, maxPages+1, 10) : 
        images = convert_from_path(pdf_path, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))

        print(f"Processing pages {page} to {min(page+10-1,maxPages)} of {pdf_path} to text...")
        for pageNum,image in enumerate(images):
            text = pytesseract.image_to_string(image,lang='eng')
            print(f"Converted page {page+pageNum} to text (length = {len(text)})")
            with open(f'{pdf_path[:-4]}_page{page+pageNum}.txt', 'w') as the_file:
                the_file.seek(0)
                the_file.write(text)

def clean_pdf_text_files(pdf_path, document_level = "paragraph"):
    """
    Clean the text files created from a pdf.

    Args:
    pdf_path (str): The path to the pdf file.
    document_level (str): The level of the document to clean. Options are "page", "paragraph", or "sentence". Default is "paragraph".

    Returns:
    clean_documents (pd.Series): The cleaned text documents
    """
    # Read the pages as a list of strings
    txt_files = glob.glob(f"{pdf_path[:-4]}_page*.txt")
    print(f"Found {len(txt_files)} text files for {pdf_path}.")
    pages = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as the_file:
            text = the_file.read()
            pages.append(text)
    # If document_level == "page", then clean by replacing "\n" with " " 
    if document_level == "page":
        clean_documents = [page.replace("\n", " ") for page in pages]
    if document_level == "paragraph":
        # Combine the fill document
        clean_documents = " ".join(pages)
        # Split into paragraphs as "\n\n"
        clean_documents = clean_documents.split("\n\n")
        # Remove empty paragraphs
        clean_documents = [paragraph for paragraph in clean_documents if len(paragraph) > 0]
        # Replace \n with " " in each paragraph
        clean_documents = [paragraph.replace("\n", " ") for paragraph in clean_documents]
    if document_level == "sentence":
        # Combine the fill document
        clean_documents = " ".join(pages)
        # Replace \n with " "
        clean_documents = clean_documents.replace("\n", " ")
        # Split the document into sentences
        clean_documents = clean_documents.split(". ")
        # Remove empty sentences
        clean_documents = [sentence for sentence in clean_documents if len(sentence) > 0]
    return pd.Series(clean_documents)

def pdf_to_corpus(pdf_path, document_level = "paragraph", lang = "eng"):
    """
    Convert a pdf to text documents at a specified level.

    Args:
    pdf_path (str): The path to the pdf file.
    document_level (str): The level of the document to clean. Options are "page", "paragraph", or "sentence". Default is "paragraph".
    lang (str): The language of the text in the pdf. Default is "eng" for English.

    Returns:
    clean_documents (pd.Series): The cleaned text documents
    """
    pdf_to_text_files(pdf_path, lang)
    return clean_pdf_text_files(pdf_path, document_level)

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
