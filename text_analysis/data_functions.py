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
import pytesseract
import glob
import os
import spacy
from pdf2image import pdfinfo_from_path, convert_from_path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import time


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Load language data products
nltk.download('stopwords')
nlp = en_core_web_md.load()
lang_map = pd.read_csv("text_analysis/language_name_map.csv")

### Helper functions

def map_language(lang_name):
    """
    Map a language name to the corresponding OCR language code, NLTK language name, and spaCy language code.

    Args:
        lang_name (str): The name of the language.

    Returns:
        str: The language code.
        str: The NLTK language name.
        str: The spaCy language code.
    """
    ocr_code = lang_map.loc[lang_map["OCR_LanguageName"] == lang_name, "OCR_LangCode"].values[0]
    try:
        nltk_lang = lang_map.loc[lang_map["OCR_LanguageName"] == lang_name, "NLTK_Language"].values[0]
    except IndexError:
        # If there's no nltk language, removing stopwords will be skipped
        nltk_lang = None
    try: 
        spacy_code = lang_map.loc[lang_map["OCR_LanguageName"] == lang_name, "Spacy_LangCode"].values[0]
    except IndexError:
        # If there's no spaCy code, use the multi-language pipeline
        spacy_code = "xx"
    return ocr_code, nltk_lang, spacy_code
        

### OCR functions

import pytesseract
from pdf2image import pdfinfo_from_path, convert_from_path

# Function to convert a pdf to text files for each page

def pdf_to_text_files(pdf_path, lang = "eng"):
    """
    Convert a pdf to text files for each page.
    
    Args:
    pdf_path (str): The path to the pdf file.
    lang (str): The language of the text in the pdf. Default is "eng" for English.

    Returns:
        None
    """
    root_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path)

    # Check for and create subfolder for text files
    if not os.path.exists(f"{root_dir}/{file_name[:-4]}_text_files"):
        os.makedirs(f"{root_dir}/{file_name[:-4]}_text_files")

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
            with open(f'{root_dir}/{file_name[:-4]}_text_files/{file_name[:-4]}_page{page+pageNum}.txt', 'w') as the_file:
                the_file.seek(0)
                the_file.write(text)

    print(f"Text files created for {pdf_path}.")
    return None

# Function to clean the text files created from a pdf

def clean_pdf_text_files(pdf_path, document_level = "paragraph"):
    """
    Clean the text files created from a pdf.

    Args:
    pdf_path (str): The path to the pdf file.
    document_level (str): The level of the document to clean. Options are "page", "paragraph", or "sentence". Default is "paragraph".

    Returns:
    clean_documents (pd.Series): The cleaned text documents
    """

    root_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path)
    
    # Read the pages as a list of strings
    txt_files = glob.glob(f'{root_dir}/{file_name[:-4]}_text_files/{file_name[:-4]}_page*.txt')
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
    clean_documents = pd.Series(clean_documents)
    clean_documents.name = "Text"    
    
    return clean_documents

# Wrap both functions

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


def preprocess_text(text_data, lang = None):
    """
    Preprocess the given text data into a list of strings.

    Args:
        data (pd.Series): A pandas Series containing text data.
        lang (str): The language of the data used to remove stopwords.

    Returns:
        list: A list of strings.
    """
    # Tokenize text data
    documents = text_data.str.lower().str.split()
    # Remove stopwords and punctuation
    try:
        stopwords = nltk.corpus.stopwords.words(lang)
        documents = documents.apply(lambda x: [tok for tok in x if tok.isalnum() and tok not in stopwords])
        print("Text data preprocessed (tokenized, lowercased, stopwords removed).\n\n")
    except:
        print(f"Stopwords not found for language '{lang}'. Text preprocessed (tokenized and lowercased).\n\n")
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

def text_to_topics(text_data, lang = None, num_topics = 20, num_iter = 10):
    """
    This function takes in text data and returns the topics for each document.

    Args:
        data (pd.Series): A pandas Series containing text data.
        lang (str): The language of the data used to remove stopwowrds.
        num_topics (int): The number of topics to train the model on. Default is 20.
        num_iter (int): The number of iterations to train the model for. Default is 10.

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


#### Pest event functions

### Classification

# Detecting reports of pest events

def train_model(data, text_col="Text", label_col="Target", vectorizer=TfidfVectorizer(), classifier=LinearSVC()):
    """
    Train a classification model on the given data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the text data and labels.
        text_col (str): The name of the column containing the text data. Default is "Text".
        label_col (str): The name of the column containing the labels. Default is "Target".
        vectorizer: The vectorizer to convert text data into vectors. Default is TfidfVectorizer().
        clf: The classification model to train. Default is LinearSVC().

    Returns:
        tuple: A tuple containing the trained model and the vectorizer.
    """
    # 

    # Split the data into text and labels
    X = data[text_col]
    y = data[label_col]
    # Convert text data into vectors
    vects = vectorizer.fit_transform(X)
    # Train the classification model
    classifier.fit(vects, y)
    return classifier, vectorizer


# Define a function to produce classification metrics
def test_model(X, y, classifier, vectorizer):
    """
    Test a classification model on the given data.

    Args:
        X (pd.Series): A pandas Series containing the text data.
        y (pd.Series): A pandas Series containing the labels.
        clf: The classification model to test.
        vectorizer: The vectorizer used to convert text data into vectors.

    Returns:
        tuple: A tuple containing the predictions and a dictionary of classification metrics.
    """
   # test model and calculate accuracy
    vects_test = vectorizer.transform(X) 
    # convert test docs to vectors based on what we learned from the training data
    pred = classifier.predict(vects_test)

    # calculate accuracy, precision, recall, and fscore based on pred output and ground truth labels (y_test)
    accuracy = accuracy_score(y, pred)
    prf = precision_recall_fscore_support(y, pred)
    precision = prf[0][1]
    recall = prf[1][1]
    fscore = prf[2][1]
    
    return pred, {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall, 'fscore' : fscore}

# Run train and test process over k-split data for each model 

def test_multiple_models(X, y, models, vectorizer, selection_metric = "fscore", k = 10, random_state = 42, verbose = False):
    """
    Test multiple models using k-fold cross-validation.

    Args:
        X (pd.Series): A pandas Series containing the text data.
        y (pd.Series): A pandas Series containing the labels.
        models (list): A list of sklearn classification models to test.
        vectorizer: The vectorizer used to convert text data into vectors.
        k: The number of folds for k-fold cross-validation. Default is 10.
        random_state: The random state for k-fold cross-validation. Default is 42.

    Returns:
        pd.DataFrame: A DataFrame containing results of model testing.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    results_table = pd.DataFrame(
    columns=[
        'model', 
        'accuracy', 'accuracy_sd', 
        'precision', 'precision_sd',
        'recall', 'recall_sd',
        'fscore', 'fscore_sd',
        ])

    
    for model in models:

        clf = model
        acc_score = []
        prec_score = []
        rec_score = []
        f_score = []

        # this splits your data into 10 folds
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model 
            vectorizer = TfidfVectorizer(stop_words='english', min_df=0.001, ngram_range=(1, 2))
            # Transform training data into vectors
            vects_train = vectorizer.fit_transform(X_train)
            # Train the model
            clf.fit(vects_train, y_train)

            # Test model and calculate metrics
            pred, scores_dict = test_model(X_test, y_test, clf, vectorizer)
            
            acc = scores_dict['accuracy']
            prec = scores_dict['precision']
            rec = scores_dict['recall']
            f = scores_dict['fscore']

            acc_score.append(acc)
            prec_score.append(prec)
            rec_score.append(rec)
            f_score.append(f)

        # Compute average and standard deviation of scores across folds
            
        avg_acc_score = sum(acc_score)/k
        avg_prec_score = sum(prec_score)/k
        avg_rec_score = sum(rec_score)/k
        avg_f_score = sum(f_score)/k

        sd_acc_score = np.std(acc_score)
        sd_prec_score = np.std(prec_score)
        sd_rec_score = np.std(rec_score)
        sd_f_score = np.std(f_score)

        results_table = pd.concat([results_table, pd.DataFrame({
            'model' : model,
            'accuracy' : avg_acc_score,
            'accuracy_sd' : sd_acc_score,
            'precision' : avg_prec_score,
            'precision_sd' : sd_prec_score,
            'recall' : avg_rec_score,
            'recall_sd' : sd_rec_score,
            'fscore' : avg_f_score,
            'fscore_sd' : sd_f_score
            }, index=[0]),], ignore_index=True) 

        # Sort the result table by the selection metric
        results_table = results_table.sort_values(by=selection_metric, ascending=False)

        if verbose == True:
            # Print results summary for each model

            print('Model : {}'.format(model))
            
            print('accuracy of each fold - {}'.format(acc_score))
            print('Avg accuracy : {}, +/- {}'.format(avg_acc_score, sd_acc_score))
            print('\n')

            print('precision of each fold - {}'.format(prec_score))
            print('Avg precision : {} +/- {}'.format(avg_prec_score, sd_prec_score))
            print('\n')

            print('recall of each fold - {}'.format(rec_score))
            print('Avg recall : {} +/- {}'.format(avg_rec_score, sd_rec_score))
            print('\n')

            print('fscore of each fold - {}'.format(f_score))
            print('Avg fscore : {} +/- {}'.format(avg_f_score, sd_f_score))
            print('\n')

            print('--------------------------------------------------------')
    return results_table


### Extracting locations and geoparsing

# Create the spaCy nlp object based on the language

def create_lang_nlp(spacy_lang_code):
    """
    Create a spaCy nlp object based on the language.

    Args:
        lang (str): The language of the nlp object.

    Returns:
        spacy.lang: The spaCy nlp object.
    """
    # Maybe there is a cleaner way to do this?
    if lang == "ca":
    try:
        nlp = ca_core_news_md.load()
    except OSError:
        spacy.cli.download("ca_core_news_md")
        nlp = ca_core_news_md.load()

    elif lang == "zh":
        try:
            nlp = zh_core_news_md.load()
        except OSError:
            spacy.cli.download("zh_core_news_md")
            nlp = zh_core_news_md.load()

    elif lang == "da":
        try:
            nlp = da_core_news_md.load()
        except OSError:
            spacy.cli.download("da_core_news_md")
            nlp = da_core_news_md.load()

    elif lang == "el":
        try:
            nlp = el_core_news_md.load()
        except OSError:
            spacy.cli.download("el_core_news_md")
            nlp = el_core_news_md.load()

    elif lang == "en":
        try:
            nlp = en_core_news_md.load()
        except OSError:
            spacy.cli.download("en_core_news_md")
            nlp = en_core_news_md.load()

    elif lang == "fi":
        try:
            nlp = fi_core_news_md.load()
        except OSError:
            spacy.cli.download("fi_core_news_md")
            nlp = fi_core_news_md.load()

    elif lang == "fr":
        try:
            nlp = fr_core_news_md.load()
        except OSError:
            spacy.cli.download("fr_core_news_md")
            nlp = fr_core_news_md.load()

    elif lang == "de":
        try:
            nlp = de_core_news_md.load()
        except OSError:
            spacy.cli.download("de_core_news_md")
            nlp = de_core_news_md.load()

    elif lang == "hr":
        try:
            nlp = hr_core_news_md.load()
        except OSError:
            spacy.cli.download("hr_core_news_md")
            nlp = hr_core_news_md.load()

    elif lang == "it":
        try:
            nlp = it_core_news_md.load()
        except OSError:
            spacy.cli.download("it_core_news_md")
            nlp = it_core_news_md.load()

    elif lang == "ja":
        try:
            nlp = ja_core_news_md.load()
        except OSError:
            spacy.cli.download("ja_core_news_md")
            nlp = ja_core_news_md.load()

    elif lang == "ko":
        try:
            nlp = ko_core_news_md.load()
        except OSError:
            spacy.cli.download("ko_core_news_md")
            nlp = ko_core_news_md.load()

    elif lang == "lt":
        try:
            nlp = lt_core_news_md.load()
        except OSError:
            spacy.cli.download("lt_core_news_md")
            nlp = lt_core_news_md.load()

    elif lang == "mk":
        try:
            nlp = mk_core_news_md.load()
        except OSError:
            spacy.cli.download("mk_core_news_md")
            nlp = mk_core_news_md.load()

    elif lang == "nl":
        try:
            nlp = nl_core_news_md.load()
        except OSError:
            spacy.cli.download("nl_core_news_md")
            nlp = nl_core_news_md.load()

    elif lang == "nb":
        try:
            nlp = nb_core_news_md.load()
        except OSError:
            spacy.cli.download("nb_core_news_md")
            nlp = nb_core_news_md.load()

    elif lang == "pl":
        try:
            nlp = pl_core_news_md.load()
        except OSError:
            spacy.cli.download("pl_core_news_md")
            nlp = pl_core_news_md.load()

    elif lang == "pt":
        try:
            nlp = pt_core_news_md.load()
        except OSError:
            spacy.cli.download("pt_core_news_md")
            nlp = pt_core_news_md.load()

    elif lang == "ro":
        try:
            nlp = ro_core_news_md.load()
        except OSError:
            spacy.cli.download("ro_core_news_md")
            nlp = ro_core_news_md.load()

    elif lang == "ru":
        try:
            nlp = ru_core_news_md.load()
        except OSError:
            spacy.cli.download("ru_core_news_md")
            nlp = ru_core_news_md.load()

    elif lang == "sl":
        try:
            nlp = sl_core_news_md.load()
        except OSError:
            spacy.cli.download("sl_core_news_md")
            nlp = sl_core_news_md.load()

    elif lang == "es":
        try:
            nlp = es_core_news_md.load()
        except OSError:
            spacy.cli.download("es_core_news_md")
            nlp = es_core_news_md.load()

    elif lang == "sv":
        try:
            nlp = sv_core_news_md.load()
        except OSError:
            spacy.cli.download("sv_core_news_md")
            nlp = sv_core_news_md.load()

    elif lang == "uk":
        try:
            nlp = uk_core_news_md.load()
        except OSError:
            spacy.cli.download("uk_core_news_md")
            nlp = uk_core_news_md.load()

    else:
        try:
            nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            spacy.cli.download("xx_ent_wiki_sm")
            nlp = spacy.load("xx_ent_wiki_sm")

    return nlp

# Extract location entities from text

def get_loc_ents(text, origin = np.nan):
    """
    Given a post, return a list of unique locations mentioned in the post.
    This includes: unique entities, entities neighboring entity context, 
    and entities with post origin context.

    Args:
        text (str): The text of the post.
        post_origin (str): The country of origin of the post, optional.

    Returns:
        list: A list of unique locations mentioned in the post.
    """
    # Create NLP object
    nlp = en_core_web_md.load()
    doc = nlp(text)
    # Get entities
    ents = [ent.text for ent in doc.ents if ent.label_ in ["GPE","LOC"]]
    # Get entity spans
    spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in ["GPE","LOC"]]
    # Get the difference between the start of one entity and the end of the previous entity
    spans_diff = [spans[i][0] - spans[i-1][1] for i in range(1,len(spans))]
    # If EntSpan is less than 5, concatenate the two entities in EntText
    ents_multi = [ents[i-1] + ", " + ents[i] for i in range(1,len(ents)) if spans_diff[i-1] < 5]
    # Keep only unique entities in a single post
    ents_unique = list(set(ents))
    # Concat EntUnique and Country into a new column
    if origin == origin:
       ents_country = [ent + ", " + origin for ent in ents_unique]
    else:
        ents_country = []
    return ents_unique, ents_multi, ents_country

# Create dictionary of geocoded locations

def geocode_locs(locs, geocoder, key):
    """
    Geocode a list of locations using a geocoder.
    
    Args:
        locs (list): A list of locations to geocode.
        geocoder: The geocoder to use.
        key: The API key for the geocoder.

    Returns:
        dict: A dictionary of geocoded locations.
    """
    geo_loc_dict = {}
    error_locs = []
    i = 1

    for loc in locs:
        if loc == "":
            continue
        else:
            result = geocoder.osm(loc, key)
            geo_loc_dict[loc] = result
            # Catch 502 errors and wait 5 seconds before trying again
            if "ERROR" in result.status:
                time.sleep(5)
                result = geocoder.osm(loc, key)
                geo_loc_dict[loc] = result
                if "ERROR" in result.status:
                    print(f"{i}. '{loc}' not found")
                    error_locs.append(loc)
                    i += 1
                    continue
    return geo_loc_dict, error_locs

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
