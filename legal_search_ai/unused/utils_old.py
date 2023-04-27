import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
<<<<<<< HEAD:backend/utils_old.py
import re
=======
from gensim.models import Word2Vec
import re
import csv
from textblob import TextBlob
>>>>>>> main:legal_search_ai/unused/utils_old.py
from unicodedata import normalize
from gensim import corpora, models
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import euclidean_distances
import os
from rank_bm25 import BM25Okapi
<<<<<<< HEAD:backend/utils_old.py
=======
<<<<<<<< HEAD:backend/utils.py
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
========
from sklearn.feature_extraction.text import TfidfVectorizer
>>>>>>> main:legal_search_ai/unused/utils_old.py


# Download the stopwords resource
nltk.download('stopwords')
<<<<<<< HEAD:backend/utils_old.py


def load_preprocessed_text_data(text_data):
    # Normalization
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove special characters and digits
=======
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def compute_query_tfidf_matrix(preprocessed_query):
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Use transform to convert queries into TF-IDF feature vectors
    query_tfidf_matrix = vectorizer.fit_transform([preprocessed_query])

    return query_tfidf_matrix


def store_tokenized_docs(documents):
    tokenized_docs = [word_tokenize(text=doc.lower()) for doc in documents]

    with open('data/tokenized_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(tokenized_docs)

    return tokenized_docs


def load_tokenized_docs():
    with open('data/tokenized_docs.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tokenized_docs = list(reader)
    return tokenized_docs


def load_corpus(tokenized_docs):
    # Create a dictionary of unique words from the tokenized documents
    dictionary = corpora.Dictionary(tokenized_docs)

    # Convert the tokenized documents into "bag of words" representation
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    return corpus

def store_word2vev_model(corpus):
    # Train a Word2Vec model on your text data
    embedding_size = 100
    model = Word2Vec(corpus, size=embedding_size, window=5, min_count=5, workers=4)

    # Save the trained model to a file
    model.save("data/word2vec_model.bin")
    return model


def load_word2vev_model():
    # Load the pre-trained Word2Vec model from a file
    model = Word2Vec.load("data/word2vec_model.bin")
    return model


def generate_embeddings(word2vev_model, normalized_query, documents):
    model = word2vev_model
    # Generate embeddings for the normalized query
    query_embedding = []
    for word in normalized_query.split():
        if word in model.wv:
            query_embedding.append(model.wv[word])

    # Generate embeddings for the documents in the dataset
    document_embeddings = []
    for document in documents:
        document_embedding = []
        for word in document.split():
            if word in model.wv:
                document_embedding.append(model.wv[word])
        document_embeddings.append(document_embedding)
    
    return query_embedding, document_embeddings


def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove redundant spaces and tabs
    text = re.sub(r'\s+', ' ', text).strip()

    # Handle special cases - replace numbers with "#NUM"
    text = re.sub(r'\d+', '#NUM', text)

    # Handle special cases - replace URLs with "#URL"
    text = re.sub(r'http\S+', '#URL', text)

    # Handle special cases - replace dates with "#DATE"
    date_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(\d{4}[./-]\d{1,2}[./-]\d{1,2})'
    text = re.sub(date_pattern, '#DATE', text)

    # Perform spell checking and correction
    blob = TextBlob(text)
    text = ' '.join(blob.correct() for blob in blob.words)

    return text


>>>>>>>> main:legal_search_ai/unused/utils_old.py


def load_raw_case_docs():
    # Set the files_directory variable to the current directory
    files_directory = "case_docs"

    # Read case details from text files and store in a list
    case_documents = []
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(files_directory, file_name), 'r') as f:
                content = f.read()
                case_documents.append(content)
    return case_documents


def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove redundant spaces and tabs
    text = re.sub(r'\s+', ' ', text).strip()

    # Handle special cases - replace numbers with "#NUM"
    text = re.sub(r'\d+', '#NUM', text)

    # Handle special cases - replace URLs with "#URL"
    text = re.sub(r'http\S+', '#URL', text)

    # Handle special cases - replace dates with "#DATE"
    date_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(\d{4}[./-]\d{1,2}[./-]\d{1,2})'
    text = re.sub(date_pattern, '#DATE', text)

    # Perform spell checking and correction
    blob = TextBlob(text)
    text = ' '.join(blob.correct() for blob in blob.words)

    return text


def preprocess_text(text):
    # Normalization
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove special characters
>>>>>>> main:legal_search_ai/unused/utils_old.py
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Convert to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


<<<<<<< HEAD:backend/utils_old.py

def load_raw_case_docs():
    # Set the files_directory variable to the current directory
    files_directory = "case_docs"

    # Read case details from text files and store in a list
    case_documents = []
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(files_directory, file_name), 'r') as f:
                content = f.read()
                case_documents.append(content)
    return case_documents


def load_case_docs():
    # Get the current directory path
    # current_directory = os.getcwd()

    # Set the files_directory variable to the current directory
    files_directory = "case_docs"

    # Read case details from text files and store in a list
    case_documents = []
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(files_directory, file_name), 'r') as f:
                content = f.read()
                content_list = content.split()
                case_documents.append(content_list)
    
    return case_documents


def extract_keywords_run1(query):
    # Step 1: Create an index of case documents

    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents
    
    # Create a dictionary from the case documents
    dictionary = corpora.Dictionary(case_documents)

    # Convert the case documents into a Bag-of-Words representation
    corpus = [dictionary.doc2bow(doc) for doc in case_documents]

    # Step 2: Extract keywords from the query

    # query = "..."  # Query text
    stop_words = set(stopwords.words('english'))  # Stopwords to filter out

    # Tokenize the query and remove stop words
    query_terms = [term for term in query.lower().split() if term not in stop_words]

    # Step 3: Extract keywords based on IDF values

    # Compute the IDF values for terms in the dictionary
    tfidf = TfidfModel(corpus)
    idf_values = {dictionary.get(id): value for id, value in tfidf.idfs.items()}

    # Sort the terms by IDF values in descending order
    sorted_terms = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)

    # Extract the top 50% of terms based on IDF values
    num_terms_to_use = len(sorted_terms) // 2
    keywords = [term for term, idf in sorted_terms[:num_terms_to_use]]

    return keywords


def extract_keywords_run2(query):
    # Step 1: Create an index of case documents

    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents

    # Create a dictionary from the case documents
    dictionary = corpora.Dictionary(case_documents)

    # Convert the case documents into a Bag-of-Words representation
    corpus = [dictionary.doc2bow(doc) for doc in case_documents]

    # Step 2: Extract keywords from the query

    # query = "..."  # Query text
    stop_words = set(stopwords.words('english'))  # Stopwords to filter out

    # Tokenize the query and remove stop words
    query_terms = [term for term in query.lower().split() if term not in stop_words]

    # Step 3: Extract keywords based on IDF values and entire query

    # Compute the IDF values for terms in the dictionary
    tfidf = TfidfModel(corpus)
    idf_values = {dictionary.get(id): value for id, value in tfidf.idfs.items()}

    # Extract the entire query as keywords
    keywords = query_terms + [query]

    return keywords


def extract_keywords_and_vectors_run3(query):
    # Step 1: Create an index of case documents

    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents

    # Create a dictionary from the case documents
    dictionary = corpora.Dictionary(case_documents)

    # Convert the case documents into a Bag-of-Words representation
    corpus = [dictionary.doc2bow(doc) for doc in case_documents]

    # Step 2: Extract keywords from the query

    # query = "..."  # Query text
    stop_words = set(stopwords.words('english'))  # Stopwords to filter out

    # Tokenize the query and remove stop words
    query_terms = [term for term in query.lower().split() if term not in stop_words]

    # Step 3: Extract keywords based on TF-IDF and word2vec

    # Compute the TF-IDF values for terms in the dictionary
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]

    # Extract the first 50% of the term and case documents of the TF-IDF for each query
    keywords = []
    for doc_bow, doc_tfidf in zip(corpus, tfidf_corpus):
        # Sort the terms by TF-IDF values in descending order
        sorted_terms = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)
        # Extract the first 50% of the terms and add to keywords
        keywords += [dictionary.get(term_id) for term_id, _ in sorted_terms[:len(sorted_terms)//2]]

    # Step 4: Use word2vec to represent the query and case documents

    # Train a word2vec model on the case documents
    word2vec_model = models.Word2Vec(case_documents)

    # Represent the query as word2vec vector
    query_vector = None

    # Calculate query vector
    try:
        query_vector = sum(word2vec_model.wv[term] for term in query_terms if term in word2vec_model.wv) / len(query_terms)
    except KeyError or ZeroDivisionError as e:
        # Handle KeyError here, e.g. by setting query_vector to a default value
        query_vector = None
        print(f"Error: {e} not found in word2vec model. Using default value for query vector.")

    return keywords, query_vector


def retrieve_documents_bm25(query_keywords):
    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents

    # Create a dictionary from the case documents
    dictionary = corpora.Dictionary(case_documents)

    # Convert the case documents into a Bag-of-Words representation
    corpus = [dictionary.doc2bow(doc) for doc in case_documents]

    # Step 4: Use BM25 to rank case documents based on relevance to query

    # Convert the query terms into a Bag-of-Words representation
    query_bow = dictionary.doc2bow(query_keywords)

    # Create the BM25 index for the corpus
    bm25_obj = BM25Okapi(corpus)

    # Get the average document length in the corpus
    # avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)

    # Compute BM25 scores for all case documents
    scores = bm25_obj.get_scores(query_bow)
    # scores = bm25_obj.get_scores(query_bow, avg_doc_len)

    # Step 5: Retrieve the top-ranked case documents

    # Sort the case documents by BM25 scores in descending order
    sorted_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Retrieve the top-ranked case documents
    top_docs_indices = [doc_idx for doc_idx, score in sorted_docs]
    top_documents = [case_documents[doc_idx] for doc_idx in top_docs_indices]

    return top_documents


def retrieve_documents_word2vec(query_keywords, query_vector):
    # Step 1: Create an index of case documents

    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents

    # Step 4: Use word2vec to represent the query and case documents

    # Train a word2vec model on the case documents
    word2vec_model = models.Word2Vec(case_documents)

    ## Represent the query and case documents as word2vec vectors
    
    query_vector = None
    # Calculate query vector
    try:
        query_vector = sum(word2vec_model.wv[term] for term in query_keywords if term in word2vec_model.wv) / len(query_keywords)
    except KeyError or ZeroDivisionError as e:
        # Handle KeyError here, e.g. by setting query_vector to a default value
        query_vector = None
        print(f"Error: {e} not found in word2vec model. Using default value for query vector.")
    
    case_documents_vectors = None
    # Calculate case documents vectors
    try:
        case_documents_vectors = [sum(word2vec_model.wv[term] for term in doc if term in word2vec_model.wv) / len(doc) for doc in case_documents if len(doc) > 0]
    except KeyError or ZeroDivisionError as e:
        # Handle KeyError here, e.g. by setting query_vector to a default value
        case_documents_vectors = None
        print(f"Error: {e} not found in word2vec model. Using default value for query vector.")
    
    
    # Step 5: Rank case documents based on Euclidean distance

    # Compute the Euclidean distances between the query vector and case documents vectors
    distances = euclidean_distances([query_vector], case_documents_vectors)[0]

    # Step 6: Retrieve the top-ranked case documents

    # Sort the case documents by distances in ascending order
    sorted_docs = sorted(enumerate(distances), key=lambda x: x[1])

    # Retrieve the top-ranked case documents
    top_docs_indices = [doc_idx for doc_idx, dist in sorted_docs]
    top_documents = [case_documents[doc_idx] for doc_idx in top_docs_indices]

    return top_documents


def combine_results(documents_bm25_run1, documents_bm25_run2, documents_word2vec_run3):
    # Combine the results from all three runs
    merged_documents = []

    # Convert document lists to sets for faster comparison
    documents_bm25_run1_tuples = [tuple(doc) for doc in documents_bm25_run1]
    set_documents_bm25_run1 = set(documents_bm25_run1_tuples)
    
    documents_bm25_run2_tuples = [tuple(doc) for doc in documents_bm25_run2]
    set_documents_bm25_run2 = set(documents_bm25_run2_tuples)
    
    documents_word2vec_run3_tuples = [tuple(doc) for doc in documents_word2vec_run3]
    set_documents_word2vec_run3 = set(documents_word2vec_run3_tuples)
    

    # Find common documents in Run 1 and Run 2
    common_documents_bm25 = set_documents_bm25_run1.intersection(set_documents_bm25_run2)

    # Find common documents in Run 1, Run 2, and Run 3
    common_documents = common_documents_bm25.intersection(set_documents_word2vec_run3)

    # Add documents from Run 1 that are not in common with Run 2 and Run 3
    for doc in documents_bm25_run1_tuples:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add documents from Run 2 that are not in common with Run 1 and Run 3
    for doc in documents_bm25_run2_tuples:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add documents from Run 3 that are not in common with Run 1 and Run 2
    for doc in documents_word2vec_run3_tuples:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add common documents from Run 1, Run 2, and Run 3
    for doc in common_documents:
        merged_documents.append(doc)

    return merged_documents



if __name__ == "__main__":
    pass
=======
def store_preprocessed_docs(documents):
    preprocessed_docs = [preprocess_text(text=doc) for doc in documents]

    with open('data/preprocessed_docs.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for doc in preprocessed_docs:
            writer.writerow([doc])
    
    return preprocessed_docs


def load_preprocessed_docs():
    preprocessed_docs = []

    with open('data/preprocessed_docs.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            preprocessed_docs.append(row[0])

    return preprocessed_docs


def store_tokenized_docs(documents):
    tokenized_docs = [word_tokenize(text=doc.lower()) for doc in documents]

    with open('data/tokenized_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(tokenized_docs)

    return tokenized_docs


def load_tokenized_docs():
    with open('data/tokenized_docs.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tokenized_docs = list(reader)
    return tokenized_docs


def load_corpus(tokenized_docs):
    # Create a dictionary of unique words from the tokenized documents
    dictionary = corpora.Dictionary(tokenized_docs)

    # Convert the tokenized documents into "bag of words" representation
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    return corpus

def store_word2vev_model(corpus):
    # Train a Word2Vec model on your text data
    embedding_size = 100
    model = Word2Vec(corpus, size=embedding_size, window=5, min_count=5, workers=4)

    # Save the trained model to a file
    model.save("data/word2vec_model.bin")
    return model


def load_word2vev_model():
    # Load the pre-trained Word2Vec model from a file
    model = Word2Vec.load("data/word2vec_model.bin")
    return model


def generate_embeddings(word2vev_model, normalized_query, documents):
    model = word2vev_model
    # Generate embeddings for the normalized query
    query_embedding = []
    for word in normalized_query.split():
        if word in model.wv:
            query_embedding.append(model.wv[word])

    # Generate embeddings for the documents in the dataset
    document_embeddings = []
    for document in documents:
        document_embedding = []
        for word in document.split():
            if word in model.wv:
                document_embedding.append(model.wv[word])
        document_embeddings.append(document_embedding)
    
    return query_embedding, document_embeddings


def store_TF_IDF_feature_vectors(preprocessed_query, preprocessed_docs):
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Use fit_transform to convert documents into TF-IDF feature vectors
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Use transform to convert queries into TF-IDF feature vectors
    query_tfidf_matrix = vectorizer.transform([preprocessed_query])

    # Save tfidf_matrix to disk in NPZ format
    save_npz('data/tfidf_matrix.npz', tfidf_matrix)

    # Save query_tfidf_matrix to disk in NPZ format
    save_npz('data/query_tfidf_matrix.npz', query_tfidf_matrix)

    return query_tfidf_matrix, tfidf_matrix


def load_TF_IDF_feature_vectors():
    tfidf_matrix = load_npz('data/tfidf_matrix.npz')
    query_tfidf_matrix = load_npz('data/query_tfidf_matrix.npz')
    return query_tfidf_matrix, tfidf_matrix


def compute_cosine_similarity_score(query_tfidf_matrix, tfidf_matrix):
    # Compute cosine similarity between queries and documents
    similarity_scores = cosine_similarity(query_tfidf_matrix, tfidf_matrix)
    return similarity_scores


def load_top_documents(similarity_scores, documents):
    # Number of top documents to retrieve
    num_top_docs = 5

    # # Loop through similarity scores for each query
    # for i in range(similarity_scores.shape[0]):
    #     # Get similarity scores for current query
    #     query_scores = similarity_scores[i]
    #     # Get indices that would sort the similarity scores in descending order
    #     sorted_indices = np.argsort(query_scores)[::-1]
    #     # Get the top num_top_docs document indices
    #     top_doc_indices = sorted_indices[:num_top_docs]
        
    # Get the indices of the top documents
    top_doc_indices = similarity_scores.argsort(axis=1)[:, ::-1][:, :num_top_docs]

    # Retrieve the top documents from the original document set
    top_docs = [documents[i] for i in top_doc_indices.flatten()]
    return top_docs
>>>>>>> main:legal_search_ai/unused/utils_old.py
