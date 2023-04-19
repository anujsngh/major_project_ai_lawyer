import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from unicodedata import normalize
from gensim import corpora, models
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import euclidean_distances
import os
from rank_bm25 import BM25Okapi
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