import nltk
from gensim import corpora, models
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import euclidean_distances
import os
from rank_bm25 import BM25Okapi
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz, csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import PyPDF2
from functions import *


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def load_raw_docs(doc_type):
    if doc_type == "Case":
        files_directory = "data/case_docs"
    elif doc_type == "Statute":
        files_directory = "data/statute_docs/IT_ACT_2000"
    
    # Read case details from text files and store in a list
    documents = []
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(files_directory, file_name), 'r') as f:
                content = f.read()
                documents.append(content)
    
    return documents


def store_preprocessed_docs(documents, doc_type):
    preprocessed_docs = [preprocess_text(text=doc) for doc in documents]

    if doc_type == "Case":
        docs_path = 'data/temp/case/preprocessed_docs.csv'
    elif doc_type == "Statute":
        docs_path = 'data/temp/statute/preprocessed_docs.csv'

    with open(docs_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for doc in preprocessed_docs:
            writer.writerow([doc])
    
    return preprocessed_docs


def load_preprocessed_docs(doc_type):
    preprocessed_docs = []

    if doc_type == "Case":
        docs_path = 'data/temp/case/preprocessed_docs.csv'
    elif doc_type == "Statute":
        docs_path = 'data/temp/statute/preprocessed_docs.csv'

    with open(docs_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            preprocessed_docs.append(row[0])

    return preprocessed_docs


def store_doc_tfidf_matrix(preprocessed_query, preprocessed_docs, doc_type):
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Use fit_transform to convert documents into TF-IDF feature vectors
    doc_tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Use transform to convert queries into TF-IDF feature vectors
    query_tfidf_matrix = vectorizer.transform([preprocessed_query])

    if doc_type == "Case":
        file_path = 'data/temp/case/doc_tfidf_matrix.npz'
    elif doc_type == "Statute":
        file_path = 'data/temp/statute/doc_tfidf_matrix.npz'

    # Save the doc_tfidf_matrix to a file
    np.savez(file_path, data=doc_tfidf_matrix.data, indices=doc_tfidf_matrix.indices,
         indptr=doc_tfidf_matrix.indptr, shape=doc_tfidf_matrix.shape)
    
    return query_tfidf_matrix, doc_tfidf_matrix


def load_doc_tfidf_matrix(preprocessed_query, doc_type):
    if doc_type == "Case":
        file_path = 'data/temp/case/doc_tfidf_matrix.npz'
    elif doc_type == "Statute":
        file_path = 'data/temp/statute/doc_tfidf_matrix.npz'
    
    # Load the doc_tfidf_matrix from the file
    data = np.load(file_path)
    doc_tfidf_matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                                                    shape=data['shape'])

    # Create a new TfidfVectorizer object and manually set the attributes
    vectorizer_new = TfidfVectorizer()
    vectorizer_new.idf_ = data['idf']
    vectorizer_new.vocabulary_ = data['vocab']
    vectorizer_new.n_features_ = data['n_features']

    # Use the precomputed doc_tfidf_matrix as input for the new TfidfVectorizer
    # Use transform to convert queries into TF-IDF feature vectors
    query_tfidf_matrix = vectorizer_new.transform([preprocessed_query])

    return query_tfidf_matrix, doc_tfidf_matrix


def compute_cosine_similarity_score(query_tfidf_matrix, doc_tfidf_matrix):
    # Compute cosine similarity between queries and documents
    similarity_scores = cosine_similarity(query_tfidf_matrix, doc_tfidf_matrix)
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


if __name__ == "__main__":
    pass