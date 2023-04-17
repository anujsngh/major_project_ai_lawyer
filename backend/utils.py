from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import TfidfModel, bm25model
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import euclidean_distances
import os


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
                case_documents.append(content)
    
    # return case_documents
    return ["This is a document about cats.",
    "This document is about dogs.",
    "This document is about cats and dogs."]


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
    query_vector = sum(word2vec_model[term] for term in query_terms) / len(query_terms)

    return keywords, query_vector


def retrieve_documents_bm25(query_keywords):
    # Step 1: Create an index of case documents

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
    bm25_obj = bm25model(corpus)

    # Get the average document length in the corpus
    avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)

    # Compute BM25 scores for all case documents
    # scores = bm25.get_scores(query_bow)
    scores = bm25_obj.get_scores(query_bow, avg_doc_len)

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

    # Represent the query and case documents as word2vec vectors
    query_vector = sum(word2vec_model[term] for term in query_keywords) / len(query_keywords)
    case_documents_vectors = [sum(word2vec_model[term] for term in doc) / len(doc) for doc in case_documents]

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
    # Step 1: Create an index of case documents

    # Load the case documents into a list or a corpus object
    case_documents = load_case_docs()  # List of case documents
    
    # Combine the results from all three runs
    merged_documents = []

    # Convert document lists to sets for faster comparison
    set_documents_bm25_run1 = set(documents_bm25_run1)
    set_documents_bm25_run2 = set(documents_bm25_run2)
    set_documents_word2vec_run3 = set(documents_word2vec_run3)

    # Find common documents in Run 1 and Run 2
    common_documents_bm25 = set_documents_bm25_run1.intersection(set_documents_bm25_run2)

    # Find common documents in Run 1, Run 2, and Run 3
    common_documents = common_documents_bm25.intersection(set_documents_word2vec_run3)

    # Add documents from Run 1 that are not in common with Run 2 and Run 3
    for doc in documents_bm25_run1:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add documents from Run 2 that are not in common with Run 1 and Run 3
    for doc in documents_bm25_run2:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add documents from Run 3 that are not in common with Run 1 and Run 2
    for doc in documents_word2vec_run3:
        if doc not in common_documents:
            merged_documents.append(doc)

    # Add common documents from Run 1, Run 2, and Run 3
    for doc in common_documents:
        merged_documents.append(doc)

    return merged_documents



if __name__ == "__main__":
    print(load_case_docs())
