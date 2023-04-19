from flask import Flask, request, jsonify
# from utils import extract_keywords_run1, extract_keywords_run2, extract_keywords_and_vectors_run3, retrieve_documents_bm25, retrieve_documents_word2vec, combine_results, load_raw_case_docs
# from utils_new import *
from new import run1


# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# Set up logging
logging.basicConfig(filename='logs.log', level=logging.DEBUG)


app = Flask(__name__)

# API endpoint for handling search requests
@app.route('/api/search', methods=['GET'])
def search():
    # return "hello"
    query = request.args.get('query')
    task = request.args.get('task')

    search_results = run1(query)

    return jsonify(search_results)



    # # Placeholder search results for demonstration purposes
    # search_results = [
    #     {
    #         'docement_id': 1,
    #         'content': {
    #             'title': 'Result 1 Title',
    #             'summary': 'Result 1 Summary'
    #         } 
    #     },
    #     {
    #         'docement_id': 2,
    #         'content': {
    #             'title': 'Result 2 Title',
    #             'summary': 'Result 2 Summary'
    #         }
    #     },
    #     {
    #         'docement_id': 3,
    #         'content': {
    #             'title': 'Result 3 Title',
    #             'summary': 'Result 3 Summary'
    #         }                
    #     }
    # ]
    
    


    # query = request.args.get('query')
    # task = request.args.get('task')

    # documents = load_raw_case_docs()

    # preprocessed_docs = load_preprocessed_docs(documents=documents)
    # preprocessed_query = preprocess_text(text=query)

    # # tokenized_docs = load_tokenized_docs(preprocessed_docs)

    # # corpus = load_corpus(tokenized_docs=tokenized_docs)
    
    # try:
    #     query_tfidf_matrix, tfidf_matrix = load_TF_IDF_feature_vectors()
    # except:
    #     query_tfidf_matrix, tfidf_matrix = store_TF_IDF_feature_vectors(preprocessed_query, preprocessed_docs)

    # similarity_scores = compute_cosine_similarity_score(query_tfidf_matrix, tfidf_matrix)

    # top_docs = load_top_documents(similarity_scores, documents)

    # # Create a list of dictionaries to represent the top documents
    # top_documents_json = []
    # for i, document in enumerate(top_docs):
    #     doc_dict = {"document_id": i+1, "content": document}  # Create a dictionary for each document
    #     top_documents_json.append(doc_dict)  # Add the dictionary to the list

    # return jsonify(top_documents_json)
    

    

    #############################  v1  ##################################################

    # # Process search query and task, and retrieve search results
    # # Replace this with your actual backend logic

    # ### For Task 1 ###
    # # Extract keywords from the query for Run 1
    # query_keywords_run1 = extract_keywords_run1(query)
    
    # # Extract keywords from the query for Run 2
    # query_keywords_run2 = extract_keywords_run2(query)
    
    # # Extract keywords and compute word2vec vectors for Run 3
    # query_keywords_run3, query_vector_run3 = extract_keywords_and_vectors_run3(query)
    # # Retrieve documents using BM25 for Run 1 and Run 2
    # documents_bm25_run1 = retrieve_documents_bm25(query_keywords_run1)
    # documents_bm25_run2 = retrieve_documents_bm25(query_keywords_run2)
    
    # # Retrieve documents using word2vec and Euclidean distance for Run 3
    # documents_word2vec_run3 = retrieve_documents_word2vec(query_keywords_run3, query_vector_run3)
    # # Combine the results from all three runs
    # merged_documents = combine_results(documents_bm25_run1, documents_bm25_run2, documents_word2vec_run3)
    
    # # case_documents = load_raw_case_docs()
    
    # # # Retrieve the top 3 ranked case documents
    # # # Assuming merged_documents is a list of tuples with multiple values per item
    # # top_docs_indices = [doc_idx for doc_idx, *_ in merged_documents[:3]]  # Get indices of top 3 documents
    # # top_documents = [case_documents[doc_idx] for doc_idx in top_docs_indices]  # Get top 3 documents
    # top_documents = merged_documents
    
    # # Create a list of dictionaries to represent the top documents
    # top_documents_json = []
    # for i, document in enumerate(top_documents):
    #     doc_dict = {"document_id": i+1, "content": document}  # Create a dictionary for each document
    #     top_documents_json.append(doc_dict)  # Add the dictionary to the list

    # # Create a JSON object
    # # json_obj = json.dumps(top_documents_json, indent=4)  # Convert the list of dictionaries to JSON

    # # print(json_obj)  # Print the JSON object

    # search_results = top_documents_json

    # # Placeholder search results for demonstration purposes
    # # search_results = [
    # #     {
    # #         'id': 1,
    # #         'title': 'Result 1 Title',
    # #         'summary': 'Result 1 Summary'
    # #     },
    # #     {
    # #         'id': 2,
    # #         'title': 'Result 2 Title',
    # #         'summary': 'Result 2 Summary'
    # #     },
    # #     {
    # #         'id': 3,
    # #         'title': 'Result 3 Title',
    # #         'summary': 'Result 3 Summary'
    # #     }
    # # ]
    
    # return jsonify(search_results)


    ###################################  v2   ##############################################

    # # Load and preprocess documents and queries
    # documents = load_raw_case_docs()  # List of documents
    # queries = [query]  # List of queries

    # stop_words = set(stopwords.words("english"))
    # ps = PorterStemmer()

    # # Perform preprocessing on documents and queries
    # def preprocess(text):
    #     tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    #     tokens = [ps.stem(token) for token in tokens if token.isalnum()]  # Stemming
    #     tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    #     return " ".join(tokens)

    # preprocessed_docs = [preprocess(doc) for doc in documents]
    # preprocessed_queries = [preprocess(query) for query in queries]

    # # Feature extraction using TF-IDF
    # vectorizer = TfidfVectorizer()
    # tfidf_docs = vectorizer.fit_transform(preprocessed_docs)
    # tfidf_queries = vectorizer.transform(preprocessed_queries)

    # # Similarity measure using cosine similarity
    # similarity_scores = cosine_similarity(tfidf_queries, tfidf_docs)

    # # Ranking and retrieval
    # num_top_docs = 5  # Number of top documents to retrieve for each query

    # for i, query in enumerate(queries):
    #     print(f"Query {i+1}: {query}")
    #     top_doc_indices = similarity_scores[i].argsort()[::-1][:num_top_docs]  # Sort by similarity and get top doc indices
    #     top_docs = [documents[idx] for idx in top_doc_indices]
    #     print("Top Documents:")
    #     for j, doc in enumerate(top_docs):
    #         print(f"Document {j+1}: {doc}")
    #     print("\n")

    # # Create a list of dictionaries to represent the top documents
    # top_documents_json = []
    # for i, document in enumerate(top_docs):
    #     doc_dict = {"document_id": i+1, "content": document}  # Create a dictionary for each document
    #     top_documents_json.append(doc_dict)  # Add the dictionary to the list

    # return jsonify(top_documents_json)






if __name__ == '__main__':
    app.run(debug=True)
